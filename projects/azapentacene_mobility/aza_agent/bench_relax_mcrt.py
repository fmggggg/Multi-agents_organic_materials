"""Effect of the quick relaxation on the objective (2026-11-26). Clari env.
For each molecule x method in bench_relax results: lowest-energy crystal before (first 50/Z draws, campaign single
points) and after relaxation, rank changes, volume change, and MCRT mobility of the 5 lowest relaxed + unrelaxed.
usage: python -m aza_agent.bench_relax_mcrt BENCH_DIR"""
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .workers.client import start_mcrt

EV = 96.485332
bench = Path(sys.argv[1])
rows = [json.loads(l) for l in open(bench / "results.jsonl")]
mols = {r["tag"]: r for r in rows if r["kind"] == "molecule"}
groups = defaultdict(list)
for r in rows:
    if r["kind"] == "crystal" and r["method"].startswith("pool"):
        groups[(r["tag"], r["method"])].append(r)
work = bench / "mcrt_eval"
cifdir = work / "cifs"
cifdir.mkdir(parents=True, exist_ok=True)
jobs, recs = {}, []
for (tag, m), rs in sorted(groups.items()):
    camp, mol_id = tag.rsplit("_", 1)
    camp_dir = Path("campaigns") / camp
    for r in rs:
        r["e0"] = r["E0_campaign_eV"] / r["Z"] * EV
        r["e1"] = r["E_eV"] / r["Z"] * EV
    e0 = np.array([r["e0"] for r in rs]); e1 = np.array([r["e1"] for r in rs])
    by0 = sorted(rs, key=lambda r: r["e0"]); by1 = sorted(rs, key=lambda r: r["e1"])
    rank0 = {r["crystal_id"]: k + 1 for k, r in enumerate(by0)}
    rank1 = {r["crystal_id"]: k + 1 for k, r in enumerate(by1)}
    rec = dict(tag=tag, method=m, n=len(rs), spearman_E0_E1=float(spearmanr(e0, e1)[0]),
               unrelaxed_min=by0[0]["crystal_id"], its_relaxed_rank=rank1[by0[0]["crystal_id"]],
               relaxed_min=by1[0]["crystal_id"], its_unrelaxed_rank=rank0[by1[0]["crystal_id"]],
               n_within_7kJ_relaxed=int(np.sum(e1 - e1.min() <= 7.2)),
               V_ratio_median=float(np.median([r["V"] / r["V0"] for r in rs])),
               V_ratio_relaxed_min=by1[0]["V"] / by1[0]["V0"], wall_s=rs[0]["pool_wall_seconds"])
    zf = zipfile.ZipFile(bench / f"{tag}_{m}.zip")
    for k, r in enumerate(by1[:5]):
        cid = f"{tag}__{m}__R{k + 1}__{r['crystal_id']}"
        (cifdir / f"{cid}.cif").write_bytes(zf.read(r["crystal_id"] + ".cif"))
        jobs[cid] = (len(recs), f"relaxed_R{k + 1}")
    raw = zipfile.ZipFile(camp_dir / "structures" / mol_id / "raw_cifs.zip")
    cid = f"{tag}__{m}__U1__{by0[0]['crystal_id']}"
    (cifdir / f"{cid}.cif").write_bytes(raw.read(by0[0]["crystal_id"] + ".cif"))
    jobs[cid] = (len(recs), "unrelaxed_U1")
    recs.append(rec)
mc = start_mcrt(work / "logs", gpu="0")
res = mc.call(cmd="predict", cifs=[str(cifdir / f"{c}.cif") for c in jobs], work_dir=str(work / "tmp"))
mc.close()
for x in res["results"]:
    i, key = jobs[x["cif_id"]]
    recs[i]["mu_" + key] = round(x["mobility"], 3)
for x in res.get("missing", []):
    print("MCRT missing", x)
with open(bench / "relax_effect.csv", "w", newline="") as h:
    keys = sorted({k for r in recs for k in r}, key=lambda k: (not k in ("tag", "method"), k))
    w = csv.DictWriter(h, keys)
    w.writeheader()
    w.writerows(recs)
for r in recs:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
