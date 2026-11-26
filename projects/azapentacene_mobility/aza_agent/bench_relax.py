"""Timing / effect benchmark for a quick MACE-OFF23-small relaxation of Clari draws (2026-11-25).

Runs in the BOMLIP_CSP env. Input: demo-campaign molecules, first N draws per Z (s000..), QC-passed draws only
(the QC status is taken from the campaign's crystals.csv; skipped draws still count in n_draws).
Methods (all start from the identical unrelaxed CIFs, fmax target 0.05 eV/A, step cap --steps):
  ase_fixed   ASE LBFGS(maxstep 0.08), fixed cell, one structure at a time   (subset --ase-n)
  ase_cell    ASE LBFGS on FrechetCellFilter, cell + positions, one at a time  (subset --ase-n)
  pool_fixed / pool_cell [_f32]  the same ASE protocols on ALL draws, --workers processes sharing the GPU
  batch_fixed BOMLIP-CSP batched BFGS (OptimizableBatch), all draws of a molecule in one batch
  batch_cell  BOMLIP-CSP batched BFGS (OptimizableUnitCellBatch, P = 0), all draws in one batch
Every relaxed structure gets: energy, max force, volume, severe-overlap check and bond-graph integrity check.
usage: bench_relax.py --out DIR --mol CAMPAIGN_DIR:MOL_ID [...] [--per-z 50] [--steps 100]
"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
EV_TO_KJMOL = 96.485332


def load_draws(camp: Path, mol_id: str, per_z: int):
    import csv
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    smiles = next(r["smiles"] for r in csv.DictReader(open(camp / "molecules.csv")) if r["mol_id"] == mol_id)
    status = {r["crystal_id"]: r for r in csv.DictReader(open(camp / "crystals.csv")) if r["mol_id"] == mol_id}
    zf = zipfile.ZipFile(camp / "structures" / mol_id / "raw_cifs.zip")
    draws = []
    for cid, r in sorted(status.items()):
        if int(r["sample_idx"]) >= per_z:
            continue
        d = dict(crystal_id=cid, Z=int(r["Z"]), qc_status=r["status"])
        if r["status"] == "scored_energy":
            st = Structure.from_str(zf.read(cid + ".cif").decode(), fmt="cif")
            d["atoms"] = AseAtomsAdaptor.get_atoms(st)
            d["E0_eV_campaign"] = float(r["E_total_eV"])
        draws.append(d)
    return smiles, draws


def post(atoms, d, ref, qc):
    ov = qc.severe_overlap_check(atoms, factor=qc.OVERLAP_FACTOR)
    integ = qc.integrity_check(atoms, ref, d["Z"])
    return dict(V=float(atoms.get_volume()), overlap_ok=bool(ov["accepted"]), intact=bool(integ["intact"]),
                integrity_reason=integ["reason"])


def run_ase(draws, calc, steps, cell, ref, qc):
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS
    out = []
    for d in draws:
        at = d["atoms"].copy()
        at.calc = calc
        t0 = time.perf_counter()
        e0 = float(at.get_potential_energy())
        obj = FrechetCellFilter(at, scalar_pressure=0.0) if cell else at
        opt = LBFGS(obj, maxstep=0.08, logfile=None)
        conv = bool(opt.run(fmax=0.05, steps=steps))
        e1 = float(at.get_potential_energy())
        fmax = float(np.linalg.norm(at.get_forces(), axis=1).max())
        sec = time.perf_counter() - t0
        out.append(dict(crystal_id=d["crystal_id"], Z=d["Z"], E0_eV=e0, E_eV=e1, fmax=fmax, steps=int(opt.nsteps),
                        converged=conv, seconds=sec, V0=float(d["atoms"].get_volume()), **post(at, d, ref, qc)))
        out[-1]["_atoms"] = at
    return out


_W = {}


def _pool_init(dtype):
    import torch
    from mace.calculators import mace_off
    from aza_agent import qc
    torch.set_num_threads(2)
    _W["calc"] = mace_off(model="small", device="cuda", default_dtype=dtype)
    _W["qc"] = qc


def _pool_task(job):
    from ase.io import write
    d, steps, cell, smiles = job
    qc = _W["qc"]
    r = run_ase([d], _W["calc"], steps, cell, qc.reference_graph(smiles), qc)[0]
    b = io.BytesIO()
    write(b, r.pop("_atoms"), format="cif")
    r["_cif"] = b.getvalue().decode()
    return r


def run_pool(draws, steps, cell, smiles, workers, dtype):
    """Independent ASE relaxations in `workers` spawned processes sharing the GPU (2 CPU threads each)."""
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers, initializer=_pool_init, initargs=(dtype,)) as pool:
        t_ready = time.perf_counter()
        res = pool.map(_pool_task, [(d, steps, cell, smiles) for d in draws], chunksize=1)
    wall = time.perf_counter() - t0
    for r in res:
        r["seconds_cpu"] = r["seconds"]
        r["seconds"] = wall / len(res)       # wall-clock share, incl. worker start-up
        r["pool_wall_seconds"] = wall
        r["pool_startup_seconds"] = t_ready - t0
    return res


def run_batch(draws, calc, steps, cell, ref, qc, device):
    import torch
    import batchopt
    from batchopt.pbc_graph_legacy import radius_graph_pbc_mem_effi
    # the CUDA neighbour-list extension needs a JIT build (nvcc + ninja, absent here); use BOMLIP's pure-torch version
    batchopt.radius_graph_pbc_cuda = radius_graph_pbc_mem_effi
    # the fork's convert_batch omits data["shifts"] (= unit_shifts @ cell) that its models.prepare_graph reads
    from mace.calculators.mace import MACECalculator
    if not getattr(MACECalculator, "_aza_shift_patch", False):
        _orig = MACECalculator.convert_batch

        def _convert(self, gbatch):
            cb = _orig(self, gbatch)
            cells = cb.cell.view(-1, 3, 3)
            gid = cb.batch[cb.edge_index[0]]
            cb.shifts = torch.einsum("ei,eij->ej", cb.unit_shifts.to(cells.dtype), cells[gid])
            return cb
        MACECalculator.convert_batch = _convert
        MACECalculator._aza_shift_patch = True
    from batchopt.atoms_to_graphs import AtomsToGraphs
    from batchopt.relaxation import OptimizableBatch, OptimizableUnitCellBatch
    from batchopt.relaxation.optimizers import BFGS
    from batchopt.utils import data_list_collater
    a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g = data_list_collater([a2g.convert(d["atoms"].copy()) for d in draws]).to(device)
    ob = (OptimizableUnitCellBatch(g, trainer=calc, numpy=False, scalar_pressure=0.0) if cell
          else OptimizableBatch(g, trainer=calc, numpy=False))
    e0 = ob.get_potential_energies().detach().cpu().numpy().ravel().copy()
    opt = BFGS(ob, alpha=70.0, maxstep=0.2)
    opt.run(0.05, steps)
    fmax = ob.get_max_forces(apply_constraint=True).detach().cpu().numpy().ravel()
    atoms_out = ob.get_atoms_list()
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    out = []
    for d, at, a, f in zip(draws, atoms_out, e0, fmax):
        out.append(dict(crystal_id=d["crystal_id"], Z=d["Z"], E0_eV=float(a), fmax=float(f), steps=int(opt.nsteps),
                        seconds=sec / len(draws), batch_seconds=sec, V0=float(d["atoms"].get_volume()),
                        **post(at, d, ref, qc)))
        out[-1]["_atoms"] = at
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mol", nargs="+", required=True, help="CAMPAIGN_DIR:MOL_ID")
    ap.add_argument("--per-z", type=int, default=50)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ase-n", type=int, default=8, help="structures per molecule for the slow ASE baselines")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["batch_cell", "batch_fixed", "ase_cell", "ase_fixed"])
    a = ap.parse_args()
    import torch
    from mace.calculators import mace_off
    from ase.io import write
    from aza_agent import qc
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    calc = mace_off(model="small", device=device, default_dtype="float64")
    print("MACE r_max", float(calc.models[0].r_max), flush=True)
    jl = (a.out / "results.jsonl").open("a")
    for spec in a.mol:
        camp, mol_id = spec.rsplit(":", 1)
        camp = Path(camp)
        smiles, draws = load_draws(camp, mol_id, a.per_z)
        ok = [d for d in draws if "atoms" in d]
        ref = qc.reference_graph(smiles)
        tag = f"{camp.name}_{mol_id}"
        print(f"{tag}: {len(draws)} draws, {len(ok)} QC-ok, atoms/str {[len(d['atoms']) for d in ok[:1]]}", flush=True)
        jl.write(json.dumps(dict(kind="molecule", tag=tag, smiles=smiles, n_draws=len(draws), n_qc_ok=len(ok),
                                 qc_counts={s: sum(d["qc_status"] == s for d in draws)
                                            for s in {d["qc_status"] for d in draws}})) + "\n")
        for m in a.methods:
            sub = ok[: a.ase_n] if m.startswith("ase") else ok
            if m.startswith("pool"):
                res = run_pool(ok, a.steps, m.startswith("pool_cell"), smiles, a.workers,
                               "float32" if m.endswith("_f32") else "float64")
                zpath = a.out / f"{tag}_{m}.zip"
                with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                    for r in res:
                        zf.writestr(r["crystal_id"] + ".cif", r.pop("_cif"))
                with zipfile.ZipFile(zpath) as zf:
                    assert zf.testzip() is None and len(zf.namelist()) == len(res)
            elif m.startswith("ase"):
                res = run_ase(sub, calc, a.steps, m == "ase_cell", ref, qc)
                for r in res:
                    r.pop("_atoms")
            else:
                res = run_batch(sub, calc, a.steps, m == "batch_cell", ref, qc, device)
                # final energies with the standard ASE calculator (same model) = consistency check of batched energies
                zpath = a.out / f"{tag}_{m}.zip"
                with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                    for r in res:
                        at = r.pop("_atoms")
                        at.calc = calc
                        r["E_eV"] = float(at.get_potential_energy())
                        b = io.BytesIO()
                        write(b, at, format="cif")
                        zf.writestr(r["crystal_id"] + ".cif", b.getvalue())
                with zipfile.ZipFile(zpath) as zf:
                    assert zf.testzip() is None and len(zf.namelist()) == len(res)
            camp_e0 = {d["crystal_id"]: d["E0_eV_campaign"] for d in ok}
            for r in res:
                r.update(kind="crystal", tag=tag, method=m, steps_cap=a.steps, E0_campaign_eV=camp_e0[r["crystal_id"]])
                jl.write(json.dumps(r) + "\n")
            jl.flush()
            sec = sum(r["seconds"] for r in res)
            dE = [(r["E_eV"] - r["E0_eV"]) / r["Z"] * EV_TO_KJMOL for r in res]
            print(f"  {m:11s} n={len(res):3d} total {sec:7.1f}s  per-str {sec / len(res):6.2f}s  "
                  f"steps {np.median([r['steps'] for r in res]):.0f}  median dE/mol {np.median(dE):7.1f} kJ/mol  "
                  f"intact {sum(r['intact'] for r in res)}/{len(res)}  overlap_ok {sum(r['overlap_ok'] for r in res)}  "
                  f"median fmax {np.median([r['fmax'] for r in res]):.3f}  "
                  f"E0 vs campaign max|d| {max(abs(r['E0_eV'] - r['E0_campaign_eV']) for r in res):.2e} eV",
                  flush=True)
    jl.close()


if __name__ == "__main__":
    main()
