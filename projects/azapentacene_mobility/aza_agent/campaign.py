"""Agent loop: read table -> select new molecule -> tool call (Clari -> MACE -> MCRT) -> write table.

The results table (molecules.csv + crystals.csv) is the agent's only memory between turns, so a
stopped campaign resumes from the files. Every proposal, including rejected ones, goes to proposals.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from rdkit import Chem

from . import space
from .pipeline import PipelineConfig, append_csv

MOL_FIELDS = ["mol_id", "iteration", "policy", "smiles", "label", "skeleton", "sites", "n_N", "formula", "status",
              "reason", "mu_lowestE", "E_latt_min_kJmol", "E_inter_best_kJmol", "strain_best_kJmol", "density_best", "best_Z", "best_crystal_id",
              "n_draws", "n_qc_ok", "n_overlap", "n_broken", "n_parse_error", "n_energy", "n_window",
              "n_window_scored", "mu_mean_window", "mu_max_window", "sec_clari", "sec_qc", "sec_mace", "sec_mcrt",
              "relaxed", "n_relax_fail", "sec_relax",
              "graph_key", "aroma_key", "rationale", "timestamp"]
PROP_FIELDS = ["iteration", "policy", "input", "smiles", "label", "ok", "reason", "action", "timestamp"]

DEFAULT_GOAL = ("From the given chemical space, find the molecule with maximum charge mobility, "
                "using the available tools.")

SYSTEM = """You are a computational materials chemist running a closed-loop search for organic semiconductors.

RESEARCHER GOAL: {goal}

CHEMICAL SPACE (aza-substituted five-ring catacondensed acenes). Every candidate is one of 10 unbranched
fused-hexagon skeletons, with nitrogen replacing one or more of its 14 C-H positions. Skeleton codes give the
annelation of the three inner rings (L linear, A/B angular with the same/opposite turn). C-H sites are numbered
1..14 around the perimeter; the atom-map numbers below show where each site is:
{space}
Rules enforced by the tools: 22 C/N ring atoms, all aromatic, neutral, at least 1 N; molecules in the
reference dataset ({n_known} known azapentacenes the mobility model was trained around) and molecules already
tested in this campaign are rejected as duplicates (symmetry-equivalent site sets are the same molecule).
Known molecules have 1-12 N (most 4-8).

TOOLS. evaluate_candidate runs the full (expensive) chain and consumes one unit of budget:
  Clari-H crystal structure prediction ({draws} trial packings: Z in {zs}, {spz} each) -> severe-overlap and
  molecular-integrity QC -> {relax_text} ->
  MCRT mobility (cm^2 V^-1 s^-1) of the lowest-energy packing, plus the other packings within {win} kJ/mol.
  Energies per molecule: E_latt = E_crys/Z - E(relaxed gas molecule); it splits into E_inter (packing, frozen
  geometry) + strain (distortion of the generated molecule vs. the relaxed one). Packings are ranked by {rank}.
The OBJECTIVE for a molecule is mu_lowestE = predicted mobility of its lowest-energy predicted crystal.
check_candidate is free: use it to validate ideas and avoid duplicates. get_crystals shows crystal-level rows.

HOW TO WORK, each turn: read the results table you are given, reason about what structural features
(skeleton, number and position of N, symmetry) correlate with high mu_lowestE, then evaluate exactly ONE new
molecule. Balance exploitation (close variants of the best molecules) against exploration (skeletons / N
patterns not yet tested), especially early. Predictions are noisy (unrelaxed packings, limited sampling), so
do not over-trust a single number. In evaluate_candidate give a concise rationale and an updated research
notebook (hypotheses, what to try next); the notebook is shown to you again next turn."""

TOOLS = [
    dict(name="check_candidate",
         description="Free validity + novelty check. Give either skeleton+sites or a SMILES.",
         parameters=dict(type="object", properties=dict(
             skeleton=dict(type="string", description="skeleton code, e.g. LLL"),
             sites=dict(type="array", items=dict(type="integer"), description="aza sites 1..14"),
             smiles=dict(type="string")), additionalProperties=False)),
    dict(name="get_crystals",
         description="Crystal-level results (Z, energy, density, cell, mobility, QC status) for one evaluated molecule.",
         parameters=dict(type="object", properties=dict(mol_id=dict(type="string"),
                                                        max_rows=dict(type="integer")),
                         required=["mol_id"], additionalProperties=False)),
    dict(name="evaluate_candidate",
         description="Run Clari -> MACE -> MCRT on ONE new molecule (consumes budget, ends this turn).",
         parameters=dict(type="object", properties=dict(
             skeleton=dict(type="string"), sites=dict(type="array", items=dict(type="integer")),
             smiles=dict(type="string"),
             rationale=dict(type="string", description="why this molecule, in 1-3 sentences"),
             notebook=dict(type="string", description="updated research notebook (<= 2000 characters)")),
             required=["rationale", "notebook"], additionalProperties=False)),
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


class Campaign:
    def __init__(self, d: Path):
        self.d = d
        self.cfg = json.loads((d / "campaign.json").read_text())
        self.pcfg = PipelineConfig(**{**self.cfg["pipeline"], "z_values": tuple(self.cfg["pipeline"]["z_values"])})

    # ---------------------------------------------------------------- table views
    def molecules(self):
        return read_csv(self.d / "molecules.csv")

    def tested_keys(self):
        rows = self.molecules()
        return {r["graph_key"] for r in rows}, {r["aroma_key"] for r in rows}

    def table_digest(self, max_rows=300) -> str:
        rows = self.molecules()
        if not rows:
            return "(no molecules evaluated yet)"
        ok = [r for r in rows if r["status"] == "evaluated"]
        ok.sort(key=lambda r: -float(r["mu_lowestE"]))
        lines = ["rank | mol_id | label (skeleton:sites) | n_N | mu_lowestE | mu_mean_win (n_win) | E_latt_min kJ/mol | "
                 "E_inter kJ/mol | strain kJ/mol | density | best_Z | QC ok/draws | iter"]
        for k, r in enumerate(ok[:max_rows]):
            lines.append(f"{k+1} | {r['mol_id']} | {r['label']} | {r['n_N']} | {fmt(r['mu_lowestE'])} | "
                         f"{fmt(r['mu_mean_window'])} ({r['n_window']}) | {fmt(r['E_latt_min_kJmol'], 1)} | "
                         f"{fmt(r['E_inter_best_kJmol'], 1)} | {fmt(r['strain_best_kJmol'], 1)} | "
                         f"{fmt(r['density_best'], 3)} | {r['best_Z']} | {r['n_qc_ok']}/{r['n_draws']} | {r['iteration']}")
        bad = [r for r in rows if r["status"] != "evaluated"]
        if bad:
            lines.append("\nFailed evaluations:")
            lines += [f"{r['mol_id']} | {r['label']} | {r['reason']}" for r in bad]
        return "\n".join(lines)

    # ---------------------------------------------------------------- tools
    def resolve(self, args) -> dict:
        if args.get("smiles"):
            smi = args["smiles"]
        elif args.get("skeleton"):
            try:
                smi = space.build_from_sites(args["skeleton"].strip().upper(), args.get("sites") or [])
            except Exception as e:
                return dict(ok=False, reason=f"build_failed: {e}", input=json.dumps(args))
        else:
            return dict(ok=False, reason="give skeleton+sites or smiles", input=json.dumps(args))
        gk, ak = self.tested_keys()
        r = space.check(smi, gk, ak, min_n=self.cfg.get("min_n", 1))
        r["input"] = json.dumps({k: v for k, v in args.items() if k in ("skeleton", "sites", "smiles")})
        return r

    def novel_neighbours(self, r, n=6) -> list[str]:
        """Novel, valid site sets one edit away (add / remove / move one N) from a rejected skeleton:sites proposal."""
        sk, sites = r.get("skeleton"), r.get("sites")
        if not sk or not sites:
            return []
        gk, ak = self.tested_keys()
        cur, seen, out = sorted(sites), set(), []
        cands = [cur + [x] for x in range(1, 15) if x not in cur] + [[y for y in cur if y != x] for x in cur]
        cands += [sorted([y for y in cur if y != x] + [x + d]) for x in cur for d in (-1, 1)
                  if 1 <= x + d <= 14 and x + d not in cur]
        for c in cands:
            key = (sk, tuple(sorted(c)))
            if not c or key in seen:
                continue
            seen.add(key)
            try:
                chk = space.check(space.build_from_sites(sk, c), gk, ak, min_n=self.cfg.get("min_n", 1))
            except Exception:
                continue
            if chk["ok"] and chk["label"] not in out:
                out.append(chk["label"])
            if len(out) >= n:
                break
        return out

    def log_proposal(self, it, policy, r, action):
        append_csv(self.d / "proposals.csv", PROP_FIELDS, [dict(
            iteration=it, policy=policy, input=r.get("input", ""), smiles=r.get("smiles", ""), label=r.get("label", ""),
            ok=r.get("ok"), reason=r.get("reason"), action=action, timestamp=now())])

    def get_crystals(self, mol_id, max_rows=20):
        rows = [r for r in read_csv(self.d / "crystals.csv") if r["mol_id"] == mol_id]
        rows.sort(key=lambda r: (r["energy_rank"] == "", int(r["energy_rank"] or 0)))
        keep = ["crystal_id", "Z", "status", "energy_rank", "dE_kJmol", "E_latt_kJmol", "E_inter_kJmol",
                "E_strain_kJmol", "density_gcm3",
                "a", "b", "c", "alpha", "beta", "gamma", "mobility_cm2Vs", "qc_reason"]
        out = []
        for r in rows[:max_rows]:
            out.append({k: (fmt(r[k], 3) if k in ("dE_kJmol", "E_latt_kJmol", "E_inter_kJmol", "E_strain_kJmol", "mobility_cm2Vs") else r[k]) for k in keep})
        return dict(mol_id=mol_id, n_total=len(rows), rows=out)

    def evaluate(self, tb, it, policy, r, rationale) -> dict:
        mol_id = f"M{len(self.molecules()) + 1:04d}"
        try:
            s = tb.evaluate(r["smiles"], mol_id)
        except Exception as e:
            s = dict(status="failed", reason=f"pipeline_error: {type(e).__name__}: {e}"[:300])
        row = dict(mol_id=mol_id, iteration=it, policy=policy, smiles=r["smiles"], label=r["label"],
                   skeleton=r["skeleton"], sites="-".join(map(str, r["sites"] or [])), n_N=r["n_N"],
                   formula=r["formula"], graph_key=r["graph_key"], aroma_key=r["aroma_key"],
                   rationale=rationale, timestamp=now(), **s)
        append_csv(self.d / "molecules.csv", MOL_FIELDS, [row])
        return row


# -------------------------------------------------------------------- policies
RELAX_TEXT = {False: "MACE-OFF23-small single-point energy per molecule (structures are NOT relaxed)",
              True: "quick MACE-OFF23-small relaxation (cell + positions, <= {n} steps) and single-point energy per "
                    "molecule (draws whose bonds change on relaxation are discarded)"}


def relax_text(pcfg) -> str:
    return RELAX_TEXT[pcfg.relax_steps > 0].format(n=pcfg.relax_steps)


def random_policy(camp: Campaign, tb, it, rng: random.Random):
    sks = sorted(camp.cfg.get("random_skeletons") or space.skeletons())
    for _ in range(1000):
        n = rng.randint(1, 12)
        args = dict(skeleton=rng.choice(sks), sites=sorted(rng.sample(range(1, 15), n)))
        r = camp.resolve(args)
        camp.log_proposal(it, "random", r, "evaluate" if r["ok"] else "rejected")
        if r["ok"]:
            return camp.evaluate(tb, it, "random", r, f"uniform random skeleton from {'/'.join(sks)}, "
                                                      "N count 1-12, random sites")
    raise RuntimeError("random policy could not find a new molecule")


def llm_policy(camp: Campaign, tb, it, log):
    from .llm import respond

    cfg = camp.cfg
    n_done = len(camp.molecules())
    notebook_path = camp.d / "notebook.md"
    notebook = notebook_path.read_text() if notebook_path.exists() else "(empty)"
    _, _, n_known = space.known_keys()
    instructions = SYSTEM.format(goal=cfg["goal"], space=space.describe_space(), n_known=n_known,
                                 draws=len(camp.pcfg.z_values) * camp.pcfg.samples_per_z,
                                 zs="/".join(map(str, camp.pcfg.z_values)), spz=camp.pcfg.samples_per_z,
                                 win=camp.pcfg.energy_window_kjmol,
                                 rank="E_latt (total energy)" if camp.pcfg.rank_energy == "total" else "E_inter",
                                 relax_text=relax_text(camp.pcfg))
    msg = (f"Turn {it}. Budget: {n_done} of {cfg['budget']} evaluations used.\n\n"
           f"RESULTS TABLE (evaluated molecules, best first):\n{camp.table_digest()}\n\n"
           f"YOUR NOTEBOOK FROM LAST TURN:\n{notebook}\n\n"
           "Choose and evaluate one new molecule now.")
    items = [dict(role="user", content=msg)]
    prev = None
    for step in range(cfg.get("max_tool_calls", 12)):
        prev, text, calls = respond(cfg["model"], items, instructions=instructions, tools=TOOLS, prev=prev,
                                    ledger=camp.d / "ledger.jsonl", effort=cfg.get("effort"), tag=f"turn{it}")
        log(dict(turn=it, step=step, text=text, calls=[(n, a) for _, n, a in calls]))
        if not calls:
            items = [dict(role="user", content="Please call evaluate_candidate with one new molecule.")]
            continue
        items = []
        for call_id, name, args in calls:
            if name == "check_candidate":
                r = camp.resolve(args)
                camp.log_proposal(it, "llm", r, "check")
                out = {k: r.get(k) for k in ("ok", "reason", "smiles", "label", "n_N", "formula")}
            elif name == "get_crystals":
                out = camp.get_crystals(args["mol_id"], args.get("max_rows", 20))
            elif name == "evaluate_candidate":
                r = camp.resolve(args)
                if not r["ok"]:
                    camp.log_proposal(it, "llm", r, "rejected")
                    out = dict(ok=False, reason=r["reason"], note="not evaluated; budget not used; pick another")
                else:
                    camp.log_proposal(it, "llm", r, "evaluate")
                    notebook_path.write_text(args.get("notebook", "")[:4000])
                    row = camp.evaluate(tb, it, "llm", r, args.get("rationale", ""))
                    log(dict(turn=it, evaluated=row["mol_id"], label=row["label"], mu=row.get("mu_lowestE"),
                             status=row["status"]))
                    return row
            else:
                out = dict(error=f"unknown tool {name}")
            items.append(dict(type="function_call_output", call_id=call_id, output=json.dumps(out, default=str)))
    log(dict(turn=it, warning="no evaluation chosen within tool-call limit"))
    return None


FRUGAL_NOTE = """

REQUEST-LIMITED RUN: only evaluate_candidate is available (no free checks, no crystal-level queries).
If a proposal is rejected (duplicate of a known/tested molecule, or outside the space) the budget is NOT used:
you get the reason plus a few novel valid site sets one edit away; pick one of them or another idea."""


def llm_policy_chat(camp: Campaign, tb, it, log):
    """Same loop as llm_policy, via OpenRouter Chat Completions (history kept locally within a turn).
    frugal=True (default for this provider): evaluate_candidate is the only tool -> ~1 request per turn."""
    from .llm import chat

    cfg = camp.cfg
    frugal = cfg.get("frugal", True)
    tools = [t for t in TOOLS if t["name"] == "evaluate_candidate"] if frugal else TOOLS
    n_done = len(camp.molecules())
    notebook_path = camp.d / "notebook.md"
    notebook = notebook_path.read_text() if notebook_path.exists() else "(empty)"
    _, _, n_known = space.known_keys()
    system = SYSTEM.format(goal=cfg["goal"], space=space.describe_space(), n_known=n_known,
                           draws=len(camp.pcfg.z_values) * camp.pcfg.samples_per_z,
                           zs="/".join(map(str, camp.pcfg.z_values)), spz=camp.pcfg.samples_per_z,
                           win=camp.pcfg.energy_window_kjmol,
                           rank="E_latt (total energy)" if camp.pcfg.rank_energy == "total" else "E_inter",
                           relax_text=relax_text(camp.pcfg))
    if frugal:
        system = system.replace("check_candidate is free: use it to validate ideas and avoid duplicates. "
                                "get_crystals shows crystal-level rows.\n", "") + FRUGAL_NOTE
    msg = (f"Turn {it}. Budget: {n_done} of {cfg['budget']} evaluations used.\n\n"
           f"RESULTS TABLE (evaluated molecules, best first):\n{camp.table_digest()}\n\n"
           f"YOUR NOTEBOOK FROM LAST TURN:\n{notebook}\n\n"
           "Choose and evaluate one new molecule now (call evaluate_candidate).")
    messages = [dict(role="system", content=system), dict(role="user", content=msg)]
    for step in range(cfg.get("max_tool_calls", 6)):
        am, text, calls = chat(cfg["model"], messages, tools=tools, ledger=camp.d / "ledger.jsonl",
                               effort=cfg.get("effort"), tag=f"turn{it}")
        messages.append(am)
        log(dict(turn=it, step=step, text=text[:2000], calls=[(n, a if a is not None else raw) for _, n, a, raw in calls]))
        if not calls:
            messages.append(dict(role="user", content="Please call evaluate_candidate with one new molecule."))
            continue
        for call_id, name, args, raw in calls:
            if args is None:
                out = dict(ok=False, reason=f"arguments are not valid JSON: {str(raw)[:200]}")
            elif name == "check_candidate" and not frugal:
                r = camp.resolve(args)
                camp.log_proposal(it, "llm", r, "check")
                out = {k: r.get(k) for k in ("ok", "reason", "smiles", "label", "n_N", "formula")}
            elif name == "get_crystals" and not frugal:
                out = camp.get_crystals(args.get("mol_id", ""), args.get("max_rows", 20))
            elif name == "evaluate_candidate":
                try:
                    r = camp.resolve(args)
                except Exception as e:  # malformed sites etc.
                    r = dict(ok=False, reason=f"bad_arguments: {type(e).__name__}: {e}"[:200], input=json.dumps(args))
                if not r["ok"]:
                    camp.log_proposal(it, "llm", r, "rejected")
                    out = dict(ok=False, reason=r["reason"], note="not evaluated; budget not used; pick another")
                    if frugal:
                        try:
                            ident = space.identify(Chem.MolFromSmiles(r["smiles"])) if r.get("smiles") else None
                        except Exception:
                            ident = None
                        if ident and ident[0]:
                            out["novel_one_edit_away"] = camp.novel_neighbours(dict(skeleton=ident[0], sites=ident[1]))
                else:
                    camp.log_proposal(it, "llm", r, "evaluate")
                    notebook_path.write_text(str(args.get("notebook", ""))[:4000])
                    row = camp.evaluate(tb, it, "llm", r, str(args.get("rationale", "")))
                    log(dict(turn=it, evaluated=row["mol_id"], label=row["label"], mu=row.get("mu_lowestE"),
                             status=row["status"]))
                    return row
            else:
                out = dict(error=f"tool {name} is not available in this run")
            messages.append(dict(role="tool", tool_call_id=call_id, content=json.dumps(out, default=str)))
    log(dict(turn=it, warning="no evaluation chosen within tool-call limit"))
    return None


# -------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--dir", type=Path, required=True)
    a.add_argument("--goal", default=DEFAULT_GOAL)
    a.add_argument("--policy", choices=["llm", "random"], default="llm")
    a.add_argument("--budget", type=int, default=30)
    a.add_argument("--model", default="gpt-6.1-sol")
    a.add_argument("--effort", default="medium")
    a.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    a.add_argument("--no-frugal", action="store_true",
                   help="openrouter: also offer check_candidate/get_crystals (costs more requests)")
    a.add_argument("--samples-per-z", type=int, default=64)
    a.add_argument("--z", type=int, nargs="+", default=[2, 4])
    a.add_argument("--gpu", default="0")
    a.add_argument("--rank-energy", choices=["total", "inter"], default="total",
                   help="total = MACE E_crys/Z of the unrelaxed structure (default); inter = minus frozen intramolecular energy")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--relax-steps", type=int, default=0, help="0 = no relaxation; e.g. 100 = quick MACE-OFF relax")
    a.add_argument("--relax-fixed-cell", action="store_true", help="relax positions only (default: cell + positions)")
    a.add_argument("--relax-dtype", default="float32")
    a.add_argument("--relax-workers", type=int, default=4)
    a.add_argument("--random-skeletons", nargs="+", default=None,
                   help="random policy only: restrict to these skeleton codes (e.g. LLL)")
    b = sub.add_parser("run")
    b.add_argument("--dir", type=Path, required=True)
    b.add_argument("--max-iterations", type=int, default=None, help="stop after this many turns in this session")
    c = sub.add_parser("report")
    c.add_argument("--dir", type=Path, required=True)
    f = sub.add_parser("retry-failed", help="re-run pipeline_error molecules in place (same mol_id, same Clari seeds); "
                                            "the original failed rows are kept in failed_attempts.csv")
    f.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "init":
        args.dir.mkdir(parents=True, exist_ok=False)
        pc = PipelineConfig(z_values=tuple(args.z), samples_per_z=args.samples_per_z, gpu=args.gpu, seed=args.seed,
                            rank_energy=args.rank_energy, relax_steps=args.relax_steps,
                            relax_cell=not args.relax_fixed_cell, relax_dtype=args.relax_dtype,
                            relax_workers=args.relax_workers)
        cfg = dict(goal=args.goal, policy=args.policy, budget=args.budget, model=args.model, effort=args.effort,
                   provider=args.provider, frugal=not args.no_frugal, random_skeletons=args.random_skeletons,
                   min_n=1, max_tool_calls=12 if args.provider == "openai" else 6, seed=args.seed, created=now(),
                   pipeline={**pc.__dict__, "z_values": list(pc.z_values)})
        (args.dir / "campaign.json").write_text(json.dumps(cfg, indent=2))
        print(f"initialised {args.dir}")
        return

    camp = Campaign(args.dir)
    if args.cmd == "report":
        print(camp.table_digest())
        return

    from .pipeline import Toolbox

    logf = (args.dir / "agent_log.jsonl").open("a")

    def log(obj):
        obj["t"] = now()
        logf.write(json.dumps(obj, default=str) + "\n")
        logf.flush()
        print(json.dumps(obj, default=str)[:400], flush=True)

    tb = Toolbox(camp.pcfg, args.dir)
    log(dict(event="toolbox_ready", load_seconds=tb.load_seconds, models=tb.model_ids))
    if args.cmd == "retry-failed":
        try:
            rows = camp.molecules()
            for row in rows:
                if row["status"] != "failed":
                    continue
                append_csv(args.dir / "failed_attempts.csv", MOL_FIELDS, [row])
                t0 = time.time()
                try:
                    s = tb.evaluate(row["smiles"], row["mol_id"])
                except Exception as e:
                    s = dict(status="failed", reason=f"pipeline_error: {type(e).__name__}: {e}"[:300])
                row.update({k: "" for k in MOL_FIELDS if k not in ("mol_id", "iteration", "policy", "smiles", "label",
                            "skeleton", "sites", "n_N", "formula", "graph_key", "aroma_key", "rationale")})
                row.update(s, timestamp=now())
                row["reason"] = (str(s.get("reason") or "") + " [retried; first attempt in failed_attempts.csv]").strip()
                log(dict(event="retry", mol_id=row["mol_id"], label=row["label"], mu=row.get("mu_lowestE"),
                         status=row["status"], seconds=round(time.time() - t0, 1)))
            tmp = args.dir / "molecules.csv.tmp"
            tmp.unlink(missing_ok=True)
            append_csv(tmp, MOL_FIELDS, rows)
            tmp.replace(args.dir / "molecules.csv")
        finally:
            tb.close()
            logf.close()
        print(camp.table_digest(20))
        return
    rng = random.Random(camp.cfg["seed"] + len(camp.molecules()))
    session_turns = 0
    try:
        while len(camp.molecules()) < camp.cfg["budget"]:
            if args.max_iterations is not None and session_turns >= args.max_iterations:
                break
            it = len(camp.molecules()) + 1
            t0 = time.time()
            if camp.cfg["policy"] == "random":
                row = random_policy(camp, tb, it, rng)
                log(dict(turn=it, evaluated=row["mol_id"], label=row["label"], mu=row.get("mu_lowestE"),
                         status=row["status"]))
            else:
                pol = llm_policy_chat if camp.cfg.get("provider", "openai") == "openrouter" else llm_policy
                row = pol(camp, tb, it, log)
            session_turns += 1
            log(dict(turn=it, seconds=round(time.time() - t0, 1)))
            if row is None:
                break
    finally:
        tb.close()
        logf.close()
    print(camp.table_digest(20))


if __name__ == "__main__":
    main()
