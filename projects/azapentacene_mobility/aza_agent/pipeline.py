"""Property-prediction chain for one molecule:
Clari-H sampling -> overlap/integrity QC -> [optional quick MACE-OFF relax -> post-relax overlap/integrity QC] ->
MACE-OFF float64 single points -> MCRT mobility on the lowest-energy crystal (+ the others within the window).

Simplifications (winter-school budget, stated in every summary):
  * relax_steps = 0: Clari structures are NOT relaxed; MACE-OFF23 small single points rank them as generated.
  * relax_steps > 0: quick relax (<= relax_steps LBFGS steps, fmax 0.05, cell + positions, float32), not converged
    to a true minimum in every case; any bond-graph change or new severe overlap = relax failure (excluded).
  * Mobility objective = MCRT prediction for the lowest-MACE-energy crystal.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem

from . import qc
from .workers.client import start_mace, start_mcrt

EV_TO_KJMOL = 96.48533212
from .config import CLARI_ROOT, HF_HOME


@dataclass
class PipelineConfig:
    clari_model: str = "clari-h"
    z_values: tuple[int, ...] = (2, 4)
    samples_per_z: int = 32
    clari_steps: int = 50
    energy_window_kjmol: float = 7.2
    score_window_max: int = 10       # MCRT on at most this many crystals inside the window (incl. the minimum)
    mace_model: str = "small"
    rank_energy: str = "total"       # "total": E_crys/Z (the specified simplification); "inter": E_crys/Z - frozen intramolecular
    gpu: str = "0"
    seed: int = 0
    relax_steps: int = 0             # 0 = no relaxation (original winter-school setting)
    relax_cell: bool = True          # FrechetCellFilter (cell + positions); False = fixed cell
    relax_fmax: float = 0.05
    relax_dtype: str = "float32"
    relax_workers: int = 4


CRYSTAL_FIELDS = ["crystal_id", "mol_id", "Z", "sample_idx", "status", "qc_reason", "overlap_pairs",
                  "min_flagged_distance", "natoms", "E_total_eV", "E_per_mol_kJmol", "E_latt_kJmol",
                  "E_intra_kJmol", "E_inter_kJmol", "E_strain_kJmol", "rank_key",
                  "dE_kJmol", "energy_rank", "in_window", "mobility_cm2Vs", "density_gcm3",
                  "a", "b", "c", "alpha", "beta", "gamma", "volume",
                  "relax_steps_taken", "relax_converged", "relax_fmax", "relax_seconds", "E0_unrelaxed_eV",
                  "E_relax_lowprec_eV", "volume0", "density0_gcm3"]


def append_csv(path: Path, fields: list[str], rows: list[dict]):
    new = not path.exists() or path.stat().st_size == 0
    if not new:  # keep the file's own column order (older campaigns have fewer columns)
        with path.open(newline="") as h:
            fields = next(csv.reader(h))
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


class Toolbox:
    """Holds the three models resident for the whole campaign."""

    def __init__(self, cfg: PipelineConfig, campaign_dir: Path):
        import os
        import sys

        self.cfg = cfg
        self.dir = campaign_dir
        logs = campaign_dir / "logs"
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
        os.environ.setdefault("HF_HOME", HF_HOME)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        if CLARI_ROOT not in sys.path:
            sys.path.insert(0, CLARI_ROOT)
        from clari.inference.inputs import make_request
        from clari.inference.sample import ClariSampler, _seed_everything

        self._make_request = make_request
        self._seed_everything = _seed_everything
        t = time.time()
        self.clari = ClariSampler(cfg.clari_model, device="cuda", n_steps=cfg.clari_steps, torch_threads=2,
                                  seed=cfg.seed, filter_clashing=False)
        _ = self.clari.lit  # load now, not inside the first evaluation
        self.load_seconds = {"clari": round(time.time() - t, 1)}
        t = time.time()
        self.mace = start_mace(logs, gpu=cfg.gpu, model=cfg.mace_model)
        self.load_seconds["mace"] = round(time.time() - t, 1)
        t = time.time()
        self.mcrt = start_mcrt(logs, gpu=cfg.gpu)
        self.load_seconds["mcrt"] = round(time.time() - t, 1)
        self.model_ids = dict(csp=f"Clari ({cfg.clari_model}, {cfg.clari_steps} steps, EMA, bf16)",
                              energy=self.mace.info["model"] + (f" relax {cfg.relax_steps} steps ({cfg.relax_dtype}, cell={cfg.relax_cell}) + " if cfg.relax_steps > 0 else " ") + "single point, " + self.mace.info["dtype"],
                              mobility="MCRT Jay mobility epoch=46-step=62134")

    def close(self):
        self.mace.close()
        self.mcrt.close()

    # ------------------------------------------------------------------ stages
    def gas_energy(self, smiles: str, rows: list[dict] | None = None) -> dict:
        """Gas reference: MACE-optimised RDKit (ETKDG + MMFF) conformer. ETKDG fails for some aza-[5]helicenes;
        then the start geometry is the first intact molecule unwrapped from a QC-passed Clari crystal."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0:
            AllChem.MMFFOptimizeMolecule(mol)
            pos = mol.GetConformer().GetPositions().tolist()
            sym = [a.GetSymbol() for a in mol.GetAtoms()]
            source = "rdkit_etkdg_mmff"
        else:
            start = next((r["_mols"][0] for r in rows or [] if r.get("status") == "qc_ok"), None)
            if start is None:
                raise RuntimeError("gas conformer: ETKDG embedding failed and no intact crystal molecule")
            sym, pos = start["symbols"], start["positions"]
            source = "clari_crystal_molecule"
        r = self.mace.call(cmd="gas", symbols=sym, positions=pos, fmax=0.01, steps=500)
        return dict(E_gas_eV=r["energy_eV"], gas_opt_steps=r["steps"], gas_converged=r["converged"], gas_source=source)

    def sample(self, smiles: str, mol_id: str, raw_dir: Path) -> list[dict]:
        rows = []
        raw_dir.mkdir(parents=True, exist_ok=True)
        for z in self.cfg.z_values:
            # deterministic per molecule and Z (stable across resumes); bf16 GPU kernels may still differ slightly
            self._seed_everything(self.cfg.seed * 1_000_003 + int(mol_id.lstrip("MT")) * 101 + z)
            req = self._make_request(smiles, id=f"{mol_id}_Z{z}", copies=z, samples=self.cfg.samples_per_z)
            crystals = self.clari.sample_request(req, pbar=False)
            for idx, cr in enumerate(crystals):
                cid = f"{mol_id}_Z{z}_s{idx:03d}"
                p = raw_dir / f"{cid}.cif"
                p.write_text(cr.to_cif())
                rows.append(dict(crystal_id=cid, mol_id=mol_id, Z=z, sample_idx=idx, path=str(p)))
            # a short Clari return would otherwise be silent: record the shortfall as explicit rows
            for idx in range(len(crystals), self.cfg.samples_per_z):
                rows.append(dict(crystal_id=f"{mol_id}_Z{z}_s{idx:03d}", mol_id=mol_id, Z=z, sample_idx=idx,
                                 path="", status="clari_no_output"))
        return rows

    def qc(self, rows: list[dict], smiles: str):
        ref = qc.reference_graph(smiles)
        for r in rows:
            if r.get("status"):
                continue
            try:
                st = Structure.from_file(r["path"])
                at = AseAtomsAdaptor.get_atoms(st)
                r.update(natoms=len(at), density_gcm3=round(float(st.density), 4), volume=round(st.volume, 3),
                         **{k: round(v, 4) for k, v in zip(("a", "b", "c"), st.lattice.abc)},
                         **{k: round(v, 3) for k, v in zip(("alpha", "beta", "gamma"), st.lattice.angles)})
                ov = qc.severe_overlap_check(at, factor=qc.OVERLAP_FACTOR)
                r["overlap_pairs"] = ov["severe_overlap_pairs"]
                if not ov["accepted"]:
                    ex = ov["examples"][0]
                    r.update(status="skipped_severe_overlap", min_flagged_distance=ov["min_flagged_distance_angstrom"],
                             qc_reason=f"{ex['element_i']}-{ex['element_j']} {ex['distance_angstrom']:.3f}A "
                                       f"< {ex['threshold_angstrom']:.3f}A (factor {qc.OVERLAP_FACTOR})")
                    continue
                integ = qc.integrity_check(at, ref, int(r["Z"]))
                if not integ["intact"]:
                    r.update(status="broken_molecule", qc_reason=integ["reason"])
                    continue
                r.update(status="qc_ok", qc_reason="ok", _mols=integ["molecules"])
            except Exception as e:
                r.update(status="parse_error", qc_reason=f"{type(e).__name__}: {e}"[:200])

    def relax(self, rows: list[dict], smiles: str, out_dir: Path):
        """Quick relax of QC-passed draws, then the same overlap + bond-graph checks on the relaxed structure."""
        todo = [r for r in rows if r.get("status") == "qc_ok"]
        if not todo:
            return
        c = self.cfg
        res = self.mace.call(cmd="relax", cifs=[r["path"] for r in todo], out_dir=str(out_dir), steps=c.relax_steps,
                             fmax=c.relax_fmax, cell=c.relax_cell, dtype=c.relax_dtype, workers=c.relax_workers)
        ref = qc.reference_graph(smiles)
        for r, x in zip(todo, res["results"]):
            r.pop("_mols", None)
            if "error" in x:
                r.update(status="relax_error", qc_reason=x["error"][:200], relax_seconds=x.get("seconds"))
                continue
            r.update(relax_steps_taken=x["steps"], relax_converged=int(x["converged"]), relax_fmax=round(x["fmax"], 4),
                     relax_seconds=x["seconds"], E0_unrelaxed_eV=x["E0_eV"], E_relax_lowprec_eV=x["E_eV"],
                     volume0=r.get("volume"), density0_gcm3=r.get("density_gcm3"), raw_path=r["path"],
                     path=x["out_path"])
            try:
                st = Structure.from_file(x["out_path"])
                at = AseAtomsAdaptor.get_atoms(st)
                r.update(density_gcm3=round(float(st.density), 4), volume=round(st.volume, 3),
                         **{k: round(v, 4) for k, v in zip(("a", "b", "c"), st.lattice.abc)},
                         **{k: round(v, 3) for k, v in zip(("alpha", "beta", "gamma"), st.lattice.angles)})
                ov = qc.severe_overlap_check(at, factor=qc.OVERLAP_FACTOR)
                if not ov["accepted"]:
                    ex = ov["examples"][0]
                    r.update(status="relax_failed_overlap", min_flagged_distance=ov["min_flagged_distance_angstrom"],
                             qc_reason=f"after relax: {ex['element_i']}-{ex['element_j']} {ex['distance_angstrom']:.3f}A")
                    continue
                integ = qc.integrity_check(at, ref, int(r["Z"]))
                if not integ["intact"]:
                    r.update(status="relax_failed_bonds", qc_reason="after relax: " + integ["reason"])
                    continue
                r.update(status="qc_ok", qc_reason="ok", _mols=integ["molecules"])
            except Exception as e:
                r.update(status="relax_error", qc_reason=f"post-relax parse {type(e).__name__}: {e}"[:200])

    def energies(self, rows: list[dict], e_gas_ev: float | None):
        todo = [r for r in rows if r.get("status") == "qc_ok"]
        if not todo:
            return
        res = self.mace.call(cmd="energies", cifs=[r["path"] for r in todo])["results"]
        for r, e in zip(todo, res):
            if "error" in e:
                r.update(status="energy_error", qc_reason=e["error"][:200])
                continue
            per_mol = e["energy_eV"] / int(r["Z"])
            intra = float(np.mean(self.mace.call(cmd="molecules", molecules=r.pop("_mols"))["energies_eV"]))
            r.update(status="scored_energy", E_total_eV=e["energy_eV"], E_per_mol_kJmol=per_mol * EV_TO_KJMOL,
                     E_latt_kJmol=(per_mol - e_gas_ev) * EV_TO_KJMOL,
                     E_intra_kJmol=intra * EV_TO_KJMOL, E_inter_kJmol=(per_mol - intra) * EV_TO_KJMOL,
                     E_strain_kJmol=(intra - e_gas_ev) * EV_TO_KJMOL)
            r["rank_key"] = r["E_per_mol_kJmol"] if self.cfg.rank_energy == "total" else r["E_inter_kJmol"]
        scored = sorted((r for r in rows if r.get("status") == "scored_energy"), key=lambda r: r["rank_key"])
        if not scored:
            return
        emin = scored[0]["rank_key"]
        for k, r in enumerate(scored):
            r["energy_rank"] = k + 1
            r["dE_kJmol"] = r["rank_key"] - emin
            r["in_window"] = int(r["dE_kJmol"] <= self.cfg.energy_window_kjmol)

    def mobility(self, rows: list[dict], work: Path):
        scored = sorted((r for r in rows if r.get("energy_rank")), key=lambda r: r["energy_rank"])
        chosen = [r for r in scored if r["in_window"]][: self.cfg.score_window_max]
        if not chosen:
            return
        res = self.mcrt.call(cmd="predict", cifs=[r["path"] for r in chosen], work_dir=str(work))
        by_id = {x["cif_id"]: x["mobility"] for x in res["results"]}
        for r in chosen:
            if r["crystal_id"] in by_id:
                r["mobility_cm2Vs"] = by_id[r["crystal_id"]]
            else:
                r["qc_reason"] = (r.get("qc_reason") or "") + "; mcrt_missing"

    # ------------------------------------------------------------------ full chain
    def evaluate(self, smiles: str, mol_id: str) -> dict:
        mdir = self.dir / "structures" / mol_id
        raw = mdir / "raw"
        t = {}
        t0 = time.time()
        rows = self.sample(smiles, mol_id, raw)
        t["clari"] = time.time() - t0
        t0 = time.time()
        self.qc(rows, smiles)
        t["qc"] = time.time() - t0
        relaxed = mdir / "relaxed"
        if self.cfg.relax_steps > 0:
            t0 = time.time()
            self.relax(rows, smiles, relaxed)
            t["relax"] = time.time() - t0
        t0 = time.time()
        gas = self.gas_energy(smiles, rows)
        self.energies(rows, gas["E_gas_eV"])
        t["mace"] = time.time() - t0
        t0 = time.time()
        self.mobility(rows, self.dir / "tmp" / f"mcrt_{mol_id}")
        t["mcrt"] = time.time() - t0

        # keep the lowest-energy crystal loose, archive everything else as a verified zip
        scored = sorted((r for r in rows if r.get("energy_rank")), key=lambda r: r["energy_rank"])
        best = scored[0] if scored else None
        if best:
            shutil.copy(best["path"], mdir / (f"{best['crystal_id']}_relaxed.cif" if self.cfg.relax_steps > 0
                                              else f"{best['crystal_id']}.cif"))
        for src, zname in ((raw, "raw_cifs.zip"), (relaxed, "relaxed_cifs.zip")):
            cifs = sorted(src.glob("*.cif")) if src.exists() else []
            if not cifs:
                continue
            zpath = mdir / zname
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in cifs:
                    zf.write(p, p.name)
            with zipfile.ZipFile(zpath) as zf:
                if zf.testzip() is not None or len(zf.namelist()) != len(cifs):
                    raise RuntimeError(f"zip verification failed for {zpath}")
            shutil.rmtree(src)

        append_csv(self.dir / "crystals.csv", CRYSTAL_FIELDS, rows)
        counts = {s: sum(1 for r in rows if r.get("status") == s) for s in
                  ("clari_no_output", "parse_error", "skipped_severe_overlap", "broken_molecule", "energy_error",
                   "relax_error", "relax_failed_overlap", "relax_failed_bonds")}
        win = [r for r in scored if r.get("in_window") and r.get("mobility_cm2Vs") is not None]
        mus = [r["mobility_cm2Vs"] for r in win]
        summary = dict(
            n_draws=len(rows),
            n_qc_ok=sum(1 for r in rows if r.get("status") in ("qc_ok", "scored_energy", "energy_error", "relax_error",
                                                                 "relax_failed_overlap", "relax_failed_bonds")),
            n_relax_fail=counts["relax_error"] + counts["relax_failed_overlap"] + counts["relax_failed_bonds"],
            relaxed=int(self.cfg.relax_steps > 0),
            n_overlap=counts["skipped_severe_overlap"], n_broken=counts["broken_molecule"],
            n_parse_error=counts["parse_error"] + counts["clari_no_output"], n_energy=len(scored),
            E_gas_eV=gas["E_gas_eV"], gas_source=gas["gas_source"], gas_converged=gas["gas_converged"],
            best_crystal_id=best["crystal_id"] if best else "", best_Z=best["Z"] if best else "",
            E_latt_min_kJmol=best["E_latt_kJmol"] if best else None,
            E_inter_best_kJmol=best["E_inter_kJmol"] if best else None,
            strain_best_kJmol=best["E_strain_kJmol"] if best else None,
            density_best=best.get("density_gcm3") if best else None,
            mu_lowestE=best.get("mobility_cm2Vs") if best else None,
            n_window=sum(1 for r in scored if r.get("in_window")), n_window_scored=len(mus),
            mu_mean_window=float(np.mean(mus)) if mus else None, mu_max_window=float(np.max(mus)) if mus else None,
            **{f"sec_{k}": round(v, 1) for k, v in t.items()},
        )
        if best is None:
            summary["status"], summary["reason"] = "failed", "no_crystal_passed_qc_and_energy"
        elif summary["mu_lowestE"] is None:
            summary["status"], summary["reason"] = "failed", "mcrt_failed_on_lowest_energy_crystal"
        else:
            summary["status"], summary["reason"] = "evaluated", "ok"
        (mdir / "summary.json").write_text(json.dumps(dict(smiles=smiles, mol_id=mol_id, **summary,
                                                           models=self.model_ids, config=self.cfg.__dict__),
                                                      indent=2, default=str))
        return summary
