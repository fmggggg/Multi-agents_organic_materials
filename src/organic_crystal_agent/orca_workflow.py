from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from organic_crystal_agent.remote_workflow import RemoteWorkflowJob, SlurmCSPWorkflow, load_job_from_path, save_job_to_path
from organic_crystal_agent.workflow_state import initialize_stage_state, update_stage_state


ORCACalculationMode = Literal["sp", "opt", "freq", "optfreq"]
ORCAPropertyProfile = Literal["frontier", "hole_reorg", "pair_interaction", "pre_csp_dft_opt"]


class ORCAPropertyInput(BaseModel):
    molecule_id: str = ""
    molecule_label: str
    structure_path: str
    smiles: str | None = None
    design_family: str = ""
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    proposal_notes: list[str] = Field(default_factory=list)
    objective: str = "Estimate molecular electronic properties."
    property_profile: ORCAPropertyProfile = "frontier"
    calculation_mode: ORCACalculationMode = "sp"
    charge: int = 0
    multiplicity: int = 1
    fragment_a_path: str = ""
    fragment_b_path: str = ""
    method: str = "B3LYP"
    basis: str = "def2-SVP"
    # Default to serial ORCA runs to avoid MPI startup issues on this cluster.
    nprocs: int = Field(default=1, ge=1)
    memory_mb_per_core: int = Field(default=4000, ge=1)
    extra_keywords: list[str] = Field(default_factory=list)


class ORCAPropertySummary(BaseModel):
    property_profile: str = "frontier"
    calculation_mode: str
    total_energy_hartree: float | None = None
    homo_ev: float | None = None
    lumo_ev: float | None = None
    gap_ev: float | None = None
    frontier_splitting_ev: float | None = None
    hole_reorganization_energy_ev: float | None = None
    pair_interaction_energy_ev: float | None = None
    dipole_debye: float | None = None
    isotropic_polarizability_au: float | None = None
    imaginary_frequency_count: int | None = None
    termination_ok: bool = False
    method: str = ""
    basis: str = ""
    optimized_structure_path: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SlurmORCASettings(BaseModel):
    server_label: str = "slurm-server"
    jobs_root: str = "/users/ffmmgg/scratch/OCAgent/jobs/orca"
    orca_binary: str = "$ORCA_BINDIR/orca"
    orca_env_activate: str = "module load orca/6.0.1"
    # Barkla/Cooper cluster partitions do not include a generic "cpu" queue.
    # Use the standard CPU partition used elsewhere in this workspace.
    partition: str = "cooper"
    time_limit: str = "1-00:00:00"
    # Match ORCAPropertyInput.nprocs default (serial by default).
    cpus_per_task: int = 1
    mem_gb: int = 8


class SlurmORCAWorkflow:
    def __init__(self, settings: SlurmORCASettings | None = None) -> None:
        self.settings = settings or SlurmORCASettings()

    def submit(self, payload: ORCAPropertyInput) -> RemoteWorkflowJob:
        structure_path = Path(payload.structure_path).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        workdir = self._make_job_dir(payload)
        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        staged_structure = inputs_dir / structure_path.name
        shutil.copy2(structure_path, staged_structure)

        manifest = self._build_manifest(payload=payload, workdir=workdir, staged_structure=staged_structure)
        (workdir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        initialize_stage_state(workdir, ["property"])
        self._write_inputs(manifest)
        submit_script = workdir / "submit_orca.sh"
        submit_script.write_text(self._build_submit_script(manifest), encoding="utf-8")
        submit_script.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(submit_script)],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = SlurmCSPWorkflow._parse_job_id(result.stdout)
        return RemoteWorkflowJob(
            workflow_name="orca-slurm",
            job_id=job_id,
            status="queued",
            server_label=self.settings.server_label,
            molecule_label=payload.molecule_label,
            smiles=payload.smiles,
            working_directory=str(workdir),
            submit_command=f"sbatch {submit_script}",
            status_command=f"squeue -h -j {job_id} -o %T || sacct -j {job_id} --format=State --parsable2 --noheader",
            fetch_command=f"organic-crystal-agent orca-fetch --job-dir {workdir}",
            notes=[
                f"Staged structure: {staged_structure.name}",
                f"Profile: {payload.property_profile}",
                f"Mode: {payload.calculation_mode}",
                f"Method: {payload.method}/{payload.basis}",
            ],
        )

    def requeue(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        workdir = Path(job.working_directory)
        submit_script = workdir / "submit_orca.sh"
        result = subprocess.run(["sbatch", str(submit_script)], check=True, capture_output=True, text=True)
        job_id = SlurmCSPWorkflow._parse_job_id(result.stdout)
        return job.model_copy(
            update={
                "job_id": job_id,
                "status": "queued",
                "submit_command": f"sbatch {submit_script}",
                "status_command": f"squeue -h -j {job_id} -o %T || sacct -j {job_id} --format=State --parsable2 --noheader",
            }
        )

    def get_status(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        updated = job.model_copy(deep=True)
        if not job.job_id.isdigit():
            return updated

        queue_result = subprocess.run(
            ["squeue", "-h", "-j", job.job_id, "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
        queue_state = queue_result.stdout.strip().splitlines()
        if queue_state:
            updated.status = SlurmCSPWorkflow._map_slurm_state(queue_state[0].strip())
            return updated

        acct_result = subprocess.run(
            ["sacct", "-j", job.job_id, "--format=State", "--parsable2", "--noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        acct_lines = [line.strip() for line in acct_result.stdout.splitlines() if line.strip()]
        if acct_lines:
            updated.status = SlurmCSPWorkflow._map_slurm_state(acct_lines[0].split("|")[0].strip())
        return updated

    def fetch_results(self, job: RemoteWorkflowJob) -> ORCAPropertySummary:
        if job.status not in {"succeeded", "running", "queued"}:
            raise RuntimeError(f"Cannot fetch results for ORCA job in terminal status {job.status!r}.")
        workdir = Path(job.working_directory)
        manifest = self._load_manifest(workdir)
        summary = self._summarize_results(workdir, manifest)
        summary.property_profile = manifest["property_profile"]
        summary.calculation_mode = manifest["calculation_mode"]
        summary.method = manifest["method"]
        summary.basis = manifest["basis"]
        summary.notes.extend(
            [
                f"Objective: {manifest['objective']}",
                f"Structure source: {manifest['structure_path']}",
            ]
        )
        if summary.termination_ok:
            update_stage_state(workdir, "property", "completed")
        (workdir / "orca_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return summary

    def _make_job_dir(self, payload: ORCAPropertyInput) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label_slug = re.sub(r"[^a-z0-9]+", "-", payload.molecule_label.lower()).strip("-")
        id_slug = re.sub(r"[^a-z0-9]+", "-", payload.molecule_id.lower()).strip("-")
        slug = "-".join(part for part in [id_slug, label_slug] if part) or "orca-job"
        job_dir = Path(self.settings.jobs_root).expanduser().resolve() / f"{ts}_{slug}_{payload.calculation_mode}"
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    def _build_manifest(self, payload: ORCAPropertyInput, workdir: Path, staged_structure: Path) -> dict:
        return {
            "molecule_id": payload.molecule_id,
            "molecule_label": payload.molecule_label,
            "smiles": payload.smiles,
            "design_family": payload.design_family,
            "rationale": payload.rationale,
            "evidence": list(payload.evidence),
            "proposal_notes": list(payload.proposal_notes),
            "structure_path": str(staged_structure),
            "objective": payload.objective,
            "property_profile": payload.property_profile,
            "calculation_mode": payload.calculation_mode,
            "charge": payload.charge,
            "multiplicity": payload.multiplicity,
            "fragment_a_path": payload.fragment_a_path,
            "fragment_b_path": payload.fragment_b_path,
            "method": payload.method,
            "basis": payload.basis,
            "nprocs": payload.nprocs,
            "memory_mb_per_core": payload.memory_mb_per_core,
            "extra_keywords": payload.extra_keywords,
            "settings": self.settings.model_dump(),
            "workdir": str(workdir),
        }

    def _build_input_file(self, manifest: dict, *, charge: int | None = None, multiplicity: int | None = None) -> str:
        mode_keywords = {
            "sp": [],
            "opt": ["Opt"],
            "freq": ["Freq"],
            "optfreq": ["Opt", "Freq"],
        }[manifest["calculation_mode"]]
        keywords = [manifest["method"], manifest["basis"], *mode_keywords, *manifest["extra_keywords"]]
        return (
            f"! {' '.join(keywords)}\n"
            f"%pal nprocs {int(manifest['nprocs'])} end\n"
            f"%maxcore {int(manifest['memory_mb_per_core'])}\n"
            f"* xyzfile {int(manifest['charge'] if charge is None else charge)} "
            f"{int(manifest['multiplicity'] if multiplicity is None else multiplicity)} "
            f"{Path(manifest['structure_path']).name}\n"
        )

    def _write_inputs(self, manifest: dict) -> None:
        workdir = Path(manifest["workdir"])
        profile = manifest["property_profile"]
        if profile in {"frontier", "pre_csp_dft_opt"}:
            (workdir / "property.inp").write_text(self._build_input_file(manifest), encoding="utf-8")
            return
        if profile == "hole_reorg":
            inputs_dir = workdir / "reorg_inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            neutral_opt = dict(manifest)
            neutral_opt["calculation_mode"] = "opt"
            cation_opt = dict(manifest)
            cation_opt["calculation_mode"] = "opt"
            neutral_sp = dict(manifest)
            neutral_sp["calculation_mode"] = "sp"
            cation_sp = dict(manifest)
            cation_sp["calculation_mode"] = "sp"
            (inputs_dir / "neutral_opt.inp").write_text(
                self._build_input_file(neutral_opt, charge=0, multiplicity=1),
                encoding="utf-8",
            )
            (inputs_dir / "cation_opt.inp").write_text(
                self._build_input_file(cation_opt, charge=1, multiplicity=2),
                encoding="utf-8",
            )
            (inputs_dir / "neutral_sp_on_cation.inp").write_text(
                self._build_input_file(neutral_sp, charge=0, multiplicity=1),
                encoding="utf-8",
            )
            (inputs_dir / "cation_sp_on_neutral.inp").write_text(
                self._build_input_file(cation_sp, charge=1, multiplicity=2),
                encoding="utf-8",
            )
            return
        if profile == "pair_interaction":
            if not manifest["fragment_a_path"] or not manifest["fragment_b_path"]:
                raise ValueError("pair_interaction profile requires fragment_a_path and fragment_b_path.")
            inputs_dir = workdir / "pair_inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            dimer_sp = dict(manifest)
            dimer_sp["calculation_mode"] = "sp"
            mono_a = dict(manifest)
            mono_a["calculation_mode"] = "sp"
            mono_b = dict(manifest)
            mono_b["calculation_mode"] = "sp"
            (inputs_dir / "dimer_sp.inp").write_text(
                self._build_input_file(dimer_sp, charge=0, multiplicity=1),
                encoding="utf-8",
            )
            (inputs_dir / "monomer_a_sp.inp").write_text(
                self._build_input_file(mono_a, charge=0, multiplicity=1).replace(
                    Path(manifest["structure_path"]).name, Path(manifest["fragment_a_path"]).name
                ),
                encoding="utf-8",
            )
            (inputs_dir / "monomer_b_sp.inp").write_text(
                self._build_input_file(mono_b, charge=0, multiplicity=1).replace(
                    Path(manifest["structure_path"]).name, Path(manifest["fragment_b_path"]).name
                ),
                encoding="utf-8",
            )
            return
        raise ValueError(f"Unknown ORCA property profile: {profile}")

    def _build_submit_script(self, manifest: dict) -> str:
        settings = manifest["settings"]
        env_activate = f"{settings['orca_env_activate']}\n" if settings["orca_env_activate"] else ""
        if manifest["property_profile"] == "hole_reorg":
            return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm.out
#SBATCH -J ocagent_orca
#SBATCH -p {settings['partition']}
#SBATCH -N 1
#SBATCH -t {settings['time_limit']}
#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}
#SBATCH --mem={int(settings['mem_gb'])}G
set -euo pipefail

WORKDIR='{manifest['workdir']}'
cd "$WORKDIR"
mkdir -p reorg_outputs
if [ -s reorg_outputs/neutral_opt.out ] && [ -s reorg_outputs/cation_opt.out ] && [ -s reorg_outputs/neutral_sp_on_cation.out ] && [ -s reorg_outputs/cation_sp_on_neutral.out ]; then
echo "[orca] hole_reorg outputs already exist; skipping"
else
{env_activate}cp '{manifest['structure_path']}' .
{settings['orca_binary']} reorg_inputs/neutral_opt.inp > reorg_outputs/neutral_opt.out
{settings['orca_binary']} reorg_inputs/cation_opt.inp > reorg_outputs/cation_opt.out
{settings['orca_binary']} reorg_inputs/neutral_sp_on_cation.inp > reorg_outputs/neutral_sp_on_cation.out
{settings['orca_binary']} reorg_inputs/cation_sp_on_neutral.inp > reorg_outputs/cation_sp_on_neutral.out
fi
"""
        if manifest["property_profile"] == "pair_interaction":
            return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm.out
#SBATCH -J ocagent_orca
#SBATCH -p {settings['partition']}
#SBATCH -N 1
#SBATCH -t {settings['time_limit']}
#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}
#SBATCH --mem={int(settings['mem_gb'])}G
set -euo pipefail

WORKDIR='{manifest['workdir']}'
cd "$WORKDIR"
mkdir -p pair_outputs
if [ -s pair_outputs/dimer_sp.out ] && [ -s pair_outputs/monomer_a_sp.out ] && [ -s pair_outputs/monomer_b_sp.out ]; then
echo "[orca] pair_interaction outputs already exist; skipping"
else
{env_activate}cp '{manifest['structure_path']}' .
cp '{manifest['fragment_a_path']}' .
cp '{manifest['fragment_b_path']}' .
{settings['orca_binary']} pair_inputs/dimer_sp.inp > pair_outputs/dimer_sp.out
{settings['orca_binary']} pair_inputs/monomer_a_sp.inp > pair_outputs/monomer_a_sp.out
{settings['orca_binary']} pair_inputs/monomer_b_sp.inp > pair_outputs/monomer_b_sp.out
fi
"""
        return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm.out
#SBATCH -J ocagent_orca
#SBATCH -p {settings['partition']}
#SBATCH -N 1
#SBATCH -t {settings['time_limit']}
#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}
#SBATCH --mem={int(settings['mem_gb'])}G
set -euo pipefail

WORKDIR='{manifest['workdir']}'
cd "$WORKDIR"
if [ -s orca.out ] && grep -q 'ORCA TERMINATED NORMALLY' orca.out 2>/dev/null; then
echo "[orca] property output already completed; skipping"
else
{env_activate}cp '{manifest['structure_path']}' .
{settings['orca_binary']} property.inp > orca.out
fi
"""

    def _load_manifest(self, workdir: Path) -> dict:
        manifest_path = workdir / "job_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing ORCA manifest: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_output(output_path: str | Path) -> ORCAPropertySummary:
        path = Path(output_path)
        if not path.exists():
            return ORCAPropertySummary(calculation_mode="unknown", notes=["ORCA output file is missing."])
        text = path.read_text(encoding="utf-8", errors="ignore")

        total_energy = _match_last_float(text, r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
        homo_minus_1, homo, lumo = _parse_frontier_orbitals(text)
        dipole = _match_last_float(text, r"Magnitude \(a\.u\.\)\s*:\s*-?\d+\.\d+\s+Magnitude \(Debye\)\s*:\s*(-?\d+\.\d+)")
        polar = _match_last_float(text, r"Isotropic polarizability\s*:\s*(-?\d+\.\d+)")
        imag_count = len(re.findall(r"\*\*\*imaginary mode\*\*\*", text, flags=re.IGNORECASE))
        terminated = "ORCA TERMINATED NORMALLY" in text

        notes: list[str] = []
        if not terminated:
            notes.append("ORCA output does not show normal termination.")
        if homo is None or lumo is None:
            notes.append("Could not parse HOMO/LUMO levels from ORCA output.")

        return ORCAPropertySummary(
            property_profile="frontier",
            calculation_mode="unknown",
            total_energy_hartree=total_energy,
            homo_ev=homo,
            lumo_ev=lumo,
            gap_ev=round(lumo - homo, 6) if homo is not None and lumo is not None else None,
            frontier_splitting_ev=round(homo - homo_minus_1, 6)
            if homo_minus_1 is not None and homo is not None
            else None,
            dipole_debye=dipole,
            isotropic_polarizability_au=polar,
            imaginary_frequency_count=imag_count if "VIBRATIONAL FREQUENCIES" in text else None,
            termination_ok=terminated,
            notes=notes,
        )

    def _summarize_results(self, workdir: Path, manifest: dict) -> ORCAPropertySummary:
        profile = manifest["property_profile"]
        if profile in {"frontier", "pre_csp_dft_opt"}:
            output_path = workdir / "orca.out"
            summary = self._parse_output(output_path)
            summary.property_profile = profile
            summary.calculation_mode = manifest["calculation_mode"]
            if manifest["calculation_mode"] in {"opt", "optfreq"} and summary.termination_ok:
                optimized_xyz_path = workdir / "optimized_structure.xyz"
                if self._write_optimized_xyz_from_output(output_path, optimized_xyz_path):
                    summary.optimized_structure_path = str(optimized_xyz_path)
                else:
                    summary.notes.append("Could not extract optimized Cartesian coordinates from ORCA output.")
            summary.artifact_paths = [
                str(path)
                for path in [
                    workdir / "property.inp",
                    output_path,
                    workdir / "slurm.out",
                    workdir / "job_manifest.json",
                    workdir / "optimized_structure.xyz",
                ]
                if path.exists()
            ]
            return summary
        if profile == "hole_reorg":
            outputs_dir = workdir / "reorg_outputs"
            neutral_opt = self._parse_output(outputs_dir / "neutral_opt.out")
            cation_opt = self._parse_output(outputs_dir / "cation_opt.out")
            neutral_sp_on_cation = self._parse_output(outputs_dir / "neutral_sp_on_cation.out")
            cation_sp_on_neutral = self._parse_output(outputs_dir / "cation_sp_on_neutral.out")
            terms = [
                neutral_opt.total_energy_hartree,
                cation_opt.total_energy_hartree,
                neutral_sp_on_cation.total_energy_hartree,
                cation_sp_on_neutral.total_energy_hartree,
            ]
            hole_reorg = None
            if all(term is not None for term in terms):
                hole_reorg = round(
                    ((neutral_sp_on_cation.total_energy_hartree - neutral_opt.total_energy_hartree)
                    + (cation_sp_on_neutral.total_energy_hartree - cation_opt.total_energy_hartree)) * 27.211386,
                    6,
                )
            summary = ORCAPropertySummary(
                property_profile="hole_reorg",
                calculation_mode="bundle",
                total_energy_hartree=neutral_opt.total_energy_hartree,
                hole_reorganization_energy_ev=hole_reorg,
                termination_ok=all(
                    item.termination_ok
                    for item in [neutral_opt, cation_opt, neutral_sp_on_cation, cation_sp_on_neutral]
                ),
                notes=[
                    "Hole reorganization energy computed from four ORCA energies."
                ],
                artifact_paths=[
                    str(path)
                    for path in [
                        workdir / "reorg_inputs" / "neutral_opt.inp",
                        workdir / "reorg_inputs" / "cation_opt.inp",
                        workdir / "reorg_inputs" / "neutral_sp_on_cation.inp",
                        workdir / "reorg_inputs" / "cation_sp_on_neutral.inp",
                        outputs_dir / "neutral_opt.out",
                        outputs_dir / "cation_opt.out",
                        outputs_dir / "neutral_sp_on_cation.out",
                        outputs_dir / "cation_sp_on_neutral.out",
                        workdir / "slurm.out",
                        workdir / "job_manifest.json",
                    ]
                    if path.exists()
                ],
            )
            if not summary.termination_ok:
                summary.notes.append("One or more ORCA sub-jobs did not terminate normally.")
            return summary
        if profile == "pair_interaction":
            outputs_dir = workdir / "pair_outputs"
            dimer = self._parse_output(outputs_dir / "dimer_sp.out")
            monomer_a = self._parse_output(outputs_dir / "monomer_a_sp.out")
            monomer_b = self._parse_output(outputs_dir / "monomer_b_sp.out")
            pair_energy = None
            if (
                dimer.total_energy_hartree is not None
                and monomer_a.total_energy_hartree is not None
                and monomer_b.total_energy_hartree is not None
            ):
                pair_energy = round(
                    (dimer.total_energy_hartree - monomer_a.total_energy_hartree - monomer_b.total_energy_hartree)
                    * 27.211386,
                    6,
                )
            return ORCAPropertySummary(
                property_profile="pair_interaction",
                calculation_mode="bundle",
                total_energy_hartree=dimer.total_energy_hartree,
                homo_ev=dimer.homo_ev,
                lumo_ev=dimer.lumo_ev,
                gap_ev=dimer.gap_ev,
                frontier_splitting_ev=dimer.frontier_splitting_ev,
                pair_interaction_energy_ev=pair_energy,
                termination_ok=all(item.termination_ok for item in [dimer, monomer_a, monomer_b]),
                notes=[
                    "Pair interaction energy computed as E(dimer)-E(monomer A)-E(monomer B).",
                    "Frontier splitting is provided as a first-pass dimer electronic-coupling proxy.",
                ],
                artifact_paths=[
                    str(path)
                    for path in [
                        workdir / "pair_inputs" / "dimer_sp.inp",
                        workdir / "pair_inputs" / "monomer_a_sp.inp",
                        workdir / "pair_inputs" / "monomer_b_sp.inp",
                        outputs_dir / "dimer_sp.out",
                        outputs_dir / "monomer_a_sp.out",
                        outputs_dir / "monomer_b_sp.out",
                        workdir / "slurm.out",
                        workdir / "job_manifest.json",
                    ]
                    if path.exists()
                ],
            )
        raise ValueError(f"Unknown ORCA property profile: {profile}")

    @staticmethod
    def _write_optimized_xyz_from_output(output_path: Path, target_path: Path) -> bool:
        text = output_path.read_text(encoding="utf-8", errors="ignore")
        coordinates = SlurmORCAWorkflow._extract_last_cartesian_block(text)
        if not coordinates:
            return False
        lines = [str(len(coordinates)), f"Extracted from {output_path.name}"]
        for element, x, y, z in coordinates:
            lines.append(f"{element} {x:.10f} {y:.10f} {z:.10f}")
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    @staticmethod
    def _extract_last_cartesian_block(text: str) -> list[tuple[str, float, float, float]]:
        blocks: list[list[tuple[str, float, float, float]]] = []
        lines = text.splitlines()
        idx = 0
        while idx < len(lines):
            if "CARTESIAN COORDINATES (ANGSTROEM)" not in lines[idx]:
                idx += 1
                continue
            idx += 1
            while idx < len(lines) and (
                not lines[idx].strip()
                or set(lines[idx].strip()) == {"-"}
                or lines[idx].strip().lower().startswith("atom")
            ):
                idx += 1
            block: list[tuple[str, float, float, float]] = []
            while idx < len(lines):
                line = lines[idx].strip()
                if not line or set(line) == {"-"}:
                    break
                parts = line.split()
                if len(parts) < 4:
                    break
                try:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                except ValueError:
                    break
                block.append((element, x, y, z))
                idx += 1
            if block:
                blocks.append(block)
            idx += 1
        return blocks[-1] if blocks else []


def _match_last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def _parse_frontier_orbitals(text: str) -> tuple[float | None, float | None, float | None]:
    section_match = re.search(
        r"ORBITAL ENERGIES(.*?)(?:\n\s*\*\*\*|MULLIKEN POPULATION ANALYSIS|LOEWDIN POPULATION ANALYSIS|$)",
        text,
        flags=re.DOTALL,
    )
    if not section_match:
        return None, None, None
    occupied_levels: list[float] = []
    homo = None
    lumo = None
    for line in section_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            occ = float(parts[1])
            energy_ev = float(parts[-1])
        except ValueError:
            continue
        if occ > 0:
            homo = energy_ev
            occupied_levels.append(energy_ev)
        elif occ == 0 and lumo is None:
            lumo = energy_ev
    homo_minus_1 = occupied_levels[-2] if len(occupied_levels) >= 2 else None
    return homo_minus_1, homo, lumo


def get_default_orca_workflow() -> SlurmORCAWorkflow:
    if os.environ.get("ORGANIC_CRYSTAL_AGENT_ORCA_WORKFLOW", "slurm").lower() == "slurm":
        return SlurmORCAWorkflow()
    return SlurmORCAWorkflow()


__all__ = [
    "ORCAPropertyInput",
    "ORCAPropertySummary",
    "SlurmORCASettings",
    "SlurmORCAWorkflow",
    "get_default_orca_workflow",
    "load_job_from_path",
    "save_job_to_path",
]
