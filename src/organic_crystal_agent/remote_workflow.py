from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from organic_crystal_agent.postprocess import postprocess_job_dir
from organic_crystal_agent.workflow_state import initialize_stage_state, load_stage_state, update_stage_state


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class RemoteWorkflowInput(BaseModel):
    """Input payload for an external organic crystal workflow."""

    molecule_id: str = ""
    molecule_label: str
    molecule_path: str = ""
    smiles: str | None = None
    design_family: str = ""
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    proposal_notes: list[str] = Field(default_factory=list)
    objective: str
    sample_count: int = Field(default=1000, ge=1)
    sampling_notes: list[str] = Field(default_factory=list)
    optimization_backend: str = "MACE-OFF"
    structure_optimizer_script: str = ""
    z_value: int | None = Field(default=None, ge=1)
    space_group: int | None = Field(default=None, ge=1)
    le_hull_tag: str = "0.00"
    use_real_sg_ratio: bool = False
    checkpoint_path: str = ""
    sampler_temperature: float = 1.0
    sampler_top_k: int = 20
    sampler_batch_size: int = Field(default=24, ge=1)
    sampler_expand_max_v: float = Field(default=3.0, gt=0)
    relax_profile: str = ""
    relax_partition: str = ""
    relax_gpus: int | None = Field(default=None, ge=1)
    relax_n_gpus: int | None = Field(default=None, ge=1)
    target_space_group_number: int | None = Field(default=None, ge=1)


class RemoteWorkflowJob(BaseModel):
    """Represents a remote async job submitted to a server-side workflow."""

    workflow_name: str
    job_id: str
    status: JobStatus
    server_label: str
    molecule_label: str = ""
    smiles: str | None = None
    working_directory: str = ""
    submit_command: str = ""
    status_command: str = ""
    fetch_command: str = ""
    notes: list[str] = Field(default_factory=list)


class CrystalLandscapeSummary(BaseModel):
    """Compact summary of a CSP + optimization workflow result."""

    total_sampled_structures: int = Field(ge=0)
    optimized_structures: int = Field(ge=0)
    lowest_energy_ev: float | None = None
    energy_window_ev: float | None = None
    lowest_energy_kj_mol: float | None = None
    energy_window_kj_mol: float | None = None
    unique_low_energy_forms: int | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SlurmCSPSettings(BaseModel):
    """Server-local settings for AlphaPack sampling + BOMLIP relaxation."""

    server_label: str = "slurm-server"
    jobs_root: str = "/users/ffmmgg/scratch/OCAgent/jobs"
    scratch_root: str = "/mnt/scratch/users/ffmmgg"
    alphapack_repo: str = "/mnt/scratch/users/ffmmgg/AlphaPack_llm"
    alphapack_script: str = "/mnt/scratch/users/ffmmgg/AlphaPack_llm/AlphaPackllm/AP_llm/sample_CSP_Z.py"
    alphapack_pre_dedup_script: str = (
        "/mnt/scratch/users/ffmmgg/AlphaPack_llm/AlphaPackllm/AP_llm/pre_dedup.py"
    )
    alphapack_apptainer_module: str = "apptainer/1.3.6"
    alphapack_apptainer_image: str = "/mnt/scratch/users/ffmmgg/apptainer_APllm.sif"
    default_checkpoint_path: str = (
        "/mnt/scratch/users/ffmmgg/AlphaPack_llm/hydra/singlerun/2026-03-10/"
        "llm_extra_include_0109/epoch=60-step=208952_test.ckpt"
    )
    sampler_expand_max_v: float = 3.0
    bomlip_repo: str = "/mnt/scratch/users/ffmmgg/BOMLIP-CSP/BOMLIP-CSP/mace-bench"
    optimizer_script: str = (
        "/mnt/scratch/users/ffmmgg/BOMLIP-CSP/BOMLIP-CSP/mace-bench/reproduce/mace_opt_new_extra_CSP_Z.py"
    )
    conda_envs_dir: str = "/mnt/fastscratch/users/ffmmgg/conda_envs"
    conda_pkgs_dir: str = "/mnt/fastscratch/users/ffmmgg/conda_pkgs"
    pip_cache_dir: str = "/mnt/fastscratch/users/ffmmgg/.cache/pip"
    xdg_cache_home: str = "/mnt/fastscratch/users/ffmmgg/.cache"
    tmpdir: str = "/mnt/fastscratch/users/ffmmgg/tmp"
    bomlip_env_name: str = "BOMLIP_CSP"
    partition: str = "gpu-h100"
    time_limit: str = "3-00:00:00"
    postprocess_partition: str = "cooper"
    postprocess_time_limit: str = "03:00:00"
    gpus: int = 1
    n_jobs: int = 1
    n_gpus: int = 1
    gpu_offset: int = 0
    model_name: str = "small"
    max_steps: int = 1000
    heavy_relax_partition: str = "gpu-h100"
    heavy_relax_gpus: int = 1
    heavy_relax_n_gpus: int = 1
    light_relax_partition: str = "gpu-h100"
    light_relax_gpus: int = 1
    light_relax_n_gpus: int = 1
    pre_relax_enabled: bool = True
    pre_relax_k: int = 150
    pre_relax_metric: str = "chebyshev"
    pre_relax_threshold: float = 0.05
    pre_relax_low_memory: bool = False
    slurm_retry_attempts: int = 3
    slurm_retry_delay_s: float = 3.0


class ExternalCrystalWorkflow(Protocol):
    """Async interface for a server-side CSP / optimization workflow."""

    def submit(self, payload: RemoteWorkflowInput) -> RemoteWorkflowJob:
        ...

    def get_status(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        ...

    def fetch_results(self, job: RemoteWorkflowJob) -> CrystalLandscapeSummary:
        ...


class MockCSPWorkflow:
    """Deterministic placeholder for a real remote CSP + MACE-OFF workflow."""

    def submit(self, payload: RemoteWorkflowInput) -> RemoteWorkflowJob:
        return RemoteWorkflowJob(
            workflow_name="mock-csp-maceoff",
            job_id="mock-job-001",
            status="queued",
            server_label="mock-gpu-server",
            molecule_label=payload.molecule_label,
            smiles=payload.smiles,
            working_directory="/remote/mock/job-001",
            submit_command="submit_csp.sh",
            status_command="check_job.sh mock-job-001",
            fetch_command="fetch_landscape.sh mock-job-001",
            notes=[
                f"Requested {payload.sample_count} sampled structures.",
                f"Optimization backend: {payload.optimization_backend}.",
            ],
        )

    def get_status(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        updated = job.model_copy(deep=True)
        if updated.status == "queued":
            updated.status = "running"
        elif updated.status == "running":
            updated.status = "succeeded"
        return updated

    def fetch_results(self, job: RemoteWorkflowJob) -> CrystalLandscapeSummary:
        if job.status != "succeeded":
            raise RuntimeError("Cannot fetch results before the remote workflow succeeds.")
        return CrystalLandscapeSummary(
            total_sampled_structures=1000,
            optimized_structures=1000,
            lowest_energy_ev=-12.48,
            energy_window_ev=0.19,
            unique_low_energy_forms=3,
            artifact_paths=[
                f"{job.working_directory}/landscape.csv",
                f"{job.working_directory}/optimized_structures.xyz",
            ],
            notes=[
                "Landscape suggests three low-energy basins within a narrow window.",
                "Use these basins as crystal-form hypotheses for downstream experimental prioritization.",
            ],
        )


class SlurmCSPWorkflow:
    """Adapter around AlphaPack sampling and BOMLIP MACE relaxation on Slurm."""

    def __init__(self, settings: SlurmCSPSettings | None = None) -> None:
        self.settings = settings or SlurmCSPSettings()

    def submit(self, payload: RemoteWorkflowInput) -> RemoteWorkflowJob:
        prepared = self.prepare(payload)
        workdir = Path(prepared.working_directory)
        submitted = self._submit_stage(workdir, "sampling")
        return prepared.model_copy(
            update={
                "job_id": submitted["job_id"],
                "status": "queued",
                "submit_command": submitted["submit_command"],
                "status_command": self._status_command(submitted["job_id"]),
                "fetch_command": f"organic-crystal-agent csp-fetch --job-dir {workdir}",
            }
        )

    def prepare(self, payload: RemoteWorkflowInput) -> RemoteWorkflowJob:
        if not payload.molecule_path:
            raise ValueError("molecule_path is required for the Slurm workflow.")
        molecule_path = Path(payload.molecule_path).expanduser().resolve()
        if not molecule_path.exists():
            raise FileNotFoundError(f"Molecule file not found: {molecule_path}")

        workdir = self._make_job_dir(payload)
        staged_molecule = self._stage_molecule(molecule_path, payload, workdir)

        manifest = self._build_manifest(payload=payload, workdir=workdir, staged_molecule=staged_molecule)
        manifest_path = workdir / "job_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        initialize_stage_state(workdir, ["sampling", "relaxation", "postprocess"])
        self._write_stage_jobs(
            workdir,
            {
                "sampling": {"job_id": "", "script": str(workdir / "submit_slurm.sh")},
                "relaxation": {"job_id": "", "script": str(workdir / "submit_relax.sh")},
                "postprocess": {"job_id": "", "script": str(workdir / "submit_postprocess.sh")},
            },
        )

        submit_script = workdir / "submit_slurm.sh"
        submit_script.write_text(self._build_submit_script(manifest), encoding="utf-8")
        submit_script.chmod(0o755)
        relax_script = workdir / "submit_relax.sh"
        relax_script.write_text(self._build_relax_script(manifest), encoding="utf-8")
        relax_script.chmod(0o755)
        postprocess_script = workdir / "submit_postprocess.sh"
        postprocess_script.write_text(self._build_postprocess_script(manifest), encoding="utf-8")
        postprocess_script.chmod(0o755)

        return RemoteWorkflowJob(
            workflow_name="alphapack-bomlip-slurm",
            job_id="preview",
            status="queued",
            server_label=self.settings.server_label,
            molecule_label=payload.molecule_label,
            smiles=payload.smiles,
            working_directory=str(workdir),
            submit_command=f"sbatch {submit_script}",
            status_command="",
            fetch_command=f"organic-crystal-agent csp-fetch --job-dir {workdir}",
            notes=[
                f"Staged molecule: {staged_molecule.name}",
                f"Sampler xyz_root: {staged_molecule.parent}",
                f"Requested {payload.sample_count} sampled structures.",
                f"Z={payload.z_value}" if payload.z_value is not None else "No Z constraint.",
                f"SG={payload.space_group}" if payload.space_group is not None else "No fixed SG.",
            ],
        )

    def get_status(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        updated = job.model_copy(deep=True)
        workdir = Path(job.working_directory)
        if not workdir.exists():
            return updated

        stage_state = load_stage_state(workdir)
        stage_jobs = self._load_stage_jobs(workdir)

        for stage in ("sampling", "relaxation", "postprocess"):
            stage_info = stage_jobs.get(stage, {})
            stage_job_id = str(stage_info.get("job_id", "")).strip()
            stage_status = stage_state.get(stage, "pending")

            if stage_status == "completed":
                continue

            if stage_job_id:
                mapped = self._get_slurm_job_status(stage_job_id)
                if mapped is None:
                    updated.job_id = stage_job_id
                    updated.status = job.status if job.status in {"queued", "running"} else "queued"
                    updated.submit_command = f"sbatch {stage_info.get('script', '')}".strip()
                    updated.status_command = self._status_command(stage_job_id)
                    updated.notes = self._job_notes(workdir, current_stage=stage) + [
                        "Slurm status query was temporarily unavailable; preserving the current stage state."
                    ]
                    return updated
                if mapped in {"queued", "running"}:
                    updated.job_id = stage_job_id
                    updated.status = mapped
                    updated.submit_command = f"sbatch {stage_info.get('script', '')}".strip()
                    updated.status_command = self._status_command(stage_job_id)
                    updated.notes = self._job_notes(workdir, current_stage=stage)
                    return updated
                if mapped == "succeeded":
                    update_stage_state(workdir, stage, "completed")
                    stage_state[stage] = "completed"
                    continue
                if mapped in {"failed", "cancelled"}:
                    update_stage_state(workdir, stage, mapped)
                    updated.job_id = stage_job_id
                    updated.status = mapped
                    updated.submit_command = f"sbatch {stage_info.get('script', '')}".strip()
                    updated.status_command = self._status_command(stage_job_id)
                    updated.notes = self._job_notes(workdir, current_stage=stage)
                    return updated

            submitted = self._submit_stage(workdir, stage)
            updated.job_id = submitted["job_id"]
            updated.status = "queued"
            updated.submit_command = submitted["submit_command"]
            updated.status_command = self._status_command(submitted["job_id"])
            updated.notes = self._job_notes(workdir, current_stage=stage)
            return updated

        updated.status = "succeeded"
        updated.notes = self._job_notes(workdir, current_stage="completed")
        return updated

    def requeue(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        workdir = Path(job.working_directory)
        stage_state = load_stage_state(workdir)
        for stage in ("sampling", "relaxation", "postprocess"):
            if stage_state.get(stage) != "completed":
                submitted = self._submit_stage(workdir, stage)
                return job.model_copy(
                    update={
                        "job_id": submitted["job_id"],
                        "status": "queued",
                        "submit_command": submitted["submit_command"],
                        "status_command": self._status_command(submitted["job_id"]),
                        "notes": self._job_notes(workdir, current_stage=stage),
                    }
                )
        return job.model_copy(update={"status": "succeeded", "notes": self._job_notes(workdir, current_stage="completed")})

    def fetch_results(self, job: RemoteWorkflowJob) -> CrystalLandscapeSummary:
        if job.status not in {"succeeded", "running", "queued"}:
            raise RuntimeError(f"Cannot fetch results for job in terminal status {job.status!r}.")
        workdir = Path(job.working_directory)
        manifest = self._load_manifest(workdir)
        stage_state = load_stage_state(workdir)
        if stage_state.get("postprocess") != "completed":
            postprocess_job_dir(workdir)
            update_stage_state(workdir, "postprocess", "completed")
        return self._summarize_results(manifest)

    def _make_job_dir(self, payload: RemoteWorkflowInput) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _job_slug(
            molecule_id=payload.molecule_id,
            molecule_label=payload.molecule_label,
            fallback=Path(payload.molecule_path).stem,
        )
        job_dir = Path(self.settings.jobs_root).expanduser().resolve() / f"{ts}_{slug}"
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    def _stage_molecule(self, molecule_path: Path, payload: RemoteWorkflowInput, workdir: Path) -> Path:
        stage_slug = f"{workdir.name}-input"
        inputs_dir = workdir / "inputs" / stage_slug
        inputs_dir.mkdir(parents=True, exist_ok=True)
        staged_molecule = inputs_dir / molecule_path.name
        shutil.copy2(molecule_path, staged_molecule)
        return staged_molecule

    def _build_manifest(self, payload: RemoteWorkflowInput, workdir: Path, staged_molecule: Path) -> dict:
        ckpt_path = Path(payload.checkpoint_path or self.settings.default_checkpoint_path).expanduser()
        sample_root_name = staged_molecule.parent.name
        if payload.z_value is not None:
            sample_dir_name = f"{sample_root_name}_Z{payload.z_value}_CSP_samples"
        elif payload.space_group is not None:
            sample_dir_name = f"{sample_root_name}_SG{payload.space_group:03d}_CSP_samples"
        else:
            sample_dir_name = f"{sample_root_name}_CSP_samples"

        external_sample_dir = ckpt_path.parent / sample_dir_name
        external_sample_archive = external_sample_dir.with_suffix(".zip")
        local_sample_dir = workdir / "sample_output"
        pre_relax_dir = workdir / "pre_relax"
        relax_dir = workdir / "relax"

        return {
            "molecule_id": payload.molecule_id,
            "molecule_label": payload.molecule_label,
            "molecule_path": str(staged_molecule),
            "smiles": payload.smiles,
            "design_family": payload.design_family,
            "rationale": payload.rationale,
            "evidence": list(payload.evidence),
            "proposal_notes": list(payload.proposal_notes),
            "objective": payload.objective,
            "sample_count": payload.sample_count,
            "z_value": payload.z_value,
            "space_group": payload.space_group,
            "le_hull_tag": payload.le_hull_tag,
            "use_real_sg_ratio": payload.use_real_sg_ratio,
            "checkpoint_path": str(ckpt_path),
            "external_sample_dir": str(external_sample_dir),
            "external_sample_archive": str(external_sample_archive),
            "local_sample_dir": str(local_sample_dir),
            "pre_relax_dir": str(pre_relax_dir),
            "relax_dir": str(relax_dir),
            "sampler_temperature": payload.sampler_temperature,
            "sampler_top_k": payload.sampler_top_k,
            "sampler_batch_size": payload.sampler_batch_size,
            "sampler_expand_max_v": payload.sampler_expand_max_v or self.settings.sampler_expand_max_v,
            "sampling_notes": payload.sampling_notes,
            "relax_profile": payload.relax_profile,
            "relax_partition": payload.relax_partition,
            "relax_gpus": payload.relax_gpus,
            "relax_n_gpus": payload.relax_n_gpus,
            "target_space_group_number": payload.target_space_group_number,
            "ocagent_python": sys.executable,
            "settings": self.settings.model_dump(),
        }

    def _build_submit_script(self, manifest: dict) -> str:
        s = manifest["settings"]
        partition_spec = ",".join(part.strip() for part in str(s["partition"]).split(",") if part.strip())
        sample_csv = str(Path(manifest["local_sample_dir"]) / "all_samples.csv")
        sampler_command = self._build_sampler_command(manifest)
        return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm-%j.out
#SBATCH -J ocagent_csp_sample
#SBATCH -p {partition_spec}
#SBATCH -N 1
#SBATCH -t {s['time_limit']}
#SBATCH --gres=gpu:1
set -euo pipefail

echo =========================================================
echo SLURM job: submitted  date = `date`
date_start=`date +%s`
echo =========================================================
echo Job output begins
echo -----------------
echo

WORKDIR={sh_quote(str(Path(manifest['local_sample_dir']).parent))}
mkdir -p "$WORKDIR" {sh_quote(manifest['local_sample_dir'])}
STAGE_STATE="$WORKDIR/stage_state.json"

update_stage() {{
python - "$STAGE_STATE" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
stage = sys.argv[2]
status = sys.argv[3]
state = json.loads(path.read_text()) if path.exists() else {{}}
state[stage] = status
path.write_text(json.dumps(state, indent=2))
PY
}}

echo "[csp] job started on $(date)"
echo "[csp] workdir: $WORKDIR"

if [ -s {sh_quote(sample_csv)} ]; then
echo "[csp] sampling already completed; skipping sampler"
update_stage sampling completed
else
update_stage sampling running
{sampler_command}

rm -rf {sh_quote(manifest['local_sample_dir'])}
mkdir -p {sh_quote(manifest['local_sample_dir'])}
cp -a {sh_quote(manifest['external_sample_dir'])}/. {sh_quote(manifest['local_sample_dir'])}/
{self._build_external_sample_archive_command(manifest)}
update_stage sampling completed
fi

echo "[csp] job finished on $(date)"
echo
echo ---------------
echo Job output ends
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished   date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
"""

    def _build_relax_script(self, manifest: dict) -> str:
        s = manifest["settings"]
        pre_relax_enabled = bool(s.get("pre_relax_enabled", True))
        relax_partition = manifest.get("relax_partition") or s["partition"]
        relax_gpus = int(manifest.get("relax_gpus") or s["gpus"])
        partition_spec = ",".join(part.strip() for part in str(relax_partition).split(",") if part.strip())
        optimized_landscape_csv = str(Path(manifest["local_sample_dir"]).parent / "optimized_landscape.csv")
        relax_json_dir = str(Path(manifest["relax_dir"]) / "json_result")
        optimizer_command = self._build_optimizer_command(manifest, target_folder="$OPTIMIZER_TARGET_DIR")
        pre_relax_dir = Path(manifest["pre_relax_dir"])
        pre_relax_map = pre_relax_dir / "dedup_map.tsv"
        pre_relax_command = self._build_pre_relax_command(manifest, cif_dir="$FILTERED_SAMPLE_DIR")
        filtered_sample_dir = Path(manifest["local_sample_dir"]).parent / "relax_input_by_sg"
        sg_filter_summary = Path(manifest["local_sample_dir"]).parent / "space_group_filter_summary.json"
        target_space_group_number = manifest.get("target_space_group_number")
        if target_space_group_number is not None:
            sg_filter_block = f"""
FILTERED_SAMPLE_DIR={sh_quote(str(filtered_sample_dir))}
SG_FILTER_SUMMARY={sh_quote(str(sg_filter_summary))}
rm -rf "$FILTERED_SAMPLE_DIR"
mkdir -p "$FILTERED_SAMPLE_DIR"
{sh_quote(manifest["ocagent_python"])} - <<'PY'
import csv
import json
import shutil
from pathlib import Path

sample_dir = Path({manifest["local_sample_dir"]!r})
filtered_dir = Path({str(filtered_sample_dir)!r})
summary_path = Path({str(sg_filter_summary)!r})
target_sg = int({int(target_space_group_number)!r})
sample_csv = sample_dir / "all_samples.csv"
rows = []
matched_rows = []
missing_files = []
with sample_csv.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = list(reader.fieldnames or [])
    for row in reader:
        rows.append(row)
        raw_sg = (row.get("space_group") or "").strip()
        try:
            sg_number = int(raw_sg)
        except ValueError:
            continue
        if sg_number != target_sg:
            continue
        cif_name = (row.get("cif_filename") or "").strip()
        if not cif_name:
            continue
        source = sample_dir / cif_name
        if not source.exists():
            missing_files.append(cif_name)
            continue
        shutil.copy2(source, filtered_dir / cif_name)
        matched_rows.append(row)

summary = {{
    "target_space_group_number": target_sg,
    "sample_csv": str(sample_csv),
    "filtered_dir": str(filtered_dir),
    "total_sample_rows": len(rows),
    "matched_sample_rows": len(matched_rows),
    "matched_cif_files": len(list(filtered_dir.glob("*.cif"))),
    "missing_cif_files": missing_files,
}}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

if not matched_rows:
    raise SystemExit(
        f"[csp] no sampled CIFs matched target space group {{target_sg}}; see {{summary_path}}"
    )

with (filtered_dir / "all_samples.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(matched_rows)
PY
OPTIMIZER_TARGET_DIR="$FILTERED_SAMPLE_DIR"
"""
        else:
            sg_filter_block = f"""
FILTERED_SAMPLE_DIR={sh_quote(manifest["local_sample_dir"])}
OPTIMIZER_TARGET_DIR="$FILTERED_SAMPLE_DIR"
"""
        pre_relax_block = ""
        if pre_relax_enabled:
            pre_relax_block = f"""
if [ ! -d "$PRE_RELAX_DIR" ] || [ -z "$(find "$PRE_RELAX_DIR" -maxdepth 1 -name '*.cif' 2>/dev/null | head -n 1)" ]; then
  rm -rf "$PRE_RELAX_DIR"
  mkdir -p "$PRE_RELAX_DIR"
  echo "[csp] running pre-relax dedup on sampled CIFs"
  {pre_relax_command}
fi

if [ -z "$(find "$PRE_RELAX_DIR" -maxdepth 1 -name '*.cif' 2>/dev/null | head -n 1)" ]; then
  echo "[csp] pre-relax dedup did not produce CIFs; falling back to sample_output"
  rm -rf "$PRE_RELAX_DIR"
  mkdir -p "$PRE_RELAX_DIR"
  cp -a "$FILTERED_SAMPLE_DIR"/. "$PRE_RELAX_DIR"/
fi

OPTIMIZER_TARGET_DIR="$PRE_RELAX_DIR"
"""
        return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=NONE
#SBATCH -o slurm-%j.out
#SBATCH -J ocagent_csp_relax
#SBATCH -p {partition_spec}
#SBATCH -N 1
#SBATCH -t {s['time_limit']}
#SBATCH --gres=gpu:{relax_gpus}
set -euo pipefail

echo =========================================================
echo SLURM job: submitted  date = `date`
date_start=`date +%s`
echo =========================================================
echo Job output begins
echo -----------------
echo

WORKDIR={sh_quote(str(Path(manifest['local_sample_dir']).parent))}
mkdir -p "$WORKDIR" {sh_quote(manifest['relax_dir'])}
STAGE_STATE="$WORKDIR/stage_state.json"
PRE_RELAX_DIR={sh_quote(str(pre_relax_dir))}
PRE_RELAX_MAP={sh_quote(str(pre_relax_map))}

print_runtime() {{
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo
echo ---------------
echo Job output ends
echo =========================================================
echo SLURM job: finished   date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
}}

trap print_runtime EXIT

update_stage() {{
python - "$STAGE_STATE" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
stage = sys.argv[2]
status = sys.argv[3]
state = json.loads(path.read_text()) if path.exists() else {{}}
state[stage] = status
path.write_text(json.dumps(state, indent=2))
PY
}}

if [ ! -s {sh_quote(str(Path(manifest["local_sample_dir"]) / "all_samples.csv"))} ]; then
  echo "[csp] sample_output/all_samples.csv not found; cannot start relaxation"
  exit 1
fi

if [ -s {sh_quote(optimized_landscape_csv)} ] || [ "$(find {sh_quote(relax_json_dir)} -name '*.json' 2>/dev/null | head -n 1)" ]; then
  echo "[csp] relaxation already completed; skipping optimizer"
  update_stage relaxation completed
  exit 0
fi

echo "[csp] relax job started on $(date)"
echo "[csp] workdir: $WORKDIR"
update_stage relaxation running
export PATH=/usr/bin:/bin:/usr/local/bin
unset PYTHONHOME
unset PYTHONPATH
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_SHLVL
unset _CE_M
unset _CE_CONDA
export PYTHONNOUSERSITE=1

{sg_filter_block}

{pre_relax_block}

{optimizer_command}
echo "[csp] relax job finished on $(date)"
update_stage relaxation completed
"""

    def _build_postprocess_script(self, manifest: dict) -> str:
        s = manifest["settings"]
        partition_spec = ",".join(part.strip() for part in str(s["postprocess_partition"]).split(",") if part.strip())
        workdir = str(Path(manifest["local_sample_dir"]).parent)
        python_exe = str(Path(manifest["ocagent_python"]).resolve())
        return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm-%j.out
#SBATCH -J ocagent_csp_post
#SBATCH -p {partition_spec}
#SBATCH -N 1
#SBATCH -t {s['postprocess_time_limit']}
set -euo pipefail

export NUMBA_CACHE_DIR="${{NUMBA_CACHE_DIR:-/tmp/numba_ocagent_cache}}"
mkdir -p "$NUMBA_CACHE_DIR"

{sh_quote(python_exe)} - <<'PY'
from organic_crystal_agent.postprocess import postprocess_job_dir
from organic_crystal_agent.workflow_state import update_stage_state

workdir = {workdir!r}
update_stage_state(workdir, "postprocess", "running")
postprocess_job_dir(workdir)
update_stage_state(workdir, "postprocess", "completed")
PY
"""

    def _build_sampler_command(self, manifest: dict) -> str:
        s = manifest["settings"]
        sampler_args = [
            f"--ckpt {sh_quote(manifest['checkpoint_path'])}",
            f"--xyz_root {sh_quote(str(Path(manifest['molecule_path']).parent))}",
            f"--num_samples {int(manifest['sample_count'])}",
            f"--batch_size {int(manifest['sampler_batch_size'])}",
            f"--temperature {float(manifest['sampler_temperature'])}",
            f"--top_k {int(manifest['sampler_top_k'])}",
            f"--expand_maxV {float(manifest['sampler_expand_max_v'])}",
            "--device cuda",
            f"--LE_HULL {sh_quote(manifest['le_hull_tag'])}",
        ]
        if manifest["z_value"] is not None:
            sampler_args.append(f"--Z {int(manifest['z_value'])}")
        elif manifest["space_group"] is not None:
            sampler_args.append(f"--SG {int(manifest['space_group'])}")
        else:
            sampler_args.append("--SG none")
        if manifest["use_real_sg_ratio"]:
            sampler_args.append("--use_real_sg_ratio")
        return f"""module load {sh_quote(s['alphapack_apptainer_module']).strip("'")}
cd {sh_quote(s['scratch_root'])}
ulimit -n 4096
if [ -n "${{WANDB_API_KEY:-}}" ]; then export WANDB_API_KEY="${{WANDB_API_KEY}}"; fi
apptainer exec --nv {sh_quote(s['alphapack_apptainer_image'])} bash -c "cd {s['alphapack_repo']} && python {s['alphapack_script']} {' '.join(sampler_args)}\""""

    def _build_optimizer_command(self, manifest: dict, *, target_folder: str | None = None) -> str:
        s = manifest["settings"]
        pre_relax_enabled = bool(s.get("pre_relax_enabled", True))
        relax_n_gpus = int(manifest.get("relax_n_gpus") or s["n_gpus"])
        optimizer_target = target_folder or (
            manifest["pre_relax_dir"] if pre_relax_enabled else manifest["local_sample_dir"]
        )
        optimizer_args = [
            f"--target_folder {optimizer_target}",
            f"--path {sh_quote(manifest['relax_dir'])}",
            "--molecule_single -1",
            f"--n_jobs {int(s['n_jobs'])}",
            f"--n_gpus {relax_n_gpus}",
            f"--gpu_offset {int(s['gpu_offset'])}",
            f"--max_steps {int(s['max_steps'])}",
            f"--model {sh_quote(s['model_name'])}",
        ]
        optimizer_rel = os.path.relpath(s["optimizer_script"], s["bomlip_repo"])
        bomlip_python = Path(s["conda_envs_dir"]) / s["bomlip_env_name"] / "bin" / "python"
        mace_root = Path(s["bomlip_repo"]) / "3rdparty" / "mace"
        return "\n".join(
            [
                f"export CONDA_ENVS_DIRS={sh_quote(s['conda_envs_dir'])}",
                f"export CONDA_PKGS_DIRS={sh_quote(s['conda_pkgs_dir'])}",
                f"export PIP_CACHE_DIR={sh_quote(s['pip_cache_dir'])}",
                f"export XDG_CACHE_HOME={sh_quote(s['xdg_cache_home'])}",
                f"export TMPDIR={sh_quote(s['tmpdir'])}",
                f"export PYTHONPATH={sh_quote(str(mace_root))}",
                f"cd {sh_quote(s['bomlip_repo'])}",
                f"{sh_quote(str(bomlip_python))} {sh_quote(optimizer_rel)} {' '.join(optimizer_args)}",
            ]
        )

    def _build_pre_relax_command(self, manifest: dict, *, cif_dir: str | None = None) -> str:
        s = manifest["settings"]
        python_exe = manifest["ocagent_python"]
        pre_dedup_script = s["alphapack_pre_dedup_script"]
        args = [
            f"--cif_dir {cif_dir or sh_quote(manifest['local_sample_dir'])}",
            f"--out_dir {sh_quote(manifest['pre_relax_dir'])}",
            f"--k {int(s['pre_relax_k'])}",
            f"--metric {sh_quote(s['pre_relax_metric'])}",
            f"--threshold {float(s['pre_relax_threshold'])}",
        ]
        if s.get("pre_relax_low_memory"):
            args.append("--low_memory")
        return f"{sh_quote(python_exe)} {sh_quote(pre_dedup_script)} {' '.join(args)}"

    def _build_external_sample_archive_command(self, manifest: dict) -> str:
        python_exe = manifest["ocagent_python"]
        external_dir = manifest["external_sample_dir"]
        archive_path = manifest["external_sample_archive"]
        return f"""{sh_quote(python_exe)} - <<'PY'
from pathlib import Path
import shutil
import zipfile

source = Path({external_dir!r})
archive = Path({archive_path!r})
if source.exists():
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(source))
    shutil.rmtree(source)
PY"""

    def _load_manifest(self, workdir: Path) -> dict:
        manifest_path = workdir / "job_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing job manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Backwards-compatibility: older job manifests may be missing fields added later.
        # These are safe to default based on the job working directory.
        if "pre_relax_dir" not in manifest:
            manifest["pre_relax_dir"] = str(workdir / "pre_relax")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest

    def _stage_jobs_path(self, workdir: Path) -> Path:
        return workdir / "stage_jobs.json"

    def _load_stage_jobs(self, workdir: Path) -> dict:
        path = self._stage_jobs_path(workdir)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_stage_jobs(self, workdir: Path, payload: dict) -> None:
        self._stage_jobs_path(workdir).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _script_name_for_stage(self, stage: str) -> str:
        if stage == "sampling":
            return "submit_slurm.sh"
        if stage == "relaxation":
            return "submit_relax.sh"
        if stage == "postprocess":
            return "submit_postprocess.sh"
        raise ValueError(f"Unknown stage: {stage}")

    def _submit_stage(self, workdir: Path, stage: str) -> dict[str, str]:
        script_path = workdir / self._script_name_for_stage(stage)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing {stage} script: {script_path}")
        result = self._run_slurm_command(
            ["sbatch", str(script_path)],
            cwd=str(workdir),
            check=True,
        )
        job_id = self._parse_job_id(result.stdout)
        stage_jobs = self._load_stage_jobs(workdir)
        stage_jobs.setdefault(stage, {})
        stage_jobs[stage]["job_id"] = job_id
        stage_jobs[stage]["script"] = str(script_path)
        self._write_stage_jobs(workdir, stage_jobs)
        return {"job_id": job_id, "submit_command": f"sbatch {script_path}"}

    def _get_slurm_job_status(self, job_id: str) -> JobStatus | None:
        queue_result = self._run_slurm_command(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            check=False,
        )
        if queue_result.returncode != 0 and self._is_transient_slurm_error(queue_result.stderr):
            return None
        queue_state = queue_result.stdout.strip().splitlines()
        if queue_state:
            return self._map_slurm_state(queue_state[0].strip())
        acct_result = self._run_slurm_command(
            ["sacct", "-j", job_id, "--format=State", "--parsable2", "--noheader"],
            check=False,
        )
        if acct_result.returncode != 0 and self._is_transient_slurm_error(acct_result.stderr):
            return None
        acct_lines = [line.strip() for line in acct_result.stdout.splitlines() if line.strip()]
        if acct_lines:
            return self._map_slurm_state(acct_lines[0].split("|")[0].strip())
        return "queued"

    def _status_command(self, job_id: str) -> str:
        return f"squeue -h -j {job_id} -o %T || sacct -j {job_id} --format=State --parsable2 --noheader"

    def _run_slurm_command(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        attempts = max(1, int(self.settings.slurm_retry_attempts))
        delay = max(0.0, float(self.settings.slurm_retry_delay_s))
        last_result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
            last_result = result
            if result.returncode == 0:
                return result
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            if not self._is_transient_slurm_error(f"{stdout}\n{stderr}"):
                break
            if attempt < attempts:
                time.sleep(delay)
        assert last_result is not None
        if check:
            raise subprocess.CalledProcessError(
                last_result.returncode,
                last_result.args,
                output=last_result.stdout,
                stderr=last_result.stderr,
            )
        return last_result

    @staticmethod
    def _is_transient_slurm_error(message: str) -> bool:
        text = message.lower()

        # Not transient: Slurm reports an invalid job id for completed/absent jobs.
        # In that case we should fall back to sacct rather than suppressing status updates.
        if "invalid job id" in text:
            return False

        markers = (
            "unable to contact slurm controller",
            "slurm_load_jobs error",
            "problem talking to the database",
            "resource temporarily unavailable",
            "failed to open persistent connection",
            "sending persistinit msg",
            "connect failure",
            "unable to contact",
        )
        return any(marker in text for marker in markers)

    def _job_notes(self, workdir: Path, current_stage: str) -> list[str]:
        manifest = self._load_manifest(workdir)

        molecule_path = Path(manifest.get("molecule_path", "unknown"))
        notes = [
            f"Staged molecule: {molecule_path.name}",
            f"Molecule label: {manifest.get('molecule_label', '')}".strip(),
            f"Sampler xyz_root: {molecule_path.parent}",
            f"Requested {manifest.get('sample_count', 'unknown')} sampled structures.",
            f"Current stage: {current_stage}.",
        ]
        smiles = manifest.get("smiles")
        if smiles:
            notes.append(f"SMILES: {smiles}")
        notes = [note for note in notes if note]

        pre_relax_dir = manifest.get("pre_relax_dir")
        if pre_relax_dir:
            notes.append(f"Pre-relax dedup dir: {pre_relax_dir}")

        external_sample_archive = manifest.get("external_sample_archive")
        if external_sample_archive:
            notes.append(f"External sample archive: {external_sample_archive}")
        else:
            # Older manifests may not include this key; keep refresh resilient.
            notes.append("External sample archive: (none)")

        z_value = manifest.get("z_value")
        if z_value is not None:
            notes.append(f"Z={z_value}")
        else:
            notes.append("No Z constraint.")

        space_group = manifest.get("space_group")
        if space_group is not None:
            notes.append(f"SG={space_group}")
        else:
            notes.append("No fixed SG.")

        return notes

    def _summarize_results(self, manifest: dict) -> CrystalLandscapeSummary:
        sample_dir = Path(manifest["local_sample_dir"])
        relax_dir = Path(manifest["relax_dir"])
        sample_csv = sample_dir / "all_samples.csv"
        json_dir = relax_dir / "json_result"
        final_dir = relax_dir / "cif_result_final"
        timing_csv = relax_dir / "timing.csv"
        failed_txt = relax_dir / "failed_files.txt"
        landscape_csv = Path(manifest["local_sample_dir"]).parent / "optimized_landscape.csv"

        total_sampled = 0
        if sample_csv.exists():
            with sample_csv.open("r", encoding="utf-8") as fh:
                total_sampled = max(sum(1 for _ in csv.DictReader(fh)), 0)

        energies: list[float] = []
        if landscape_csv.exists():
            with landscape_csv.open("r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    energy = row.get("energy_kj")
                    if energy not in (None, ""):
                        energies.append(float(energy))
        elif json_dir.exists():
            for path in sorted(json_dir.glob("*.json")):
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                energy = row.get("energy_kj")
                if isinstance(energy, (float, int)):
                    energies.append(float(energy))

        energies.sort()
        lowest = energies[0] if energies else None
        window = (energies[-1] - energies[0]) if len(energies) >= 2 else 0.0 if energies else None
        unique_window = None
        if energies:
            cutoff = lowest + 7.0
            unique_window = sum(1 for e in energies if e <= cutoff)

        artifact_candidates: list[Path] = [
            sample_csv,
            timing_csv,
            failed_txt,
            Path(manifest["local_sample_dir"]).parent / "merged_samples.cif",
            Path(manifest["local_sample_dir"]).parent / "merged_optimized.cif",
            Path(manifest["local_sample_dir"]).parent / "optimized_landscape.csv",
            Path(manifest["local_sample_dir"]).parent / "sample_artifacts.tar.gz",
            Path(manifest["local_sample_dir"]).parent / "relax_artifacts.tar.gz",
            final_dir,
            json_dir,
        ]

        external_sample_archive = manifest.get("external_sample_archive")
        if external_sample_archive:
            artifact_candidates.append(Path(external_sample_archive))

        pre_relax_dir = manifest.get("pre_relax_dir")
        if pre_relax_dir:
            pre_relax_path = Path(pre_relax_dir)
            artifact_candidates.extend([pre_relax_path, pre_relax_path / "dedup_map.tsv"])

        artifacts = [str(path) for path in artifact_candidates if path.exists()]
        notes = []
        if failed_txt.exists():
            failed_count = sum(1 for line in failed_txt.read_text(encoding="utf-8").splitlines() if line.strip())
            notes.append(f"{failed_count} structures failed or timed out during relaxation.")
        if sample_dir.exists():
            notes.append(f"Sample artifacts staged in {sample_dir}.")
        if final_dir.exists():
            notes.append(f"Optimized CIFs available in {final_dir}.")
        postprocess_path = Path(manifest["local_sample_dir"]).parent / "postprocess_summary.json"
        if postprocess_path.exists():
            notes.append("Postprocessed artifacts were merged and archived to control file count.")
            try:
                postprocess_summary = json.loads(postprocess_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                postprocess_summary = {}
            dedup_summary = postprocess_summary.get("dedup_summary", {})
            if dedup_summary.get("dedup_performed"):
                unique_window = int(dedup_summary.get("cluster_count", unique_window or 0))
                method = dedup_summary.get("dedup_method", "")
                notes.append(
                    f"Deduplicated optimized structures into {unique_window} unique forms"
                    + (f" using {method}." if method else ".")
                )

        return CrystalLandscapeSummary(
            total_sampled_structures=total_sampled,
            optimized_structures=len(energies),
            lowest_energy_kj_mol=lowest,
            energy_window_kj_mol=window,
            unique_low_energy_forms=unique_window,
            artifact_paths=artifacts,
            notes=notes,
        )

    @staticmethod
    def _parse_job_id(stdout: str) -> str:
        match = re.search(r"Submitted batch job\s+(\d+)", stdout)
        if not match:
            raise RuntimeError(f"Could not parse sbatch output: {stdout!r}")
        return match.group(1)

    @staticmethod
    def _map_slurm_state(state: str) -> JobStatus:
        # sacct may return values like "CANCELLED by <uid>"; squeue can also include extra qualifiers.
        # Normalize by taking the first whitespace-delimited token.
        normalized = (state.strip().upper().split() or [""])[0]
        if normalized in {"PENDING", "CONFIGURING", "COMPLETING"}:
            return "queued"
        if normalized in {"RUNNING"}:
            return "running"
        if normalized in {"COMPLETED"}:
            return "succeeded"
        if normalized in {"CANCELLED", "CANCELLED+", "PREEMPTED"}:
            return "cancelled"
        if normalized in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}:
            return "failed"
        return "queued"


def get_default_remote_workflow() -> ExternalCrystalWorkflow:
    if os.environ.get("ORGANIC_CRYSTAL_AGENT_REMOTE_WORKFLOW", "mock").lower() == "slurm":
        return SlurmCSPWorkflow()
    return MockCSPWorkflow()


def load_job_from_path(path: str) -> RemoteWorkflowJob:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RemoteWorkflowJob.model_validate(raw)


def save_job_to_path(path: str, job: RemoteWorkflowJob) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(job.model_dump(), indent=2), encoding="utf-8")


def _slugify(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = lowered.strip("-")
    return lowered or "csp-job"


def _job_slug(*, molecule_id: str, molecule_label: str, fallback: str) -> str:
    parts: list[str] = []
    if molecule_id.strip():
        parts.append(_slugify(molecule_id))
    label_slug = _slugify(molecule_label or fallback)
    if label_slug:
        parts.append(label_slug)
    return "-".join(parts) or "csp-job"


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
