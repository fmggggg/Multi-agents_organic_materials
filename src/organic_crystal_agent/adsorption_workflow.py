from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from organic_crystal_agent.property_analysis import score_property_metrics
from organic_crystal_agent.postprocess import load_spike_representative_points
from organic_crystal_agent.remote_workflow import RemoteWorkflowJob, SlurmCSPWorkflow
from organic_crystal_agent.schemas import PropertyMetrics
from organic_crystal_agent.workflow_state import initialize_stage_state, update_stage_state


class AdsorptionInput(BaseModel):
    molecule_id: str = ""
    molecule_label: str
    smiles: str | None = None
    design_family: str = ""
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    proposal_notes: list[str] = Field(default_factory=list)
    structure_path: str = ""
    structures: list[dict[str, object]] = Field(default_factory=list)
    skipped_spike_names: list[str] = Field(default_factory=list)
    source_csp_workdir: str = ""
    source_csp_job_id: str = ""
    objective: str = "Estimate adsorption for the selected crystal form."
    property_profile: str = "humid_co2_capture"
    calculation_mode: str = "single_point"
    temperature_k: float = Field(default=298.15, gt=0)
    pressure_pa: float = Field(default=6_500_000.0, gt=0)
    cycles: int = Field(default=25_000, ge=1)
    init_cycles: int = Field(default=5_000, ge=1)
    unit_cells: tuple[int, int, int] = (3, 3, 3)
    co2_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    h2o_fraction: float = Field(default=0.85, ge=0.0, le=1.0)
    relative_humidity: float = Field(default=0.75, ge=0.0, le=1.0)
    co2_partial_pressure_pa: float = Field(default=15_000.0, gt=0)
    water_saturation_pressure_pa: float = Field(default=3_167.0, gt=0)


class AdsorptionSummary(BaseModel):
    property_profile: str = "humid_co2_capture"
    calculation_mode: str = "single_point"
    methane_uptake_absolute_mmol_g: float | None = None
    methane_uptake_excess_mmol_g: float | None = None
    co2_uptake_absolute_mmol_g: float | None = None
    co2_uptake_excess_mmol_g: float | None = None
    h2o_uptake_absolute_mmol_g: float | None = None
    h2o_uptake_excess_mmol_g: float | None = None
    termination_ok: bool = False
    selected_spike_names: list[str] = Field(default_factory=list)
    skipped_spike_names: list[str] = Field(default_factory=list)
    structure_results: list[dict[str, object]] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SlurmAdsorptionSettings(BaseModel):
    server_label: str = "slurm-server"
    jobs_root: str = "/users/ffmmgg/scratch/OCAgent/jobs/adsorption"
    partition: str = "cooper"
    time_limit: str = "3-00:00:00"
    raspa_root: str = "/users/ffmmgg/scratch/RASPA2"
    raspa_env_path: str = "/mnt/fastscratch/users/ffmmgg/conda_envs/RASPA2"
    methane_script_path: str = "/users/ffmmgg/scratch/RASPA2/example/get_single_adsorbing_capacity.py"
    humid_co2_script_path: str = (
        "/users/ffmmgg/scratch/RASPA2/example/mixed_adsorb/CO2H2O_validation_mixed_adsorb.py"
    )
    methane_forcefield: str = "UFF_Dreiding"
    humid_co2_forcefield: str = "UFF_Dreiding"
    humid_replace_forcefield: bool = False
    job_name: str = "ocagent_adsorb"


class SlurmAdsorptionWorkflow:
    def __init__(self, settings: SlurmAdsorptionSettings | None = None) -> None:
        self.settings = settings or SlurmAdsorptionSettings()

    def submit(self, payload: AdsorptionInput) -> RemoteWorkflowJob:
        structure_entries = self._normalize_structure_entries(payload)
        if not structure_entries:
            raise ValueError("Adsorption submission requires at least one structure.")

        workdir = self._make_job_dir(payload)
        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        staged_structures: list[dict[str, object]] = []
        for entry in structure_entries:
            structure_path = Path(str(entry["structure_path"])).expanduser().resolve()
            if not structure_path.exists():
                raise FileNotFoundError(f"Structure file not found: {structure_path}")
            staged_structure = inputs_dir / structure_path.name
            shutil.copy2(structure_path, staged_structure)
            staged_structures.append(
                {
                    "name": str(entry["name"]),
                    "density": float(entry["density"]) if entry.get("density") is not None else None,
                    "relative_energy_kj": (
                        float(entry["relative_energy_kj"])
                        if entry.get("relative_energy_kj") is not None
                        else None
                    ),
                    "structure_path": str(staged_structure),
                }
            )

        manifest = self._build_manifest(payload=payload, workdir=workdir, staged_structures=staged_structures)
        (workdir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        initialize_stage_state(workdir, ["property"])
        submit_script = workdir / "submit_adsorb.sh"
        submit_script.write_text(self._build_submit_script(manifest), encoding="utf-8")
        submit_script.chmod(0o755)

        result = subprocess.run(["sbatch", str(submit_script)], check=True, capture_output=True, text=True)
        job_id = SlurmCSPWorkflow._parse_job_id(result.stdout)
        return RemoteWorkflowJob(
            workflow_name="raspa-adsorption-slurm",
            job_id=job_id,
            status="queued",
            server_label=self.settings.server_label,
            molecule_label=payload.molecule_label,
            smiles=payload.smiles,
            working_directory=str(workdir),
            submit_command=f"sbatch {submit_script}",
            status_command=f"squeue -h -j {job_id} -o %T || sacct -j {job_id} --format=State --parsable2 --noheader",
            fetch_command=f"organic-crystal-agent adsorb-fetch --job-dir {workdir}",
            notes=[
                f"Selected porous spike structures: {', '.join(item['name'] for item in staged_structures)}",
                f"Source CSP job: {payload.source_csp_job_id or 'unknown'}",
                f"T={payload.temperature_k} K, P={payload.pressure_pa} Pa",
                f"Profile={payload.property_profile}",
            ],
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

    def fetch_results(self, job: RemoteWorkflowJob) -> AdsorptionSummary:
        if job.status not in {"succeeded", "running", "queued"}:
            raise RuntimeError(f"Cannot fetch results for adsorption job in terminal status {job.status!r}.")
        workdir = Path(job.working_directory)
        manifest = json.loads((workdir / "job_manifest.json").read_text(encoding="utf-8"))
        summary = self._parse_adsorb_output(workdir / "adsorb.out")
        summary.property_profile = manifest["property_profile"]
        summary.calculation_mode = manifest["calculation_mode"]
        summary.selected_spike_names = [str(item["name"]) for item in manifest.get("structures", [])]
        summary.skipped_spike_names = [str(item) for item in manifest.get("skipped_spike_names", [])]
        summary.notes.extend(
            [
                f"Objective: {manifest['objective']}",
                f"Source CSP job: {manifest['source_csp_job_id']}",
                f"Temperature/K: {manifest['temperature_k']}",
                f"Pressure/Pa: {manifest['pressure_pa']}",
            ]
        )
        if manifest["property_profile"] == "humid_co2_capture":
            summary.notes.extend(
                [
                    f"CO2 fraction: {manifest['co2_fraction']}",
                    f"H2O fraction: {manifest['h2o_fraction']}",
                    f"Relative humidity: {manifest['relative_humidity']}",
                    f"CO2 partial pressure/Pa: {manifest['co2_partial_pressure_pa']}",
                    f"Water saturation pressure/Pa: {manifest['water_saturation_pressure_pa']}",
                ]
            )
        structure_results = self._collect_structure_results(workdir, manifest)
        summary.structure_results = structure_results
        successful = [row for row in structure_results if row.get("termination_ok")]
        if successful:
            if manifest["property_profile"] == "humid_co2_capture":
                best = max(successful, key=self._structure_property_score)
                best_score = self._structure_property_score(best)
                summary.co2_uptake_absolute_mmol_g = float(best["co2_uptake_absolute_mmol_g"])
                summary.co2_uptake_excess_mmol_g = (
                    float(best["co2_uptake_excess_mmol_g"])
                    if best.get("co2_uptake_excess_mmol_g") is not None
                    else None
                )
                summary.h2o_uptake_absolute_mmol_g = (
                    float(best["h2o_uptake_absolute_mmol_g"])
                    if best.get("h2o_uptake_absolute_mmol_g") is not None
                    else None
                )
                summary.h2o_uptake_excess_mmol_g = (
                    float(best["h2o_uptake_excess_mmol_g"])
                    if best.get("h2o_uptake_excess_mmol_g") is not None
                    else None
                )
                summary.termination_ok = True
                summary.notes.append(
                    f"Best humid-CO2 adsorption result from spike {best['name']} at density {best.get('density')} g/cm^3 with score {best_score:.3f}."
                )
            else:
                best = max(successful, key=lambda row: float(row.get("methane_uptake_absolute_mmol_g") or float("-inf")))
                summary.methane_uptake_absolute_mmol_g = float(best["methane_uptake_absolute_mmol_g"])
                summary.methane_uptake_excess_mmol_g = (
                    float(best["methane_uptake_excess_mmol_g"])
                    if best.get("methane_uptake_excess_mmol_g") is not None
                    else None
                )
                summary.termination_ok = True
                summary.notes.append(
                    f"Best methane uptake from spike {best['name']} at density {best.get('density')} g/cm^3."
                )
        else:
            summary.termination_ok = False
            summary.notes.append("No successful adsorption calculations were parsed.")
        results_csv = workdir / "adsorption_results.csv"
        self._write_results_csv(results_csv, structure_results)
        summary.artifact_paths.append(str(results_csv))
        summary.artifact_paths.append(str(workdir / "adsorb.out"))
        for row in structure_results:
            for artifact in row.get("artifact_paths", []):
                if artifact not in summary.artifact_paths:
                    summary.artifact_paths.append(str(artifact))
        if summary.termination_ok:
            update_stage_state(workdir, "property", "completed")
        (workdir / "adsorption_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return summary

    def _make_job_dir(self, payload: AdsorptionInput) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label_slug = re.sub(r"[^a-z0-9]+", "-", payload.molecule_label.lower()).strip("-")
        id_slug = re.sub(r"[^a-z0-9]+", "-", payload.molecule_id.lower()).strip("-")
        slug = "-".join(part for part in [id_slug, label_slug] if part) or "adsorb-job"
        if payload.source_csp_workdir:
            base_dir = (
                Path(payload.source_csp_workdir).expanduser().resolve() / "property_jobs"
            )
        else:
            base_dir = Path(self.settings.jobs_root).expanduser().resolve()
        suffix = "humid-co2" if payload.property_profile == "humid_co2_capture" else "methane"
        job_dir = base_dir / f"{ts}_{slug}_{suffix}"
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_dir

    def _build_manifest(self, payload: AdsorptionInput, workdir: Path, staged_structures: list[dict[str, object]]) -> dict:
        manifest = {
            "molecule_id": payload.molecule_id,
            "molecule_label": payload.molecule_label,
            "smiles": payload.smiles,
            "design_family": payload.design_family,
            "rationale": payload.rationale,
            "evidence": list(payload.evidence),
            "proposal_notes": list(payload.proposal_notes),
            "structures": staged_structures,
            "skipped_spike_names": list(payload.skipped_spike_names),
            "source_csp_workdir": payload.source_csp_workdir,
            "source_csp_job_id": payload.source_csp_job_id,
            "objective": payload.objective,
            "property_profile": payload.property_profile,
            "calculation_mode": payload.calculation_mode,
            "temperature_k": payload.temperature_k,
            "pressure_pa": payload.pressure_pa,
            "cycles": payload.cycles,
            "init_cycles": payload.init_cycles,
            "unit_cells": list(payload.unit_cells),
            "settings": self.settings.model_dump(),
            "workdir": str(workdir),
        }
        if payload.property_profile == "humid_co2_capture":
            manifest.update(
                {
                    "co2_fraction": payload.co2_fraction,
                    "h2o_fraction": payload.h2o_fraction,
                    "relative_humidity": payload.relative_humidity,
                    "co2_partial_pressure_pa": payload.co2_partial_pressure_pa,
                    "water_saturation_pressure_pa": payload.water_saturation_pressure_pa,
                }
            )
        return manifest

    def _build_submit_script(self, manifest: dict) -> str:
        settings = manifest["settings"]
        return f"""#!/bin/bash
#SBATCH -D {manifest['workdir']}
#SBATCH --export=ALL
#SBATCH -o {manifest['workdir']}/adsorb.out
#SBATCH -J {settings['job_name']}
#SBATCH -p {settings['partition']}
#SBATCH -t {settings['time_limit']}
set -euo pipefail

# Avoid cluster-wide PYTHONPATH pollution (e.g. spack Python 3.11 wheels) breaking the
# RASPA2 conda env (currently Python 3.6) used by the adsorption helper scripts.
unset PYTHONPATH
export PYTHONPATH=""
export PYTHONNOUSERSITE=1

export RASPA2_ROOT='{settings['raspa_root']}'
export RASPA2_ENV_PATH='{settings['raspa_env_path']}'
export RASPA2_PY_ROOT='{settings['raspa_env_path']}/lib/python3.6/site-packages/RASPA2'
export RASPA2_RESULT_DIR='{manifest['workdir']}/results/single_comp_adsorb'
export PATH="{settings['raspa_env_path']}/bin:$PATH"
PYTHON_BIN='{settings['raspa_env_path']}/bin/python'

mkdir -p "$RASPA2_RESULT_DIR"
cd '{manifest['workdir']}'
"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path('job_manifest.json').read_text())
logs_dir = Path(manifest['workdir']) / 'results' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

for entry in manifest['structures']:
    unit_cells = ",".join(str(int(x)) for x in manifest['unit_cells'])
    result_subdir = 'mixed_adsorb' if manifest['property_profile'] == 'humid_co2_capture' else 'single_comp_adsorb'
    result_dir = Path(manifest['workdir']) / 'results' / result_subdir / str(entry['name'])
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{{entry['name']}}.out"
    if manifest['property_profile'] == 'humid_co2_capture':
        cmd = [
            str(Path(manifest['settings']['raspa_env_path']) / 'bin' / 'python'),
            manifest['settings']['humid_co2_script_path'],
            '--cif-path',
            str(entry['structure_path']),
            '--workdir',
            str(result_dir),
            '--temperature',
            str(manifest['temperature_k']),
            '--pressure',
            str(manifest['pressure_pa']),
            '--co2-fraction',
            str(manifest['co2_fraction']),
            '--h2o-fraction',
            str(manifest['h2o_fraction']),
            '--relative-humidity',
            str(manifest['relative_humidity']),
            '--co2-partial-pressure',
            str(manifest['co2_partial_pressure_pa']),
            '--water-saturation-pressure',
            str(manifest['water_saturation_pressure_pa']),
            '--cycles',
            str(int(manifest['cycles'])),
            '--init-cycles',
            str(5000),
            '--forcefield',
            str(manifest['settings'].get('humid_co2_forcefield', 'UFF_Dreiding')),
        ]
        if manifest['settings'].get('humid_replace_forcefield'):
            cmd.append('--replace-forcefield')
    else:
        cmd = [
            str(Path(manifest['settings']['raspa_env_path']) / 'bin' / 'python'),
            manifest['settings']['methane_script_path'],
            '--cif_path',
            str(entry['structure_path']),
            '--result-dir',
            str(result_dir),
            '--temperature',
            str(manifest['temperature_k']),
            '--pressure',
            str(manifest['pressure_pa']),
            '--cycles',
            str(int(manifest['cycles'])),
            '--init-cycles',
            str(int(manifest['init_cycles'])),
            '--unit-cells',
            unit_cells,
        ]
    with log_path.open('w', encoding='utf-8') as handle:
        try:
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            handle.write(f"\\nADSORB_JOB_FAILED returncode={{exc.returncode}}\\n")
PY
"""

    @staticmethod
    def _parse_adsorb_output(output_path: Path) -> AdsorptionSummary:
        if not output_path.exists():
            return AdsorptionSummary(notes=["adsorb.out is missing."])

        text = output_path.read_text(encoding="utf-8", errors="ignore")
        mixed_match = re.search(
            r"ADSORB_SUMMARY_CO2H2O\s+\S+\s+avg_loading_absolute_co2=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+avg_loading_excess_co2=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+avg_loading_absolute_h2o=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+avg_loading_excess_h2o=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text,
            re.MULTILINE,
        )
        if mixed_match:
            return AdsorptionSummary(
                property_profile="humid_co2_capture",
                co2_uptake_absolute_mmol_g=float(mixed_match.group(1)),
                co2_uptake_excess_mmol_g=float(mixed_match.group(2)),
                h2o_uptake_absolute_mmol_g=float(mixed_match.group(3)),
                h2o_uptake_excess_mmol_g=float(mixed_match.group(4)),
                termination_ok=True,
            )
        summary_match = re.search(
            r"ADSORB_SUMMARY\s+\S+\s+avg_loading_absolute_methane=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+avg_loading_excess_methane=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text,
            re.MULTILINE,
        )
        if summary_match:
            return AdsorptionSummary(
                property_profile="methane_adsorption",
                methane_uptake_absolute_mmol_g=float(summary_match.group(1)),
                methane_uptake_excess_mmol_g=float(summary_match.group(2)),
                termination_ok=True,
            )

        abs_match = re.search(r"avg_loading_absolute_methane\s+avg_loading_excess_methane\s*\n([^\n]+?)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", text, re.MULTILINE)
        if not abs_match:
            notes = ["Could not parse methane adsorption summary from adsorb.out."]
            with suppress(Exception):
                text = output_path.read_text(encoding="utf-8", errors="ignore")
                if "ADSORB_JOB_FAILED" in text:
                    notes.append("RASPA adsorption subprocess failed for this structure.")
            return AdsorptionSummary(notes=notes)

        return AdsorptionSummary(
            property_profile="methane_adsorption",
            methane_uptake_absolute_mmol_g=float(abs_match.group(2)),
            methane_uptake_excess_mmol_g=float(abs_match.group(3)),
            termination_ok=True,
        )

    @staticmethod
    def _normalize_structure_entries(payload: AdsorptionInput) -> list[dict[str, object]]:
        if payload.structures:
            normalized = []
            for index, item in enumerate(payload.structures, start=1):
                if bool(item.get("skip")):
                    continue
                normalized.append(
                    {
                        "name": str(item.get("name") or f"structure-{index:02d}"),
                        "structure_path": str(item.get("structure_path") or payload.structure_path),
                        "density": item.get("density"),
                        "relative_energy_kj": item.get("relative_energy_kj"),
                    }
                )
            return normalized
        if payload.structure_path:
            return [{"name": Path(payload.structure_path).stem, "structure_path": payload.structure_path}]
        return []

    @staticmethod
    def _collect_structure_results(workdir: Path, manifest: dict) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for entry in manifest.get("structures", []):
            name = str(entry["name"])
            log_path = workdir / "results" / "logs" / f"{name}.out"
            parsed = SlurmAdsorptionWorkflow._parse_adsorb_output(log_path)
            property_score = SlurmAdsorptionWorkflow._structure_property_score(
                {
                    "termination_ok": parsed.termination_ok,
                    "methane_uptake_absolute_mmol_g": parsed.methane_uptake_absolute_mmol_g,
                    "methane_uptake_excess_mmol_g": parsed.methane_uptake_excess_mmol_g,
                    "co2_uptake_absolute_mmol_g": parsed.co2_uptake_absolute_mmol_g,
                    "co2_uptake_excess_mmol_g": parsed.co2_uptake_excess_mmol_g,
                    "h2o_uptake_absolute_mmol_g": parsed.h2o_uptake_absolute_mmol_g,
                    "h2o_uptake_excess_mmol_g": parsed.h2o_uptake_excess_mmol_g,
                }
            )
            structure_result: dict[str, object] = {
                "name": name,
                "density": entry.get("density"),
                "relative_energy_kj": entry.get("relative_energy_kj"),
                "structure_path": entry.get("structure_path"),
                "log_path": str(log_path),
                "termination_ok": parsed.termination_ok,
                "methane_uptake_absolute_mmol_g": parsed.methane_uptake_absolute_mmol_g,
                "methane_uptake_excess_mmol_g": parsed.methane_uptake_excess_mmol_g,
                "co2_uptake_absolute_mmol_g": parsed.co2_uptake_absolute_mmol_g,
                "co2_uptake_excess_mmol_g": parsed.co2_uptake_excess_mmol_g,
                "h2o_uptake_absolute_mmol_g": parsed.h2o_uptake_absolute_mmol_g,
                "h2o_uptake_excess_mmol_g": parsed.h2o_uptake_excess_mmol_g,
                "property_score": property_score,
                "notes": list(parsed.notes),
                "artifact_paths": [str(log_path)],
            }
            subdirs = [
                workdir / "results" / "single_comp_adsorb" / name,
                workdir / "results" / "mixed_adsorb" / name,
            ]
            xlsx_matches: list[Path] = []
            for result_dir in subdirs:
                xlsx_matches.extend(sorted(result_dir.glob("*.xlsx")))
            for path in xlsx_matches:
                structure_result["artifact_paths"].append(str(path))
            results.append(structure_result)
        return results

    @staticmethod
    def _write_results_csv(csv_path: Path, structure_results: list[dict[str, object]]) -> None:
        import csv

        fieldnames = [
            "name",
            "density",
            "relative_energy_kj",
            "termination_ok",
            "methane_uptake_absolute_mmol_g",
            "methane_uptake_excess_mmol_g",
            "co2_uptake_absolute_mmol_g",
            "co2_uptake_excess_mmol_g",
            "h2o_uptake_absolute_mmol_g",
            "h2o_uptake_excess_mmol_g",
            "property_score",
            "log_path",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in structure_results:
                writer.writerow({key: row.get(key) for key in fieldnames})

    @staticmethod
    def _structure_property_score(row: dict[str, object]) -> float:
        metrics = PropertyMetrics(
            methane_uptake_absolute_mmol_g=_maybe_float(row.get("methane_uptake_absolute_mmol_g")),
            methane_uptake_excess_mmol_g=_maybe_float(row.get("methane_uptake_excess_mmol_g")),
            co2_uptake_absolute_mmol_g=_maybe_float(row.get("co2_uptake_absolute_mmol_g")),
            co2_uptake_excess_mmol_g=_maybe_float(row.get("co2_uptake_excess_mmol_g")),
            h2o_uptake_absolute_mmol_g=_maybe_float(row.get("h2o_uptake_absolute_mmol_g")),
            h2o_uptake_excess_mmol_g=_maybe_float(row.get("h2o_uptake_excess_mmol_g")),
            termination_ok=bool(row.get("termination_ok")),
        )
        score = score_property_metrics(metrics)
        return float(score) if score is not None else float("-inf")


def choose_adsorption_structures(
    csp_workdir: str | Path,
    *,
    max_relative_energy_kj: float = 50.0,
    density_cluster_window: float = 0.10,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    workdir = Path(csp_workdir)
    landscape_csv = workdir / "optimized_landscape_dedup.csv"
    if not landscape_csv.exists():
        landscape_csv = workdir / "optimized_landscape.csv"
    if not landscape_csv.exists():
        raise FileNotFoundError(f"No optimized landscape CSV found in {workdir}")

    representatives = load_spike_representative_points(
        landscape_csv,
        max_relative_energy_kj=max_relative_energy_kj,
        density_cluster_window=density_cluster_window,
    )
    if not representatives:
        raise RuntimeError(f"No spike representatives found in {landscape_csv}")

    representatives = sorted(
        representatives,
        key=lambda row: (float(row["density"]), float(row["relative_energy_kj"])),
    )
    skipped = max(
        representatives,
        key=lambda row: (float(row["density"]), -float(row["relative_energy_kj"])),
    )
    selected_rows = [row for row in representatives if str(row["name"]) != str(skipped["name"])]

    resolved: list[dict[str, object]] = []
    for row in selected_rows:
        name = str(row["name"])
        cif_name = f"{name}_opt.cif"
        structure_path = _resolve_adsorption_cif(workdir, cif_name)
        resolved.append(
            {
                "name": name,
                "density": float(row["density"]),
                "relative_energy_kj": float(row["relative_energy_kj"]),
                "structure_path": str(structure_path),
            }
        )
    return resolved, {
        "name": str(skipped["name"]),
        "density": float(skipped["density"]),
        "relative_energy_kj": float(skipped["relative_energy_kj"]),
    }


def choose_adsorption_structure(
    csp_workdir: str | Path,
    *,
    max_relative_energy_kj: float = 50.0,
    density_cluster_window: float = 0.10,
) -> tuple[str, Path]:
    selected, skipped = choose_adsorption_structures(
        csp_workdir,
        max_relative_energy_kj=max_relative_energy_kj,
        density_cluster_window=density_cluster_window,
    )
    if selected:
        first = selected[0]
        return str(first["name"]), Path(str(first["structure_path"]))
    if skipped is None:
        raise RuntimeError(f"No spike representatives found in {csp_workdir}")
    raise RuntimeError(
        "No porous spike candidates remain after excluding the highest-density spike "
        f"({skipped['name']})."
    )


def _resolve_adsorption_cif(workdir: Path, cif_name: str) -> Path:
    direct = workdir / "relax" / "cif_result_final" / cif_name
    if direct.exists():
        return direct

    restore = workdir / "_restore_relax_artifacts" / "cif_result_final" / cif_name
    if restore.exists():
        return restore

    extracted = workdir / "_adsorption_selected_cif" / cif_name
    extracted.parent.mkdir(parents=True, exist_ok=True)
    if _extract_member_from_archive(workdir / "relax_artifacts.zip", cif_name, extracted):
        return extracted
    if _extract_member_from_archive(workdir / "relax_artifacts.tar.gz", cif_name, extracted):
        return extracted

    raise FileNotFoundError(f"Could not locate CIF for adsorption structure {cif_name} in {workdir}")


def _extract_member_from_archive(archive_path: Path, file_name: str, output_path: Path) -> bool:
    if not archive_path.exists():
        return False
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if member.endswith(file_name):
                    with zf.open(member) as src, output_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
        return False
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".gz":
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith(file_name):
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        return False
                    with extracted, output_path.open("wb") as dst:
                        shutil.copyfileobj(extracted, dst)
                    return True
        return False
    return False


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


__all__ = [
    "AdsorptionInput",
    "AdsorptionSummary",
    "SlurmAdsorptionSettings",
    "SlurmAdsorptionWorkflow",
    "choose_adsorption_structures",
    "choose_adsorption_structure",
]


MethaneAdsorptionInput = AdsorptionInput
MethaneAdsorptionSummary = AdsorptionSummary
SlurmMethaneAdsorptionSettings = SlurmAdsorptionSettings
SlurmMethaneAdsorptionWorkflow = SlurmAdsorptionWorkflow
