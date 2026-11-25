from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from organic_crystal_agent.remote_workflow import RemoteWorkflowJob, SlurmCSPWorkflow, load_job_from_path, save_job_to_path
from organic_crystal_agent.workflow_state import initialize_stage_state, update_stage_state


LAMMPS_PARAMS = {
    "H": {"mass": 1.008, "epsilon": 0.0157, "sigma": 2.42},
    "C": {"mass": 12.011, "epsilon": 0.105, "sigma": 3.43},
    "N": {"mass": 14.007, "epsilon": 0.069, "sigma": 3.26},
    "O": {"mass": 15.999, "epsilon": 0.060, "sigma": 3.12},
    "S": {"mass": 32.06, "epsilon": 0.274, "sigma": 3.56},
    "F": {"mass": 18.998, "epsilon": 0.050, "sigma": 3.12},
    "CL": {"mass": 35.45, "epsilon": 0.227, "sigma": 3.47},
}


class LAMMPSRelaxInput(BaseModel):
    structure_path: str
    system_label: str
    objective: str = "Run a generic LJ relaxation proxy."
    max_iter: int = Field(default=2000, ge=1)
    nprocs: int = Field(default=4, ge=1)


class LAMMPSRelaxSummary(BaseModel):
    final_potential_energy_kcal_mol: float | None = None
    atom_count: int = 0
    termination_ok: bool = False
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SlurmLAMMPSSettings(BaseModel):
    server_label: str = "slurm-server"
    jobs_root: str = "/users/ffmmgg/scratch/OCAgent/jobs/lammps"
    lammps_binary: str = "lmp"
    lammps_env_activate: str = "module load lammps/20250722-openmpi5.0.8-gcc14.2.0"
    partition: str = "cooper"
    time_limit: str = "08:00:00"
    cpus_per_task: int = 4
    mem_gb: int = 16


class SlurmLAMMPSWorkflow:
    def __init__(self, settings: SlurmLAMMPSSettings | None = None) -> None:
        self.settings = settings or SlurmLAMMPSSettings()

    def submit(self, payload: LAMMPSRelaxInput) -> RemoteWorkflowJob:
        structure_path = Path(payload.structure_path).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        workdir = self._make_job_dir(payload)
        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        staged_structure = inputs_dir / structure_path.name
        shutil.copy2(structure_path, staged_structure)

        manifest = self._build_manifest(payload, workdir, staged_structure)
        (workdir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        initialize_stage_state(workdir, ["relaxation"])
        (workdir / "cluster.data").write_text(self._build_data_file(staged_structure), encoding="utf-8")
        (workdir / "in.relax").write_text(self._build_input_script(manifest), encoding="utf-8")
        submit_script = workdir / "submit_lammps.sh"
        submit_script.write_text(self._build_submit_script(manifest), encoding="utf-8")
        submit_script.chmod(0o755)

        result = subprocess.run(["sbatch", str(submit_script)], check=True, capture_output=True, text=True)
        job_id = SlurmCSPWorkflow._parse_job_id(result.stdout)
        return RemoteWorkflowJob(
            workflow_name="lammps-slurm",
            job_id=job_id,
            status="queued",
            server_label=self.settings.server_label,
            working_directory=str(workdir),
            submit_command=f"sbatch {submit_script}",
            status_command=f"squeue -h -j {job_id} -o %T || sacct -j {job_id} --format=State --parsable2 --noheader",
            fetch_command=f"organic-crystal-agent lammps-fetch --job-dir {workdir}",
            notes=[f"Staged structure: {staged_structure.name}", f"Max iter: {payload.max_iter}"],
        )

    def requeue(self, job: RemoteWorkflowJob) -> RemoteWorkflowJob:
        workdir = Path(job.working_directory)
        submit_script = workdir / "submit_lammps.sh"
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
        queue_result = subprocess.run(["squeue", "-h", "-j", job.job_id, "-o", "%T"], check=False, capture_output=True, text=True)
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

    def fetch_results(self, job: RemoteWorkflowJob) -> LAMMPSRelaxSummary:
        workdir = Path(job.working_directory)
        log_path = workdir / "log.lammps"
        final_xyz = workdir / "final.xyz"
        text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        energy = _match_last_float(text, r"FINAL_POTENTIAL_ENERGY\s*=\s*(-?\d+\.\d+)")
        atoms = 0
        if final_xyz.exists():
            try:
                atoms = int(final_xyz.read_text(encoding="utf-8").splitlines()[0].strip())
            except Exception:
                atoms = 0
        summary = LAMMPSRelaxSummary(
            final_potential_energy_kcal_mol=energy,
            atom_count=atoms,
            termination_ok=energy is not None,
            artifact_paths=[str(path) for path in [workdir / "in.relax", workdir / "cluster.data", log_path, final_xyz] if path.exists()],
            notes=["Generic LJ relaxation proxy based on element-level Lennard-Jones parameters."],
        )
        if summary.termination_ok:
            update_stage_state(workdir, "relaxation", "completed")
        (workdir / "lammps_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return summary

    def _make_job_dir(self, payload: LAMMPSRelaxInput) -> Path:
        # Include microseconds to avoid collisions when multiple jobs are staged within the same second.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        slug = re.sub(r"[^a-z0-9]+", "-", payload.system_label.lower()).strip("-") or "lammps-job"
        path = Path(self.settings.jobs_root).expanduser().resolve() / f"{ts}_{slug}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _build_manifest(self, payload: LAMMPSRelaxInput, workdir: Path, staged_structure: Path) -> dict:
        return {
            "system_label": payload.system_label,
            "structure_path": str(staged_structure),
            "objective": payload.objective,
            "max_iter": payload.max_iter,
            "nprocs": payload.nprocs,
            "settings": self.settings.model_dump(),
            "workdir": str(workdir),
        }

    def _build_data_file(self, xyz_path: Path) -> str:
        atoms = _read_xyz(xyz_path)
        types = sorted({symbol for symbol, _ in atoms})
        type_map = {symbol: index + 1 for index, symbol in enumerate(types)}
        bounds = _bounds([coord for _, coord in atoms], padding=15.0)
        lines = [
            "LAMMPS data file generated by OCAgent",
            "",
            f"{len(atoms)} atoms",
            f"{len(types)} atom types",
            "",
            f"{bounds[0][0]:.6f} {bounds[0][1]:.6f} xlo xhi",
            f"{bounds[1][0]:.6f} {bounds[1][1]:.6f} ylo yhi",
            f"{bounds[2][0]:.6f} {bounds[2][1]:.6f} zlo zhi",
            "",
            "Masses",
            "",
        ]
        for symbol in types:
            params = LAMMPS_PARAMS.get(symbol.upper(), LAMMPS_PARAMS["C"])
            lines.append(f"{type_map[symbol]} {params['mass']:.6f}")
        lines.extend(["", "Atoms # charge", ""])
        for index, (symbol, coord) in enumerate(atoms, start=1):
            lines.append(f"{index} {type_map[symbol]} 0.0 {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}")
        return "\n".join(lines) + "\n"

    def _build_input_script(self, manifest: dict) -> str:
        atoms = _read_xyz(Path(manifest["structure_path"]))
        types = sorted({symbol for symbol, _ in atoms})
        pair_coeff_lines = []
        for index, symbol in enumerate(types, start=1):
            params = LAMMPS_PARAMS.get(symbol.upper(), LAMMPS_PARAMS["C"])
            pair_coeff_lines.append(f"pair_coeff {index} {index} {params['epsilon']:.6f} {params['sigma']:.6f}")
        pair_coeff_block = "\n".join(pair_coeff_lines)
        return f"""units real
atom_style charge
boundary f f f
read_data cluster.data
pair_style lj/cut 12.0
pair_modify mix arithmetic
{pair_coeff_block}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 50
thermo_style custom step pe etotal
min_style cg
minimize 1.0e-6 1.0e-8 {int(manifest['max_iter'])} {int(manifest['max_iter'])}
variable finalpe equal pe
print "FINAL_POTENTIAL_ENERGY = ${{finalpe}}"
write_dump all xyz final.xyz modify element {' '.join(types)}
"""

    def _build_submit_script(self, manifest: dict) -> str:
        settings = manifest["settings"]
        env_activate = f"{settings['lammps_env_activate']}\n" if settings["lammps_env_activate"] else ""
        return f"""#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o slurm.out
#SBATCH -J ocagent_lammps
#SBATCH -p {settings['partition']}
#SBATCH -N 1
#SBATCH -t {settings['time_limit']}
#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}
#SBATCH --mem={int(settings['mem_gb'])}G
set -euo pipefail

WORKDIR='{manifest['workdir']}'
cd "$WORKDIR"
if [ -s final.xyz ] && grep -q 'FINAL_POTENTIAL_ENERGY' log.lammps 2>/dev/null; then
echo "[lammps] relaxation already completed; skipping"
else
python - "$WORKDIR/stage_state.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
state = json.loads(path.read_text()) if path.exists() else {{}}
state["relaxation"] = "running"
path.write_text(json.dumps(state, indent=2))
PY
{env_activate}{settings['lammps_binary']} -in in.relax > log.lammps
fi
"""


def _read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_lines = lines[2:]
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return atoms


def _bounds(coords: list[tuple[float, float, float]], padding: float) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for axis in range(3):
        values = [coord[axis] for coord in coords]
        bounds.append((min(values) - padding, max(values) + padding))
    return bounds


def _match_last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


__all__ = [
    "LAMMPSRelaxInput",
    "LAMMPSRelaxSummary",
    "SlurmLAMMPSSettings",
    "SlurmLAMMPSWorkflow",
    "load_job_from_path",
    "save_job_to_path",
]
