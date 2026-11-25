from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from organic_crystal_agent.crystal_packing import PackingSummary
from organic_crystal_agent.lammps_workflow import LAMMPSRelaxInput, LAMMPSRelaxSummary, SlurmLAMMPSWorkflow
from organic_crystal_agent.remote_workflow import load_job_from_path, save_job_to_path


class ClusterBatchRecord(BaseModel):
    cluster_id: str
    job_path: str = ""
    working_directory: str = ""
    status: str = "queued"
    final_potential_energy_kcal_mol: float | None = None
    notes: list[str] = Field(default_factory=list)


class ClusterBatchSummary(BaseModel):
    packing_summary_path: str
    batch_manifest_path: str
    records: list[ClusterBatchRecord] = Field(default_factory=list)


def submit_cluster_relaxation_batch(
    packing_summary_path: str | Path,
    *,
    top_n: int = 5,
    max_iter: int = 2000,
    workflow: SlurmLAMMPSWorkflow | None = None,
) -> ClusterBatchSummary:
    workflow = workflow or SlurmLAMMPSWorkflow()
    packing_path = Path(packing_summary_path)
    packing = PackingSummary.model_validate(json.loads(packing_path.read_text(encoding="utf-8")))
    selected = packing.clusters[:top_n]
    batch_dir = packing_path.parent / "lammps_cluster_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    records: list[ClusterBatchRecord] = []
    for cluster in selected:
        job = workflow.submit(
            LAMMPSRelaxInput(
                structure_path=cluster.cluster_xyz_path,
                system_label=f"{packing_path.parent.name}_{cluster.cluster_id}",
                objective="Run a local cluster LJ proxy relaxation.",
                max_iter=max_iter,
            )
        )
        job_path = batch_dir / f"{cluster.cluster_id}_job.json"
        save_job_to_path(job_path, job)
        records.append(
            ClusterBatchRecord(
                cluster_id=cluster.cluster_id,
                job_path=str(job_path),
                working_directory=job.working_directory,
                status=job.status,
                notes=list(job.notes),
            )
        )
    summary = ClusterBatchSummary(
        packing_summary_path=str(packing_path),
        batch_manifest_path=str(batch_dir / "cluster_batch_manifest.json"),
        records=records,
    )
    Path(summary.batch_manifest_path).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def refresh_cluster_relaxation_batch(
    batch_manifest_path: str | Path,
    *,
    workflow: SlurmLAMMPSWorkflow | None = None,
) -> ClusterBatchSummary:
    workflow = workflow or SlurmLAMMPSWorkflow()
    manifest_path = Path(batch_manifest_path)
    summary = ClusterBatchSummary.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    updated_records: list[ClusterBatchRecord] = []
    for record in summary.records:
        job = load_job_from_path(record.job_path)
        refreshed = workflow.get_status(job)
        save_job_to_path(record.job_path, refreshed)
        updated = record.model_copy(deep=True)
        updated.status = refreshed.status
        if refreshed.status == "succeeded":
            result: LAMMPSRelaxSummary = workflow.fetch_results(refreshed)
            updated.final_potential_energy_kcal_mol = result.final_potential_energy_kcal_mol
            updated.notes = list(dict.fromkeys(updated.notes + result.notes))
        updated_records.append(updated)
    refreshed_summary = summary.model_copy(update={"records": updated_records})
    manifest_path.write_text(refreshed_summary.model_dump_json(indent=2), encoding="utf-8")
    return refreshed_summary
