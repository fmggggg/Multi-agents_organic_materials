from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from organic_crystal_agent.crystal_packing import PackingSummary
from organic_crystal_agent.orca_workflow import ORCAPropertyInput, ORCAPropertySummary, SlurmORCAWorkflow
from organic_crystal_agent.remote_workflow import RemoteWorkflowJob, load_job_from_path, save_job_to_path


class PairBatchRecord(BaseModel):
    dimer_id: str
    transport_proxy_score: float | None = None
    job_path: str = ""
    working_directory: str = ""
    status: str = "queued"
    pair_interaction_energy_ev: float | None = None
    frontier_splitting_ev: float | None = None
    notes: list[str] = Field(default_factory=list)


class PairBatchSummary(BaseModel):
    packing_summary_path: str
    batch_manifest_path: str
    records: list[PairBatchRecord] = Field(default_factory=list)


def submit_pair_interaction_batch(
    packing_summary_path: str | Path,
    *,
    top_n: int = 5,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    workflow: SlurmORCAWorkflow | None = None,
) -> PairBatchSummary:
    workflow = workflow or SlurmORCAWorkflow()
    packing_path = Path(packing_summary_path)
    packing = PackingSummary.model_validate(json.loads(packing_path.read_text(encoding="utf-8")))
    selected = sorted(
        packing.dimers,
        key=lambda item: (-(item.transport_proxy_score or -1e9), item.centroid_distance, item.dimer_id),
    )[:top_n]

    batch_dir = packing_path.parent / "orca_pair_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    records: list[PairBatchRecord] = []
    for dimer in selected:
        job = workflow.submit(
            ORCAPropertyInput(
                molecule_label=f"{packing_path.parent.name}_{dimer.dimer_id}",
                structure_path=dimer.dimer_xyz_path,
                fragment_a_path=dimer.monomer_a_xyz_path,
                fragment_b_path=dimer.monomer_b_xyz_path,
                objective="Estimate dimer pair interaction and frontier splitting from crystal packing.",
                property_profile="pair_interaction",
                calculation_mode="sp",
                method=method,
                basis=basis,
            )
        )
        job_path = batch_dir / f"{dimer.dimer_id}_job.json"
        save_job_to_path(job_path, job)
        records.append(
            PairBatchRecord(
                dimer_id=dimer.dimer_id,
                transport_proxy_score=dimer.transport_proxy_score,
                job_path=str(job_path),
                working_directory=job.working_directory,
                status=job.status,
                notes=list(job.notes),
            )
        )

    summary = PairBatchSummary(
        packing_summary_path=str(packing_path),
        batch_manifest_path=str(batch_dir / "pair_batch_manifest.json"),
        records=records,
    )
    Path(summary.batch_manifest_path).write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def refresh_pair_interaction_batch(
    batch_manifest_path: str | Path,
    *,
    workflow: SlurmORCAWorkflow | None = None,
) -> PairBatchSummary:
    workflow = workflow or SlurmORCAWorkflow()
    manifest_path = Path(batch_manifest_path)
    summary = PairBatchSummary.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    updated_records: list[PairBatchRecord] = []
    for record in summary.records:
        job = load_job_from_path(record.job_path)
        refreshed = workflow.get_status(job)
        save_job_to_path(record.job_path, refreshed)
        updated_record = record.model_copy(deep=True)
        updated_record.status = refreshed.status
        if refreshed.status == "succeeded":
            result: ORCAPropertySummary = workflow.fetch_results(refreshed)
            updated_record.pair_interaction_energy_ev = result.pair_interaction_energy_ev
            updated_record.frontier_splitting_ev = result.frontier_splitting_ev
            updated_record.notes = list(dict.fromkeys(updated_record.notes + result.notes))
        updated_records.append(updated_record)
    refreshed_summary = summary.model_copy(update={"records": updated_records})
    manifest_path.write_text(refreshed_summary.model_dump_json(indent=2), encoding="utf-8")
    return refreshed_summary
