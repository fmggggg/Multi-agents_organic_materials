from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from organic_crystal_agent.literature_agent import get_default_literature_agent
from organic_crystal_agent.naming import normalize_molecule_label
from organic_crystal_agent.known_smiles import ensure_novel_canonical_smiles
from organic_crystal_agent.crystal_packing import extract_packing_features
from organic_crystal_agent.rdkit_utils import canonicalize_smiles, smiles_to_xyz
from organic_crystal_agent.adsorption_workflow import (
    AdsorptionInput,
    AdsorptionSummary,
    SlurmAdsorptionWorkflow,
    choose_adsorption_structures,
)
from organic_crystal_agent.orca_workflow import ORCAPropertyInput, ORCAPropertySummary, get_default_orca_workflow
from organic_crystal_agent.packing_lammps import refresh_cluster_relaxation_batch, submit_cluster_relaxation_batch
from organic_crystal_agent.packing_orca import refresh_pair_interaction_batch, submit_pair_interaction_batch
from organic_crystal_agent.planning_agent import get_default_planning_agent
from organic_crystal_agent.persistence import load_discovery_campaign, save_discovery_campaign
from organic_crystal_agent.remote_workflow import (
    RemoteWorkflowInput,
    SlurmCSPWorkflow,
    load_job_from_path,
    save_job_to_path,
)
from organic_crystal_agent.property_analysis import (
    compute_packing_score,
    rank_campaign_property_candidates,
    score_property_metrics,
    summarize_cluster_batch,
    summarize_pair_batch,
)
from organic_crystal_agent.schemas import (
    CSPJobRecord,
    CampaignDecision,
    DiscoveryCampaign,
    LandscapeMetrics,
    LiteratureHypothesis,
    MoleculeRecord,
    PackingEvaluationRecord,
    PropertyJobRecord,
    PropertyMetrics,
)


class CampaignOrchestrator:
    """Stateful coordinator for long-running CSP discovery campaigns."""

    def __init__(self, workflow: SlurmCSPWorkflow | None = None) -> None:
        self.workflow = workflow or SlurmCSPWorkflow()
        self.orca_workflow = get_default_orca_workflow()
        self.adsorption_workflow = SlurmAdsorptionWorkflow()
        self.literature_agent = get_default_literature_agent()
        self.planning_agent = get_default_planning_agent()

    def create_campaign(
        self,
        campaign_id: str,
        target_description: str,
        desired_properties: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> DiscoveryCampaign:
        return DiscoveryCampaign(
            campaign_id=campaign_id,
            target_description=target_description,
            desired_properties=desired_properties or [],
            notes=notes or [],
        )

    def add_literature_hypothesis(
        self,
        campaign: DiscoveryCampaign,
        title: str,
        summary: str,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        updated.literature_hypotheses.append(
            LiteratureHypothesis(
                hypothesis_id=f"lit-{len(updated.literature_hypotheses) + 1:03d}",
                title=title,
                summary=summary,
                evidence=evidence or [],
                confidence=confidence,
            )
        )
        return updated

    def auto_add_literature_hypotheses(
        self,
        campaign: DiscoveryCampaign,
        max_hypotheses: int = 5,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        new_hypotheses = self.literature_agent.generate_hypotheses(updated, max_hypotheses=max_hypotheses)
        existing_summaries = {item.summary for item in updated.literature_hypotheses}
        for item in new_hypotheses:
            if item.summary in existing_summaries:
                continue
            updated.literature_hypotheses.append(
                item.model_copy(
                    update={
                        "hypothesis_id": f"lit-{len(updated.literature_hypotheses) + 1:03d}",
                    }
                )
            )
        return updated

    def add_molecule(
        self,
        campaign: DiscoveryCampaign,
        molecule_label: str,
        rationale: str,
        molecule_path: str = "",
        smiles: str | None = None,
        source_agent: str = "molecule-design-agent",
        design_family: str = "",
        evidence: list[str] | None = None,
        target_properties: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        clean_label = normalize_molecule_label(molecule_label)
        normalized_smiles = (
            ensure_novel_canonical_smiles(canonicalize_smiles(smiles))
            if smiles
            else smiles
        )
        updated.molecules.append(
            MoleculeRecord(
                molecule_id=f"mol-{self._next_molecule_index(updated):03d}",
                molecule_label=clean_label,
                molecule_path=molecule_path,
                smiles=normalized_smiles,
                rationale=rationale,
                source_agent=source_agent,
                design_family=design_family,
                evidence=evidence or [],
                target_properties=target_properties or [],
                notes=notes or [],
            )
        )
        return updated

    @staticmethod
    def _next_molecule_index(campaign: DiscoveryCampaign) -> int:
        max_seen = 0
        for molecule in campaign.molecules:
            try:
                max_seen = max(max_seen, int(str(molecule.molecule_id).split("-")[-1]))
            except Exception:
                continue
        return max_seen + 1

    def update_molecule(
        self,
        campaign: DiscoveryCampaign,
        molecule_id: str,
        molecule_path: str | None = None,
        smiles: str | None = None,
        rationale: str | None = None,
        design_family: str | None = None,
        evidence: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        molecule = updated.get_molecule(molecule_id)
        if molecule_path is not None:
            molecule.molecule_path = molecule_path
        if smiles is not None:
            molecule.smiles = (
                ensure_novel_canonical_smiles(canonicalize_smiles(smiles))
                if smiles
                else smiles
            )
        if rationale is not None:
            molecule.rationale = rationale
        if design_family is not None:
            molecule.design_family = design_family
        if molecule.molecule_label:
            molecule.molecule_label = normalize_molecule_label(molecule.molecule_label)
        if evidence:
            molecule.evidence.extend(item for item in evidence if item not in molecule.evidence)
        if notes:
            molecule.notes.extend(note for note in notes if note not in molecule.notes)
        return updated

    def auto_add_molecule_batch(
        self,
        campaign: DiscoveryCampaign,
        max_molecules: int = 5,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        incomplete_batch_note = (
            f"No new molecules were registered because the planner did not produce a full valid batch of "
            f"{max_molecules} molecules. Deterministic blueprint/template fallback is disabled; retry the AI "
            "molecule-design step instead."
        )
        existing_labels = {molecule.molecule_label for molecule in updated.molecules}
        existing_smiles = {molecule.smiles for molecule in updated.molecules if molecule.smiles}
        prospective: list[MoleculeRecord] = []

        # The LLM can occasionally return only duplicates or invalid/non-novel SMILES.
        # To keep the discovery loop moving, retry a few times before giving up.
        max_attempts = 3
        for _attempt in range(max_attempts):
            # Ask for a larger draft pool than the target batch size so filters
            # (RDKit validity, novelty, retrosynthesis gating) can still yield a
            # usable batch without requiring repeated heartbeat cycles.
            drafted = self.planning_agent.propose_molecules(updated, max_molecules=max_molecules * 3)
            for item in drafted:
                if item.molecule_label in existing_labels:
                    continue
                if item.smiles:
                    try:
                        item.smiles = ensure_novel_canonical_smiles(canonicalize_smiles(item.smiles))
                    except ValueError:
                        continue
                    if item.smiles in existing_smiles:
                        continue
                existing_labels.add(item.molecule_label)
                if item.smiles:
                    existing_smiles.add(item.smiles)
                prospective.append(item)
                if len(prospective) >= max_molecules:
                    break
            if len(prospective) >= max_molecules:
                break

        if not prospective:
            if incomplete_batch_note not in updated.notes:
                updated.notes.append(incomplete_batch_note)
            return updated

        # We have at least one candidate; register a partial batch instead of dropping
        # everything when the planner fails to reach the target count.
        partial_batch_note = (
            "Registered a partial molecule batch (fewer molecules than requested); "
            "some drafted candidates were filtered out (duplicates or invalid/non-novel SMILES)."
        )
        if len(prospective) < max_molecules:
            if partial_batch_note not in updated.notes:
                updated.notes.append(partial_batch_note)
        else:
            updated.notes = [note for note in updated.notes if note != partial_batch_note]

        updated.notes = [note for note in updated.notes if note != incomplete_batch_note]

        for item in prospective[:max_molecules]:
            updated = self.add_molecule(
                campaign=updated,
                molecule_label=item.molecule_label,
                rationale=item.rationale,
                molecule_path=item.molecule_path,
                smiles=item.smiles,
                source_agent=item.source_agent,
                design_family=item.design_family,
                evidence=item.evidence,
                target_properties=item.target_properties,
                notes=item.notes,
            )

        return updated

    def materialize_molecule_inputs(
        self,
        campaign: DiscoveryCampaign,
        *,
        campaign_root: str | Path,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        inputs_dir = Path(campaign_root).resolve() / "molecule_inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for molecule in updated.molecules:
            if molecule.molecule_path or not molecule.smiles:
                continue
            xyz_path = inputs_dir / f"{molecule.molecule_id}.xyz"
            comment = (
                f"{molecule.molecule_id} | {molecule.molecule_label} | {molecule.smiles} "
                f"| generated {timestamp} via RDKit ETKDGv3+MMFF/UFF"
            )
            try:
                summary = smiles_to_xyz(molecule.smiles, xyz_path, comment=comment)
            except Exception as exc:
                failure_note = f"3D generation failed for SMILES `{molecule.smiles}`: {exc}"
                if failure_note not in molecule.notes:
                    molecule.notes.append(failure_note)
                continue
            molecule.molecule_path = str(xyz_path)
            note = (
                f"3D coords generated {timestamp} via RDKit ETKDGv3+{summary['force_field']}; "
                f"XYZ={xyz_path.name}."
            )
            if note not in molecule.notes:
                molecule.notes.append(note)
        return updated

    def submit_molecule_for_csp(
        self,
        campaign: DiscoveryCampaign,
        molecule_id: str,
        sample_count: int = 1000,
        z_value: int | None = None,
        space_group: int | None = None,
        checkpoint_path: str = "",
        use_real_sg_ratio: bool = False,
        objective: str | None = None,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        molecule = updated.get_molecule(molecule_id)
        if not molecule.molecule_path:
            raise ValueError(f"Molecule {molecule_id} is missing molecule_path.")
        latest_prep = self._latest_pre_csp_dft_job(updated, molecule_id)
        if latest_prep is None:
            return self.submit_molecule_for_pre_csp_dft(
                campaign=updated,
                molecule_id=molecule_id,
                objective=objective or updated.target_description,
            )
        if latest_prep.status in {"queued", "running"}:
            note = (
                "CSP submission deferred until pre-CSP DFT optimization finishes "
                f"(current status: {latest_prep.status}, job_id={latest_prep.job_id})."
            )
            if note not in molecule.notes:
                molecule.notes.append(note)
            updated.status = "waiting_on_jobs"
            return updated
        if latest_prep.status in {"failed", "cancelled"}:
            note = (
                "CSP submission blocked because the latest pre-CSP DFT optimization did not complete "
                f"(status: {latest_prep.status}, job_id={latest_prep.job_id})."
            )
            if note not in molecule.notes:
                molecule.notes.append(note)
            updated.status = "waiting_on_jobs"
            return updated
        if latest_prep.status != "succeeded":
            note = (
                "CSP submission deferred because the latest pre-CSP DFT optimization is not yet in a usable "
                f"terminal state (status: {latest_prep.status}, job_id={latest_prep.job_id})."
            )
            if note not in molecule.notes:
                molecule.notes.append(note)
            updated.status = "waiting_on_jobs"
            return updated
        if any(job.molecule_id == molecule_id and job.status in {"queued", "running"} for job in updated.csp_jobs):
            note = "Skipped duplicate CSP submission because an active CSP job already exists for this molecule."
            if note not in molecule.notes:
                molecule.notes.append(note)
            updated.status = "waiting_on_jobs"
            return updated
        relax_profile = self._choose_relax_profile(updated)

        job_dir_hint = Path(self.workflow.settings.jobs_root) / updated.campaign_id
        job_dir_hint.mkdir(parents=True, exist_ok=True)
        workflow = SlurmCSPWorkflow(
            self.workflow.settings.model_copy(
                update={"jobs_root": str(job_dir_hint)}
            )
        )
        job = workflow.submit(
            RemoteWorkflowInput(
                molecule_id=molecule.molecule_id,
                molecule_label=molecule.molecule_label,
                molecule_path=molecule.molecule_path,
                smiles=molecule.smiles,
                design_family=molecule.design_family,
                rationale=molecule.rationale,
                evidence=list(molecule.evidence),
                proposal_notes=list(molecule.notes),
                objective=objective or updated.target_description,
                sample_count=sample_count,
                z_value=z_value,
                space_group=space_group,
                checkpoint_path=checkpoint_path,
                use_real_sg_ratio=use_real_sg_ratio,
                relax_profile=relax_profile["label"],
                relax_partition=relax_profile["partition"],
                relax_gpus=relax_profile["gpus"],
                relax_n_gpus=relax_profile["n_gpus"],
            )
        )
        job_path = Path(job.working_directory) / "job_status.json"
        save_job_to_path(job_path, job)
        molecule.status = "queued_for_csp"
        updated.status = "waiting_on_jobs"
        updated.csp_jobs.append(
            CSPJobRecord(
                molecule_id=molecule_id,
                molecule_label=molecule.molecule_label,
                smiles=molecule.smiles,
                round_index=updated.round_index,
                job_id=job.job_id,
                workflow_name=job.workflow_name,
                status=job.status,
                job_path=str(job_path),
                working_directory=job.working_directory,
                sample_count=sample_count,
                z_value=z_value,
                space_group=space_group,
                notes=list(job.notes)
                + [
                    (
                        f"Relax profile: {relax_profile['label']} "
                        f"({relax_profile['partition']}, gpu={relax_profile['gpus']}, n_gpus={relax_profile['n_gpus']})"
                    )
                ],
            )
        )
        return updated

    def submit_molecule_for_pre_csp_dft(
        self,
        campaign: DiscoveryCampaign,
        molecule_id: str,
        *,
        objective: str | None = None,
        method: str = "B3LYP",
        basis: str = "def2-SVP",
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        molecule = updated.get_molecule(molecule_id)
        if not molecule.molecule_path:
            raise ValueError(f"Molecule {molecule_id} is missing molecule_path.")

        existing = self._latest_pre_csp_dft_job(updated, molecule_id)
        if existing and existing.status in {"queued", "running", "succeeded"}:
            return updated

        job = self.orca_workflow.submit(
            ORCAPropertyInput(
                molecule_id=molecule.molecule_id,
                molecule_label=molecule.molecule_label,
                structure_path=molecule.molecule_path,
                smiles=molecule.smiles,
                design_family=molecule.design_family,
                rationale=molecule.rationale,
                evidence=list(molecule.evidence),
                proposal_notes=list(molecule.notes),
                objective=objective or "Run DFT geometry optimization before CSP sampling.",
                property_profile="pre_csp_dft_opt",
                calculation_mode="opt",
                method=method,
                basis=basis,
            )
        )
        job_path = Path(job.working_directory) / "job_status.json"
        save_job_to_path(job_path, job)
        original_path = molecule.molecule_path
        molecule.status = "queued_for_property"
        updated.status = "waiting_on_jobs"
        updated.property_jobs.append(
            PropertyJobRecord(
                molecule_id=molecule_id,
                molecule_label=molecule.molecule_label,
                smiles=molecule.smiles,
                round_index=updated.round_index,
                job_id=job.job_id,
                workflow_name=job.workflow_name,
                status=job.status,
                job_path=str(job_path),
                working_directory=job.working_directory,
                property_profile="pre_csp_dft_opt",
                calculation_mode="opt",
                method=method,
                basis=basis,
                notes=list(job.notes)
                + [
                    "Pre-CSP DFT monomer geometry optimization submitted.",
                    f"Original structure path: {original_path}",
                ],
            )
        )
        return updated

    def _choose_relax_profile(self, campaign: DiscoveryCampaign) -> dict[str, object]:
        settings = self.workflow.settings
        return {
            "label": "light-h100-1gpu",
            "partition": settings.light_relax_partition,
            "gpus": settings.light_relax_gpus,
            "n_gpus": settings.light_relax_n_gpus,
        }

    def refresh_campaign(self, campaign: DiscoveryCampaign) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        any_running = False
        any_ready = False
        for job in updated.csp_jobs:
            refreshed = self.workflow.get_status(load_job_from_path(job.job_path))
            save_job_to_path(job.job_path, refreshed)
            job.status = refreshed.status

            molecule = updated.get_molecule(job.molecule_id)
            if refreshed.status in {"queued", "running"}:
                molecule.status = "running_csp" if refreshed.status == "running" else "queued_for_csp"
                any_running = True
                continue

            if refreshed.status == "succeeded":
                summary = self.workflow.fetch_results(refreshed)
                job.artifact_paths = summary.artifact_paths
                job.notes = list(dict.fromkeys(job.notes + summary.notes))
                job.metrics = self._metrics_from_job(refreshed.working_directory, summary)
                molecule.status = "landscape_ready"
                any_ready = True
                continue

            if refreshed.status in {"failed", "cancelled"}:
                molecule.status = "rejected"
                job.notes.append(f"Job ended with status {refreshed.status}.")

        for job in updated.property_jobs:
            job_payload = load_job_from_path(job.job_path)
            property_workflow = self._property_workflow_for_record(job)
            refreshed = property_workflow.get_status(job_payload)
            save_job_to_path(job.job_path, refreshed)
            job.status = refreshed.status

            molecule = updated.get_molecule(job.molecule_id)
            if job.property_profile == "pre_csp_dft_opt":
                if refreshed.status in {"queued", "running"}:
                    molecule.status = "running_property" if refreshed.status == "running" else "queued_for_property"
                    any_running = True
                    continue

                if refreshed.status == "succeeded":
                    summary = property_workflow.fetch_results(refreshed)
                    job.artifact_paths = summary.artifact_paths
                    job.notes = list(dict.fromkeys(job.notes + summary.notes))
                    job.metrics = self._property_metrics_from_summary(summary)
                    if summary.optimized_structure_path:
                        prior_path = molecule.molecule_path
                        molecule.molecule_path = summary.optimized_structure_path
                        has_any_csp = any(csp_job.molecule_id == molecule.molecule_id for csp_job in updated.csp_jobs)
                        if not has_any_csp:
                            molecule.status = "proposed"
                        note = (
                            f"DFT-refined monomer geometry ready for CSP: {summary.optimized_structure_path} "
                            f"(from {prior_path})"
                        )
                        if note not in molecule.notes:
                            molecule.notes.append(note)
                    else:
                        molecule.status = "rejected"
                        failure_note = "Pre-CSP DFT succeeded but no optimized structure could be extracted."
                        if failure_note not in job.notes:
                            job.notes.append(failure_note)
                    continue

                if refreshed.status in {"failed", "cancelled"}:
                    molecule.status = "rejected"
                    job.notes.append(f"Pre-CSP DFT job ended with status {refreshed.status}.")
                    continue

            if refreshed.status in {"queued", "running"}:
                molecule.status = "running_property" if refreshed.status == "running" else "queued_for_property"
                any_running = True
                continue

            if refreshed.status == "succeeded":
                summary = property_workflow.fetch_results(refreshed)
                job.artifact_paths = summary.artifact_paths
                job.notes = list(dict.fromkeys(job.notes + summary.notes))
                job.metrics = self._property_metrics_from_summary(summary)
                molecule.status = "property_ready"
                any_ready = True
                continue

            if refreshed.status in {"failed", "cancelled"}:
                job.notes.append(f"Property job ended with status {refreshed.status}.")

        if updated.packing_evaluations:
            updated = self.refresh_packing_evaluations(updated)
            for record in updated.packing_evaluations:
                if record.pair_batch_status == "running" or record.cluster_batch_status == "running":
                    any_running = True
                if record.packing_score is not None:
                    any_ready = True

        if any_running:
            updated.status = "waiting_on_jobs"
        elif any_ready:
            updated.status = "analysis_ready"
        elif updated.csp_jobs:
            updated.status = "completed"
        return updated

    def submit_molecule_for_property_eval(
        self,
        campaign: DiscoveryCampaign,
        molecule_id: str,
        property_profile: str = "frontier",
        calculation_mode: str = "sp",
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "B3LYP",
        basis: str = "def2-SVP",
        objective: str | None = None,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        molecule = updated.get_molecule(molecule_id)
        if property_profile in {"methane_adsorption", "humid_co2_capture"}:
            source_job = self._latest_succeeded_csp_job(updated, molecule_id)
            selected_structures, skipped_structure = choose_adsorption_structures(source_job.working_directory)
            if not selected_structures:
                skip_note = (
                    "Skipped adsorption: no porous spike candidates remain after excluding "
                    "the highest-density spike representative."
                )
                if skipped_structure:
                    skip_note += f" Excluded spike {skipped_structure['name']}."
                if skip_note not in molecule.notes:
                    molecule.notes.append(skip_note)
                return updated
            job = self.adsorption_workflow.submit(
                AdsorptionInput(
                    molecule_id=molecule.molecule_id,
                    molecule_label=molecule.molecule_label,
                    smiles=molecule.smiles,
                    design_family=molecule.design_family,
                    rationale=molecule.rationale,
                    evidence=list(molecule.evidence),
                    proposal_notes=list(molecule.notes),
                    structures=selected_structures,
                    skipped_spike_names=[str(skipped_structure["name"])] if skipped_structure else [],
                    source_csp_workdir=source_job.working_directory,
                    source_csp_job_id=source_job.job_id,
                    objective=objective or updated.target_description,
                    property_profile=property_profile,
                    calculation_mode=calculation_mode,
                )
            )
            basis = ""
            method = ""
            extra_notes = [
                "Selected adsorption spike structures: "
                + ", ".join(str(item["name"]) for item in selected_structures)
            ]
            if skipped_structure is not None:
                extra_notes.append(
                    "Excluded highest-density spike representative from adsorption: "
                    f"{skipped_structure['name']} (density={skipped_structure['density']:.4f}, "
                    f"ΔE={skipped_structure['relative_energy_kj']:.2f} kJ/mol)"
                )
        else:
            if not molecule.molecule_path:
                raise ValueError(f"Molecule {molecule_id} is missing molecule_path.")
            job = self.orca_workflow.submit(
                ORCAPropertyInput(
                    molecule_id=molecule.molecule_id,
                    molecule_label=molecule.molecule_label,
                    structure_path=molecule.molecule_path,
                    smiles=molecule.smiles,
                    design_family=molecule.design_family,
                    rationale=molecule.rationale,
                    evidence=list(molecule.evidence),
                    proposal_notes=list(molecule.notes),
                    objective=objective or updated.target_description,
                    property_profile=property_profile,
                    calculation_mode=calculation_mode,
                    charge=charge,
                    multiplicity=multiplicity,
                    method=method,
                    basis=basis,
                )
            )
            extra_notes = []
        job_path = Path(job.working_directory) / "job_status.json"
        save_job_to_path(job_path, job)
        molecule.status = "queued_for_property"
        updated.status = "waiting_on_jobs"
        updated.property_jobs.append(
            PropertyJobRecord(
                molecule_id=molecule_id,
                molecule_label=molecule.molecule_label,
                smiles=molecule.smiles,
                round_index=updated.round_index,
                job_id=job.job_id,
                workflow_name=job.workflow_name,
                status=job.status,
                job_path=str(job_path),
                working_directory=job.working_directory,
                property_profile=property_profile,
                calculation_mode=calculation_mode,
                method=method,
                basis=basis,
                notes=list(job.notes) + extra_notes,
            )
        )
        return updated

    def auto_submit_property_jobs(
        self,
        campaign: DiscoveryCampaign,
        property_profile: str = "frontier",
        calculation_mode: str = "sp",
        method: str = "B3LYP",
        basis: str = "def2-SVP",
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        adsorption_profile = self._campaign_adsorption_profile(updated)
        if adsorption_profile is not None:
            property_profile = adsorption_profile
            calculation_mode = "single_point"
            method = ""
            basis = ""
        already_submitted: set[tuple[str, str]] = set()
        for job in updated.property_jobs:
            if job.property_profile != property_profile:
                continue
            if job.status in {"queued", "running"}:
                already_submitted.add((job.molecule_id, job.property_profile))
                continue
            if job.status == "succeeded":
                if property_profile in {"methane_adsorption", "humid_co2_capture"}:
                    metrics = job.metrics
                    if (
                        metrics is not None
                        and (
                            metrics.methane_uptake_absolute_mmol_g is not None
                            or metrics.methane_uptake_excess_mmol_g is not None
                            or metrics.co2_uptake_absolute_mmol_g is not None
                            or metrics.co2_uptake_excess_mmol_g is not None
                            or metrics.h2o_uptake_absolute_mmol_g is not None
                            or metrics.h2o_uptake_excess_mmol_g is not None
                        )
                    ):
                        already_submitted.add((job.molecule_id, job.property_profile))
                else:
                    already_submitted.add((job.molecule_id, job.property_profile))
        candidates = [
            molecule
            for molecule in updated.molecules
            if molecule.status in {"landscape_ready", "selected_for_followup", "property_ready"}
            and molecule.molecule_path
            and not any(note.startswith("Skipped adsorption:") for note in molecule.notes)
        ]
        for molecule in candidates:
            if (molecule.molecule_id, property_profile) in already_submitted:
                continue
            updated = self.submit_molecule_for_property_eval(
                campaign=updated,
                molecule_id=molecule.molecule_id,
                property_profile=property_profile,
                calculation_mode=calculation_mode,
                method=method,
                basis=basis,
            )
        return updated

    def property_ranking(self, campaign: DiscoveryCampaign) -> list[dict[str, object]]:
        return rank_campaign_property_candidates(campaign)

    def submit_molecule_for_packing_analysis(
        self,
        campaign: DiscoveryCampaign,
        molecule_id: str,
        *,
        max_dimers: int = 12,
        cluster_size: int = 4,
        pair_top_n: int = 5,
        cluster_top_n: int = 5,
        pair_method: str = "B3LYP",
        pair_basis: str = "def2-SVP",
        cluster_max_iter: int = 2000,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        csp_jobs = [
            job for job in updated.csp_jobs if job.molecule_id == molecule_id and job.status == "succeeded"
        ]
        if not csp_jobs:
            raise ValueError(f"No succeeded CSP job found for molecule {molecule_id}.")
        source_job = csp_jobs[-1]
        workdir = Path(source_job.working_directory)
        source_cif = workdir / "merged_optimized_dedup.cif"
        if not source_cif.exists():
            source_cif = workdir / "merged_optimized.cif"
        if not source_cif.exists():
            raise FileNotFoundError(f"No merged optimized CIF available for molecule {molecule_id}.")

        packing_dir = workdir / "packing_analysis"
        packing_summary = extract_packing_features(
            source_cif,
            packing_dir,
            max_dimers=max_dimers,
            cluster_size=cluster_size,
        )
        pair_summary = submit_pair_interaction_batch(
            packing_dir / "packing_summary.json",
            top_n=pair_top_n,
            method=pair_method,
            basis=pair_basis,
            workflow=self.orca_workflow,
        )
        cluster_summary = submit_cluster_relaxation_batch(
            packing_dir / "packing_summary.json",
            top_n=cluster_top_n,
            max_iter=cluster_max_iter,
        )
        existing = None
        for record in updated.packing_evaluations:
            if record.molecule_id == molecule_id:
                existing = record
                break
        new_record = PackingEvaluationRecord(
            molecule_id=molecule_id,
            source_csp_job_id=source_job.job_id,
            packing_summary_path=str(packing_dir / "packing_summary.json"),
            pair_batch_manifest_path=pair_summary.batch_manifest_path,
            cluster_batch_manifest_path=cluster_summary.batch_manifest_path,
            pair_batch_status="running",
            cluster_batch_status="running",
            notes=list(dict.fromkeys(packing_summary.notes)),
        )
        if existing is None:
            updated.packing_evaluations.append(new_record)
        else:
            existing.source_csp_job_id = new_record.source_csp_job_id
            existing.packing_summary_path = new_record.packing_summary_path
            existing.pair_batch_manifest_path = new_record.pair_batch_manifest_path
            existing.cluster_batch_manifest_path = new_record.cluster_batch_manifest_path
            existing.pair_batch_status = new_record.pair_batch_status
            existing.cluster_batch_status = new_record.cluster_batch_status
            existing.notes = list(dict.fromkeys(existing.notes + new_record.notes))
        updated.status = "waiting_on_jobs"
        return updated

    def refresh_packing_evaluations(self, campaign: DiscoveryCampaign) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        for record in updated.packing_evaluations:
            notes = list(record.notes)
            if record.pair_batch_manifest_path:
                try:
                    refreshed_pair = refresh_pair_interaction_batch(
                        record.pair_batch_manifest_path,
                        workflow=self.orca_workflow,
                    )
                    pair_summary = summarize_pair_batch(refreshed_pair.batch_manifest_path)
                    record.pair_batch_status = str(pair_summary["status"])
                    record.best_pair_interaction_energy_ev = pair_summary["best_pair_interaction_energy_ev"]  # type: ignore[index]
                    record.best_frontier_splitting_ev = pair_summary["best_frontier_splitting_ev"]  # type: ignore[index]
                except Exception as exc:
                    record.pair_batch_status = "failed"
                    notes.append(f"pair_batch_refresh_failed={exc}")
            if record.cluster_batch_manifest_path:
                try:
                    refreshed_cluster = refresh_cluster_relaxation_batch(record.cluster_batch_manifest_path)
                    cluster_summary = summarize_cluster_batch(refreshed_cluster.batch_manifest_path)
                    record.cluster_batch_status = str(cluster_summary["status"])
                    record.best_cluster_energy_kcal_mol = cluster_summary["best_cluster_energy_kcal_mol"]  # type: ignore[index]
                except Exception as exc:
                    record.cluster_batch_status = "failed"
                    notes.append(f"cluster_batch_refresh_failed={exc}")
            record.packing_score = compute_packing_score(
                record.best_pair_interaction_energy_ev,
                record.best_frontier_splitting_ev,
                record.best_cluster_energy_kcal_mol,
            )
            notes.append(
                f"packing_score={record.packing_score}" if record.packing_score is not None else "packing_score=na"
            )
            record.notes = list(dict.fromkeys(notes))
        return updated

    def packing_ranking(self, campaign: DiscoveryCampaign) -> list[dict[str, object]]:
        rows = []
        for record in campaign.packing_evaluations:
            molecule = campaign.get_molecule(record.molecule_id)
            rows.append(
                {
                    "molecule_id": record.molecule_id,
                    "molecule_label": molecule.molecule_label,
                    "packing_score": record.packing_score,
                    "best_pair_interaction_energy_ev": record.best_pair_interaction_energy_ev,
                    "best_frontier_splitting_ev": record.best_frontier_splitting_ev,
                    "best_cluster_energy_kcal_mol": record.best_cluster_energy_kcal_mol,
                    "pair_batch_status": record.pair_batch_status,
                    "cluster_batch_status": record.cluster_batch_status,
                }
            )
        rows.sort(key=lambda row: (row["packing_score"] is None, -(row["packing_score"] or -1e9), row["molecule_id"]))
        return rows

    def watch_campaign(
        self,
        campaign_path: str,
        poll_seconds: int = 300,
        max_polls: int = 0,
    ) -> DiscoveryCampaign:
        polls = 0
        while True:
            campaign = load_discovery_campaign(campaign_path)
            refreshed = self.refresh_campaign(campaign)
            save_discovery_campaign(campaign_path, refreshed)
            polls += 1
            if refreshed.status != "waiting_on_jobs":
                return refreshed
            if max_polls and polls >= max_polls:
                return refreshed
            time.sleep(poll_seconds)

    def record_decision(
        self,
        campaign: DiscoveryCampaign,
        selected_molecule_ids: list[str],
        rejected_molecule_ids: list[str],
        next_actions: list[str],
        rationale: list[str] | None = None,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        updated.decisions.append(
            CampaignDecision(
                round_index=updated.round_index,
                selected_molecule_ids=selected_molecule_ids,
                rejected_molecule_ids=rejected_molecule_ids,
                next_actions=next_actions,
                rationale=rationale or [],
            )
        )
        for molecule_id in selected_molecule_ids:
            updated.get_molecule(molecule_id).status = "selected_for_followup"
        for molecule_id in rejected_molecule_ids:
            updated.get_molecule(molecule_id).status = "rejected"
        updated.round_index += 1
        updated.status = "running"
        return updated

    def auto_decide_next_round(self, campaign: DiscoveryCampaign) -> DiscoveryCampaign:
        decision = self.planning_agent.decide_next_round(campaign)
        return self.record_decision(
            campaign=campaign,
            selected_molecule_ids=decision.selected_molecule_ids,
            rejected_molecule_ids=decision.rejected_molecule_ids,
            next_actions=decision.next_actions,
            rationale=decision.rationale,
        )

    def _metrics_from_job(self, working_directory: str, summary) -> LandscapeMetrics:
        root = Path(working_directory)
        landscape_csv = root / "optimized_landscape_dedup.csv"
        if not landscape_csv.exists():
            landscape_csv = root / "optimized_landscape.csv"
        rows = []
        if landscape_csv.exists():
            with landscape_csv.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        dedup_summary = {}
        postprocess_summary = root / "postprocess_summary.json"
        if postprocess_summary.exists():
            try:
                dedup_summary = json.loads(postprocess_summary.read_text(encoding="utf-8")).get("dedup_summary", {})
            except Exception:
                dedup_summary = {}

        densities = [float(row["density"]) for row in rows if row.get("density")]
        energies = [float(row["energy_kj"]) for row in rows if row.get("energy_kj")]
        low_density_threshold = 1.0
        low_density_count = sum(1 for density in densities if density < low_density_threshold)
        low_density_spike = None
        frontier_density = None
        frontier_energy = None
        if densities and energies:
            low_pairs = [
                (float(row["density"]), float(row["energy_kj"]))
                for row in rows
                if row.get("density") and row.get("energy_kj") and float(row["density"]) < low_density_threshold
            ]
            if low_pairs and summary.lowest_energy_kj_mol is not None:
                best_low_density_energy = min(energy for _, energy in low_pairs)
                low_density_spike = max(0.0, 10.0 - abs(best_low_density_energy - summary.lowest_energy_kj_mol))
                frontier_density, frontier_energy = min(low_pairs, key=lambda item: item[1])
            elif densities and energies:
                min_idx = min(range(len(energies)), key=lambda i: energies[i])
                frontier_density = densities[min_idx]
                frontier_energy = energies[min_idx]

        failure_fraction = None
        if summary.total_sampled_structures:
            failure_fraction = max(
                0.0,
                1.0 - (summary.optimized_structures / summary.total_sampled_structures),
            )
        porous_survival_rate = None
        if summary.optimized_structures:
            porous_survival_rate = low_density_count / summary.optimized_structures
        collapse_risk = None
        if porous_survival_rate is not None:
            collapse_risk = max(0.0, 1.0 - porous_survival_rate)
        if failure_fraction is not None and collapse_risk is not None:
            collapse_risk = min(1.0, 0.5 * collapse_risk + 0.5 * failure_fraction)

        return LandscapeMetrics(
            total_sampled_structures=summary.total_sampled_structures,
            optimized_structures=summary.optimized_structures,
            deduplicated_structures=dedup_summary.get("kept_rows"),
            deduplication_ratio=(
                (dedup_summary.get("kept_rows") / summary.optimized_structures)
                if dedup_summary.get("kept_rows") is not None and summary.optimized_structures
                else None
            ),
            deduplication_method=str(dedup_summary.get("dedup_method", "")),
            lowest_energy_kj_mol=summary.lowest_energy_kj_mol,
            energy_window_kj_mol=summary.energy_window_kj_mol,
            lowest_density_g_cm3=min(densities) if densities else None,
            low_density_spike_score=low_density_spike,
            porous_survival_rate=porous_survival_rate,
            collapse_fraction=(1.0 - porous_survival_rate) if porous_survival_rate is not None else None,
            failure_fraction=failure_fraction,
            low_density_basin_size=low_density_count,
            frontier_density_g_cm3=frontier_density,
            frontier_energy_kj_mol=frontier_energy,
            collapse_risk_score=collapse_risk,
            notes=list(summary.notes),
        )

    def _property_metrics_from_summary(self, summary: ORCAPropertySummary | AdsorptionSummary) -> PropertyMetrics:
        property_metrics = PropertyMetrics(
            property_profile=summary.property_profile,
            calculation_mode=summary.calculation_mode,
            total_energy_hartree=getattr(summary, "total_energy_hartree", None),
            homo_ev=getattr(summary, "homo_ev", None),
            lumo_ev=getattr(summary, "lumo_ev", None),
            gap_ev=getattr(summary, "gap_ev", None),
            frontier_splitting_ev=getattr(summary, "frontier_splitting_ev", None),
            hole_reorganization_energy_ev=getattr(summary, "hole_reorganization_energy_ev", None),
            pair_interaction_energy_ev=getattr(summary, "pair_interaction_energy_ev", None),
            methane_uptake_absolute_mmol_g=getattr(summary, "methane_uptake_absolute_mmol_g", None),
            methane_uptake_excess_mmol_g=getattr(summary, "methane_uptake_excess_mmol_g", None),
            co2_uptake_absolute_mmol_g=getattr(summary, "co2_uptake_absolute_mmol_g", None),
            co2_uptake_excess_mmol_g=getattr(summary, "co2_uptake_excess_mmol_g", None),
            h2o_uptake_absolute_mmol_g=getattr(summary, "h2o_uptake_absolute_mmol_g", None),
            h2o_uptake_excess_mmol_g=getattr(summary, "h2o_uptake_excess_mmol_g", None),
            dipole_debye=getattr(summary, "dipole_debye", None),
            isotropic_polarizability_au=getattr(summary, "isotropic_polarizability_au", None),
            imaginary_frequency_count=getattr(summary, "imaginary_frequency_count", None),
            termination_ok=summary.termination_ok,
            method=getattr(summary, "method", ""),
            basis=getattr(summary, "basis", ""),
            notes=list(summary.notes),
        )
        if property_metrics.property_profile != "pre_csp_dft_opt":
            property_metrics.property_score = score_property_metrics(property_metrics)
        return property_metrics

    def _property_workflow_for_record(self, job: PropertyJobRecord):
        if job.workflow_name == "raspa-adsorption-slurm" or job.property_profile in {
            "methane_adsorption",
            "humid_co2_capture",
        }:
            return self.adsorption_workflow
        return self.orca_workflow

    @staticmethod
    def _campaign_adsorption_profile(campaign: DiscoveryCampaign) -> str | None:
        haystacks = [campaign.target_description, *campaign.desired_properties]
        text = " ".join(item for item in haystacks if item).lower()
        if "co2" in text and any(token in text for token in ("humid", "h2o", "water")):
            return "humid_co2_capture"
        if "methane" in text and ("adsorption" in text or "uptake" in text):
            return "methane_adsorption"
        return None

    @staticmethod
    def _latest_succeeded_csp_job(campaign: DiscoveryCampaign, molecule_id: str) -> CSPJobRecord:
        matches = [job for job in campaign.csp_jobs if job.molecule_id == molecule_id and job.status == "succeeded"]
        if not matches:
            raise ValueError(f"No succeeded CSP job found for molecule {molecule_id}.")
        return matches[-1]

    @staticmethod
    def _latest_pre_csp_dft_job(campaign: DiscoveryCampaign, molecule_id: str) -> PropertyJobRecord | None:
        matches = [
            job
            for job in campaign.property_jobs
            if job.molecule_id == molecule_id and job.property_profile == "pre_csp_dft_opt"
        ]
        return matches[-1] if matches else None


def save_campaign(path: str, campaign: DiscoveryCampaign) -> None:
    save_discovery_campaign(path, campaign)


def load_campaign(path: str) -> DiscoveryCampaign:
    return load_discovery_campaign(path)
