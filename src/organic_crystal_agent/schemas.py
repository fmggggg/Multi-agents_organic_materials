from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


OutcomeCategory = Literal[
    "target_like",
    "competing_polymorph",
    "solvate",
    "amorphous",
    "decomposition",
    "inconclusive",
]

PXRDStatus = Literal[
    "target_match",
    "partial_match",
    "mismatch",
    "mixed",
    "weak_pattern",
    "not_collected",
]

ThermalStatus = Literal[
    "consistent",
    "phase_change",
    "solvent_loss",
    "decomposition_signal",
    "not_collected",
]

SingleCrystalStatus = Literal[
    "good_candidate",
    "microcrystalline",
    "powder_only",
    "none_observed",
    "not_checked",
]


FailureKind = Literal[
    "dense_phase",
    "competing_polymorph",
    "solvate_trap",
    "poor_crystallinity",
    "collapse_on_activation",
    "decomposition",
    "inconclusive_characterization",
]


class DiscoveryRequest(BaseModel):
    request: str
    objective: str


class MoleculeCandidate(BaseModel):
    molecule_label: str
    design_hypothesis: str
    target_application: str = "organic crystal discovery"


class CrystalFormHypothesis(BaseModel):
    candidate_id: str
    molecule_label: str
    form_label: str
    packing_hypothesis: str
    expected_advantage: str
    realizability_score: float = Field(ge=0.0, le=1.0, default=0.0)
    polymorph_risk: float = Field(ge=0.0, le=1.0, default=0.0)
    distinguishability_score: float = Field(ge=0.0, le=1.0, default=0.0)
    notes: list[str] = Field(default_factory=list)


class CrystallizationTrial(BaseModel):
    trial_id: str
    candidate_id: str
    round_index: int = Field(ge=1)
    priority: int = Field(ge=1)
    goal: str
    conditions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    expected_information_gain: str
    verification_requirements: list[str] = Field(default_factory=list)


class PXRDAssessment(BaseModel):
    status: PXRDStatus
    notes: list[str] = Field(default_factory=list)


class ThermalAssessment(BaseModel):
    status: ThermalStatus
    observed_mass_loss_percent: float | None = None
    onset_temperature_c: float | None = None
    notes: list[str] = Field(default_factory=list)


class SingleCrystalAssessment(BaseModel):
    status: SingleCrystalStatus
    notes: list[str] = Field(default_factory=list)


class CharacterizationResult(BaseModel):
    trial_id: str
    outcome_category: OutcomeCategory
    summary: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    pxrd: PXRDAssessment | None = None
    thermal: ThermalAssessment | None = None
    single_crystal: SingleCrystalAssessment | None = None


class FailureMode(BaseModel):
    kind: FailureKind
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str


class TrialRecord(BaseModel):
    trial: CrystallizationTrial
    result: CharacterizationResult
    failure_modes: list[FailureMode] = Field(default_factory=list)


class CampaignMemory(BaseModel):
    campaign_id: str = "default-campaign"
    completed_trials: list[TrialRecord] = Field(default_factory=list)
    learned_preferences: list[str] = Field(default_factory=list)

    @property
    def round_index(self) -> int:
        return len(self.completed_trials) + 1


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    tool_name: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class NextExperimentProposal(BaseModel):
    headline: str
    summary: str
    primary_trial: CrystallizationTrial
    backup_trials: list[CrystallizationTrial] = Field(default_factory=list)
    ranked_candidates: list[CrystalFormHypothesis] = Field(default_factory=list)
    research_hints: list[str] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)


MoleculeStatus = Literal[
    "proposed",
    "queued_for_csp",
    "running_csp",
    "landscape_ready",
    "queued_for_property",
    "running_property",
    "property_ready",
    "rejected",
    "selected_for_followup",
]


CampaignStatus = Literal[
    "draft",
    "running",
    "waiting_on_jobs",
    "analysis_ready",
    "completed",
    "failed",
]


class LiteratureHypothesis(BaseModel):
    hypothesis_id: str
    title: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
    design_implications: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class MoleculeRecord(BaseModel):
    molecule_id: str
    molecule_label: str
    molecule_path: str = ""
    smiles: str | None = None
    rationale: str
    source_agent: str = "molecule-design-agent"
    design_family: str = ""
    evidence: list[str] = Field(default_factory=list)
    target_properties: list[str] = Field(default_factory=list)
    status: MoleculeStatus = "proposed"
    notes: list[str] = Field(default_factory=list)


class LandscapeMetrics(BaseModel):
    total_sampled_structures: int = Field(default=0, ge=0)
    optimized_structures: int = Field(default=0, ge=0)
    deduplicated_structures: int | None = None
    deduplication_ratio: float | None = None
    deduplication_method: str = ""
    lowest_energy_kj_mol: float | None = None
    energy_window_kj_mol: float | None = None
    lowest_density_g_cm3: float | None = None
    low_density_spike_score: float | None = None
    porous_survival_rate: float | None = None
    collapse_fraction: float | None = None
    failure_fraction: float | None = None
    low_density_basin_size: int | None = None
    frontier_density_g_cm3: float | None = None
    frontier_energy_kj_mol: float | None = None
    collapse_risk_score: float | None = None
    notes: list[str] = Field(default_factory=list)


class CSPJobRecord(BaseModel):
    molecule_id: str
    molecule_label: str = ""
    smiles: str | None = None
    round_index: int = Field(ge=1, default=1)
    job_id: str
    workflow_name: str
    status: str
    job_path: str = ""
    working_directory: str = ""
    sample_count: int = Field(default=1000, ge=1)
    z_value: int | None = Field(default=None, ge=1)
    space_group: int | None = Field(default=None, ge=1)
    metrics: LandscapeMetrics | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PropertyMetrics(BaseModel):
    property_profile: str = "frontier"
    calculation_mode: str = "sp"
    total_energy_hartree: float | None = None
    homo_ev: float | None = None
    lumo_ev: float | None = None
    gap_ev: float | None = None
    frontier_splitting_ev: float | None = None
    hole_reorganization_energy_ev: float | None = None
    pair_interaction_energy_ev: float | None = None
    methane_uptake_absolute_mmol_g: float | None = None
    methane_uptake_excess_mmol_g: float | None = None
    co2_uptake_absolute_mmol_g: float | None = None
    co2_uptake_excess_mmol_g: float | None = None
    h2o_uptake_absolute_mmol_g: float | None = None
    h2o_uptake_excess_mmol_g: float | None = None
    dipole_debye: float | None = None
    isotropic_polarizability_au: float | None = None
    imaginary_frequency_count: int | None = None
    property_score: float | None = None
    termination_ok: bool | None = None
    method: str = ""
    basis: str = ""
    notes: list[str] = Field(default_factory=list)


class PropertyJobRecord(BaseModel):
    molecule_id: str
    molecule_label: str = ""
    smiles: str | None = None
    round_index: int = Field(ge=1, default=1)
    job_id: str
    workflow_name: str
    status: str
    job_path: str = ""
    working_directory: str = ""
    property_profile: str = "frontier"
    calculation_mode: str = "sp"
    method: str = ""
    basis: str = ""
    metrics: PropertyMetrics | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PackingEvaluationRecord(BaseModel):
    molecule_id: str
    source_csp_job_id: str = ""
    packing_summary_path: str = ""
    pair_batch_manifest_path: str = ""
    cluster_batch_manifest_path: str = ""
    pair_batch_status: str = ""
    cluster_batch_status: str = ""
    best_pair_interaction_energy_ev: float | None = None
    best_frontier_splitting_ev: float | None = None
    best_cluster_energy_kcal_mol: float | None = None
    packing_score: float | None = None
    notes: list[str] = Field(default_factory=list)


class CampaignDecision(BaseModel):
    round_index: int = Field(ge=1)
    selected_molecule_ids: list[str] = Field(default_factory=list)
    rejected_molecule_ids: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class PlannerTrace(BaseModel):
    round_index: int = Field(ge=1)
    planner_mode: str
    available_tools: list[str] = Field(default_factory=list)
    chosen_actions: list[str] = Field(default_factory=list)
    action_rationales: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DiscoveryCampaign(BaseModel):
    campaign_id: str
    target_description: str
    desired_properties: list[str] = Field(default_factory=list)
    status: CampaignStatus = "draft"
    round_index: int = Field(default=1, ge=1)
    literature_hypotheses: list[LiteratureHypothesis] = Field(default_factory=list)
    molecules: list[MoleculeRecord] = Field(default_factory=list)
    csp_jobs: list[CSPJobRecord] = Field(default_factory=list)
    property_jobs: list[PropertyJobRecord] = Field(default_factory=list)
    packing_evaluations: list[PackingEvaluationRecord] = Field(default_factory=list)
    decisions: list[CampaignDecision] = Field(default_factory=list)
    planner_traces: list[PlannerTrace] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def get_molecule(self, molecule_id: str) -> MoleculeRecord:
        for molecule in self.molecules:
            if molecule.molecule_id == molecule_id:
                return molecule
        raise KeyError(f"Unknown molecule_id: {molecule_id}")
