from __future__ import annotations

from pydantic import BaseModel, Field

from organic_crystal_agent.schemas import DiscoveryCampaign, MoleculeRecord


HBOND_MODE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("guanidine", ("guanidine", "guanidinium", "nc(=n)n")),
    ("thiourea", ("thiourea", "nc(=s)n")),
    ("urea", (" urea", "tris(4-ureidophenyl)", "nc(=o)n", "ureido")),
    ("sulfonamide", ("sulfonamide", "s(n)(=o)=o", "sulfonamid")),
    ("amide", ("amide", "c(=o)n")),
    ("amidine", ("amidine", "c(=n)n")),
    ("carboxylic_acid", ("carboxy", "c(=o)o", "co2h")),
    ("pyridyl_acceptor", ("pyrid",)),
    ("triazine_acceptor", ("triazine",)),
]


class CampaignSummary(BaseModel):
    campaign_id: str
    status: str
    round_index: int
    target_description: str
    desired_properties: list[str] = Field(default_factory=list)
    literature_highlights: list[str] = Field(default_factory=list)
    molecule_status_counts: dict[str, int] = Field(default_factory=dict)
    job_status_counts: dict[str, int] = Field(default_factory=dict)
    property_job_status_counts: dict[str, int] = Field(default_factory=dict)
    packing_evaluation_count: int = 0
    top_landscape_findings: list[str] = Field(default_factory=list)
    top_property_findings: list[str] = Field(default_factory=list)
    top_packing_findings: list[str] = Field(default_factory=list)
    hbond_mode_counts: dict[str, int] = Field(default_factory=dict)
    hbond_mode_highlights: list[str] = Field(default_factory=list)
    planner_context: list[str] = Field(default_factory=list)


def _infer_hbond_modes(molecule: MoleculeRecord) -> list[str]:
    text = " ".join(
        filter(
            None,
            [
                molecule.molecule_label,
                molecule.rationale,
                molecule.design_family,
                molecule.smiles or "",
                " ".join(molecule.evidence),
                " ".join(molecule.notes),
            ],
        )
    ).lower()
    matches: list[str] = []
    for mode, patterns in HBOND_MODE_PATTERNS:
        if any(pattern in text for pattern in patterns):
            matches.append(mode)
    return matches or ["unspecified_hbond_mode"]


def build_campaign_summary(campaign: DiscoveryCampaign) -> CampaignSummary:
    literature_highlights = [
        f"{item.title}: {item.summary}" for item in campaign.literature_hypotheses[:5]
    ]
    molecule_status_counts: dict[str, int] = {}
    for molecule in campaign.molecules:
        molecule_status_counts[molecule.status] = molecule_status_counts.get(molecule.status, 0) + 1

    job_status_counts: dict[str, int] = {}
    for job in campaign.csp_jobs:
        job_status_counts[job.status] = job_status_counts.get(job.status, 0) + 1
    property_job_status_counts: dict[str, int] = {}
    for job in campaign.property_jobs:
        property_job_status_counts[job.status] = property_job_status_counts.get(job.status, 0) + 1

    hbond_mode_counts: dict[str, int] = {}
    hbond_mode_members: dict[str, list[str]] = {}
    for molecule in campaign.molecules:
        for mode in _infer_hbond_modes(molecule):
            hbond_mode_counts[mode] = hbond_mode_counts.get(mode, 0) + 1
            hbond_mode_members.setdefault(mode, []).append(f"{molecule.molecule_id} ({molecule.status})")

    findings: list[str] = []
    for job in campaign.csp_jobs:
        if not job.metrics:
            continue
        findings.append(
            f"{job.molecule_id}: unique={job.metrics.deduplicated_structures if job.metrics.deduplicated_structures is not None else 'na'}, "
            f"spike={job.metrics.low_density_spike_score or 0.0:.2f}, "
            f"frontier_density={job.metrics.frontier_density_g_cm3 if job.metrics.frontier_density_g_cm3 is not None else 'na'}, "
            f"collapse_risk={job.metrics.collapse_risk_score if job.metrics.collapse_risk_score is not None else 'na'}"
        )
    property_findings: list[str] = []
    for job in campaign.property_jobs:
        if not job.metrics:
            continue
        metrics = job.metrics
        gas_terms: list[str] = []
        if metrics.co2_uptake_absolute_mmol_g is not None or metrics.h2o_uptake_absolute_mmol_g is not None:
            gas_terms.extend(
                [
                    f"co2_abs={metrics.co2_uptake_absolute_mmol_g if metrics.co2_uptake_absolute_mmol_g is not None else 'na'}",
                    f"co2_exc={metrics.co2_uptake_excess_mmol_g if metrics.co2_uptake_excess_mmol_g is not None else 'na'}",
                    f"h2o_abs={metrics.h2o_uptake_absolute_mmol_g if metrics.h2o_uptake_absolute_mmol_g is not None else 'na'}",
                    f"h2o_exc={metrics.h2o_uptake_excess_mmol_g if metrics.h2o_uptake_excess_mmol_g is not None else 'na'}",
                ]
            )
        elif metrics.methane_uptake_absolute_mmol_g is not None or metrics.methane_uptake_excess_mmol_g is not None:
            gas_terms.extend(
                [
                    f"methane_abs={metrics.methane_uptake_absolute_mmol_g if metrics.methane_uptake_absolute_mmol_g is not None else 'na'}",
                    f"methane_exc={metrics.methane_uptake_excess_mmol_g if metrics.methane_uptake_excess_mmol_g is not None else 'na'}",
                ]
            )
        property_findings.append(
            f"{job.molecule_id}: profile={metrics.property_profile}, "
            f"score={metrics.property_score if metrics.property_score is not None else 'na'}, "
            + (", ".join(gas_terms) + ", " if gas_terms else "")
            + f"gap={metrics.gap_ev if metrics.gap_ev is not None else 'na'}, "
            f"reorg={metrics.hole_reorganization_energy_ev if metrics.hole_reorganization_energy_ev is not None else 'na'}, "
            f"pair={metrics.pair_interaction_energy_ev if metrics.pair_interaction_energy_ev is not None else 'na'}, "
            f"split={metrics.frontier_splitting_ev if metrics.frontier_splitting_ev is not None else 'na'}, "
            f"dipole={metrics.dipole_debye if metrics.dipole_debye is not None else 'na'}, "
            f"polar={metrics.isotropic_polarizability_au if metrics.isotropic_polarizability_au is not None else 'na'}"
        )
    packing_findings: list[str] = []
    for record in campaign.packing_evaluations:
        packing_findings.append(
            f"{record.molecule_id}: packing_score={record.packing_score if record.packing_score is not None else 'na'}, "
            f"pair={record.best_pair_interaction_energy_ev if record.best_pair_interaction_energy_ev is not None else 'na'}, "
            f"split={record.best_frontier_splitting_ev if record.best_frontier_splitting_ev is not None else 'na'}, "
            f"cluster={record.best_cluster_energy_kcal_mol if record.best_cluster_energy_kcal_mol is not None else 'na'}"
        )
    planner_context = []
    if campaign.planner_traces:
        trace = campaign.planner_traces[-1]
        planner_context.append(f"Latest planner mode: {trace.planner_mode}")
        planner_context.append(
            f"Latest actions: {', '.join(trace.chosen_actions) if trace.chosen_actions else 'none'}"
        )
        planner_context.extend(trace.action_rationales[:3])

    hbond_mode_highlights = [
        f"{mode}: {count} molecule(s) | {', '.join(hbond_mode_members.get(mode, [])[:4])}"
        for mode, count in sorted(hbond_mode_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return CampaignSummary(
        campaign_id=campaign.campaign_id,
        status=campaign.status,
        round_index=campaign.round_index,
        target_description=campaign.target_description,
        desired_properties=campaign.desired_properties,
        literature_highlights=literature_highlights,
        molecule_status_counts=molecule_status_counts,
        job_status_counts=job_status_counts,
        property_job_status_counts=property_job_status_counts,
        packing_evaluation_count=len(campaign.packing_evaluations),
        top_landscape_findings=findings[:8],
        top_property_findings=property_findings[:8],
        top_packing_findings=packing_findings[:8],
        hbond_mode_counts=hbond_mode_counts,
        hbond_mode_highlights=hbond_mode_highlights[:8],
        planner_context=planner_context,
    )
