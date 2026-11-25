from __future__ import annotations

import json
import math
from pathlib import Path

from organic_crystal_agent.schemas import DiscoveryCampaign, PropertyMetrics


def score_property_metrics(metrics: PropertyMetrics) -> float | None:
    if (
        metrics.co2_uptake_absolute_mmol_g is not None
        or metrics.h2o_uptake_absolute_mmol_g is not None
    ):
        co2_abs = metrics.co2_uptake_absolute_mmol_g or 0.0
        h2o_abs = metrics.h2o_uptake_absolute_mmol_g or 0.0
        # APS-style humid-capture objective: combine absolute CO2 uptake with
        # a compressed selectivity term so very high-selectivity/very-low-
        # uptake outliers do not dominate the ranking.
        selectivity_ratio = (co2_abs + 1e-3) / (h2o_abs + 1e-3)
        score = co2_abs * math.sqrt(selectivity_ratio)
        if metrics.termination_ok is False:
            score -= 5.0
        return round(score, 6)
    if metrics.methane_uptake_absolute_mmol_g is not None:
        score = metrics.methane_uptake_absolute_mmol_g
        if metrics.methane_uptake_excess_mmol_g is not None:
            score += 0.1 * metrics.methane_uptake_excess_mmol_g
        if metrics.termination_ok is False:
            score -= 5.0
        return round(score, 6)
    score = 0.0
    used = 0
    if metrics.gap_ev is not None:
        score += max(0.0, 6.0 - abs(metrics.gap_ev - 3.0))
        used += 1
    if metrics.hole_reorganization_energy_ev is not None:
        score += max(0.0, 1.0 - metrics.hole_reorganization_energy_ev)
        used += 1
    if metrics.pair_interaction_energy_ev is not None:
        score += max(0.0, min(5.0, -metrics.pair_interaction_energy_ev / 0.2))
        used += 1
    if metrics.frontier_splitting_ev is not None:
        score += min(metrics.frontier_splitting_ev / 0.05, 5.0)
        used += 1
    if metrics.dipole_debye is not None:
        score += max(0.0, 3.0 - abs(metrics.dipole_debye - 1.5))
        used += 1
    if metrics.isotropic_polarizability_au is not None:
        score += min(metrics.isotropic_polarizability_au / 30.0, 3.0)
        used += 1
    if metrics.termination_ok is False:
        score -= 2.0
    return round(score / used, 6) if used else None


def rank_campaign_property_candidates(campaign: DiscoveryCampaign) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics_by_molecule = {
        job.molecule_id: job.metrics
        for job in campaign.property_jobs
        if job.metrics is not None
    }
    csp_by_molecule = {
        job.molecule_id: job.metrics
        for job in campaign.csp_jobs
        if job.metrics is not None
    }
    for molecule in campaign.molecules:
        property_metrics = metrics_by_molecule.get(molecule.molecule_id)
        if property_metrics is None:
            continue
        property_score = score_property_metrics(property_metrics)
        csp_metrics = csp_by_molecule.get(molecule.molecule_id)
        landscape_bonus = 0.0
        if csp_metrics is not None:
            if csp_metrics.low_density_spike_score is not None:
                landscape_bonus += csp_metrics.low_density_spike_score / 10.0
            if csp_metrics.collapse_risk_score is not None:
                landscape_bonus += max(0.0, 1.0 - csp_metrics.collapse_risk_score)
        total_score = None if property_score is None else round(property_score + landscape_bonus, 6)
        rows.append(
            {
                "molecule_id": molecule.molecule_id,
                "molecule_label": molecule.molecule_label,
                "property_score": property_score,
                "combined_score": total_score,
                "gap_ev": property_metrics.gap_ev,
                "hole_reorganization_energy_ev": property_metrics.hole_reorganization_energy_ev,
                "pair_interaction_energy_ev": property_metrics.pair_interaction_energy_ev,
                "methane_uptake_absolute_mmol_g": property_metrics.methane_uptake_absolute_mmol_g,
                "methane_uptake_excess_mmol_g": property_metrics.methane_uptake_excess_mmol_g,
                "co2_uptake_absolute_mmol_g": property_metrics.co2_uptake_absolute_mmol_g,
                "co2_uptake_excess_mmol_g": property_metrics.co2_uptake_excess_mmol_g,
                "h2o_uptake_absolute_mmol_g": property_metrics.h2o_uptake_absolute_mmol_g,
                "h2o_uptake_excess_mmol_g": property_metrics.h2o_uptake_excess_mmol_g,
                "frontier_splitting_ev": property_metrics.frontier_splitting_ev,
                "dipole_debye": property_metrics.dipole_debye,
                "polarizability_au": property_metrics.isotropic_polarizability_au,
                "low_density_spike_score": None if csp_metrics is None else csp_metrics.low_density_spike_score,
                "collapse_risk_score": None if csp_metrics is None else csp_metrics.collapse_risk_score,
            }
        )
    rows.sort(key=lambda row: (row["combined_score"] is None, -(row["combined_score"] or -1e9), row["molecule_id"]))
    return rows


def compute_packing_score(
    best_pair_interaction_energy_ev: float | None,
    best_frontier_splitting_ev: float | None,
    best_cluster_energy_kcal_mol: float | None,
) -> float | None:
    score = 0.0
    used = 0
    if best_pair_interaction_energy_ev is not None:
        score += max(0.0, min(6.0, -best_pair_interaction_energy_ev / 0.2))
        used += 1
    if best_frontier_splitting_ev is not None:
        score += min(best_frontier_splitting_ev / 0.05, 6.0)
        used += 1
    if best_cluster_energy_kcal_mol is not None:
        score += max(0.0, min(4.0, -best_cluster_energy_kcal_mol / 10.0))
        used += 1
    return round(score / used, 6) if used else None


def summarize_pair_batch(batch_manifest_path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(batch_manifest_path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    pair_values = [float(item["pair_interaction_energy_ev"]) for item in records if item.get("pair_interaction_energy_ev") is not None]
    split_values = [float(item["frontier_splitting_ev"]) for item in records if item.get("frontier_splitting_ev") is not None]

    terminal = {"succeeded", "failed", "cancelled"}
    statuses = [item.get("status") for item in records]
    if not records:
        status = "completed"
    elif any(state in {"queued", "running"} for state in statuses):
        status = "running"
    elif all(state in terminal for state in statuses):
        status = "completed"
    else:
        status = "running"

    return {
        "status": status,
        "best_pair_interaction_energy_ev": min(pair_values) if pair_values else None,
        "best_frontier_splitting_ev": max(split_values) if split_values else None,
    }


def summarize_cluster_batch(batch_manifest_path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(batch_manifest_path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    energies = [float(item["final_potential_energy_kcal_mol"]) for item in records if item.get("final_potential_energy_kcal_mol") is not None]

    terminal = {"succeeded", "failed", "cancelled"}
    statuses = [item.get("status") for item in records]
    if not records:
        status = "completed"
    elif any(state in {"queued", "running"} for state in statuses):
        status = "running"
    elif all(state in terminal for state in statuses):
        status = "completed"
    else:
        status = "running"

    return {
        "status": status,
        "best_cluster_energy_kcal_mol": min(energies) if energies else None,
    }
