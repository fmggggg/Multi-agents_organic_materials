from __future__ import annotations

from organic_crystal_agent.generator import get_default_generator
from organic_crystal_agent.policy import ReasoningPolicy
from organic_crystal_agent.research import MockOpenAIResearchBackend, get_default_research_backend
from organic_crystal_agent.schemas import (
    CampaignMemory,
    CharacterizationResult,
    CrystalFormHypothesis,
    CrystallizationTrial,
    FailureMode,
    NextExperimentProposal,
    Observation,
    TrialRecord,
)


def retrieve_external_research_brief(query: str, objective: str) -> Observation:
    backend = get_default_research_backend()
    warning: str | None = None
    try:
        hints = backend.search(query=query, objective=objective)
    except RuntimeError as exc:
        fallback = MockOpenAIResearchBackend()
        hints = fallback.search(query=query, objective=objective)
        warning = f"OpenAI backend unavailable, fell back to mock research backend: {exc}"
    return Observation(
        tool_name="retrieve_external_research_brief",
        summary="Retrieved a short research brief from the default external research backend.",
        payload={"hints": hints, "warning": warning},
    )


def enumerate_candidate_forms(
    request: str,
    objective: str,
    campaign_memory: dict,
    research_hints: list[str],
    max_candidates: int = 3,
) -> Observation:
    generator = get_default_generator()
    archetypes = generator.generate(
        request=request,
        objective=objective,
        campaign_memory=CampaignMemory.model_validate(campaign_memory),
        research_hints=research_hints,
        max_candidates=max_candidates,
    )
    return Observation(
        tool_name="enumerate_candidate_forms",
        summary="Enumerated initial crystal-form hypotheses for the current campaign step.",
        payload={
            "request": request,
            "objective": objective,
            "candidates": [candidate.model_dump() for candidate in archetypes[:max_candidates]],
        },
    )


def score_realizability(candidates: list[dict], campaign_memory: dict) -> Observation:
    memory = CampaignMemory.model_validate(campaign_memory)
    scored: list[CrystalFormHypothesis] = []
    dense_penalty = _memory_penalty(memory, "dense_phase")
    polymorph_penalty = _memory_penalty(memory, "competing_polymorph")
    solvate_penalty = _memory_penalty(memory, "solvate_trap")

    for raw in candidates:
        candidate = CrystalFormHypothesis.model_validate(raw)
        if "open" in candidate.packing_hypothesis or "porous" in candidate.packing_hypothesis:
            candidate.realizability_score = 0.42 - 0.05 * solvate_penalty
            candidate.polymorph_risk = 0.76 + 0.05 * solvate_penalty
            candidate.distinguishability_score = 0.61
        elif "hydrogen-bond" in candidate.molecule_label.lower():
            candidate.realizability_score = 0.60 - 0.04 * polymorph_penalty
            candidate.polymorph_risk = 0.62 + 0.06 * polymorph_penalty
            candidate.distinguishability_score = 0.75
        else:
            candidate.realizability_score = 0.74 - 0.03 * dense_penalty
            candidate.polymorph_risk = 0.29 + 0.03 * dense_penalty
            candidate.distinguishability_score = 0.72
        scored.append(candidate)

    ranked = sorted(
        scored,
        key=lambda item: (
            item.realizability_score
            - 0.5 * item.polymorph_risk
            + 0.2 * item.distinguishability_score
        ),
        reverse=True,
    )
    return Observation(
        tool_name="score_realizability",
        summary="Scored candidate crystal forms using campaign-aware heuristic penalties.",
        payload={"candidates": [candidate.model_dump() for candidate in ranked]},
    )


def propose_trial_batch(
    candidates: list[dict],
    campaign_memory: dict,
    research_hints: list[str],
    batch_size: int = 1,
) -> Observation:
    memory = CampaignMemory.model_validate(campaign_memory)
    ranked = [CrystalFormHypothesis.model_validate(raw) for raw in candidates]
    policy = ReasoningPolicy()
    primary, backups = policy.propose_trials(ranked, memory, batch_size=batch_size)
    return Observation(
        tool_name="propose_trial_batch",
        summary="Proposed the next crystallization trial and ordered backups.",
        payload={
            "primary_trial": primary.model_dump(),
            "backup_trials": [trial.model_dump() for trial in backups],
            "ranked_candidates": [candidate.model_dump() for candidate in ranked],
            "research_hints": research_hints,
        },
    )


def verify_next_experiment(proposal: dict, campaign_memory: dict) -> Observation:
    next_step = NextExperimentProposal.model_validate(proposal)
    memory = CampaignMemory.model_validate(campaign_memory)
    notes = [
        "Primary trial is expressed as a concrete experiment rather than a generic recommendation.",
        "Verification couples the selected trial to immediate PXRD and thermal checks.",
        "Batch size remains effectively one: backups exist for contingency, not parallel overcommitment.",
    ]
    if memory.completed_trials:
        notes.append(
            f"Proposal accounts for {len(memory.completed_trials)} prior trial(s) recorded in campaign memory."
        )
    return Observation(
        tool_name="verify_next_experiment",
        summary="Validated that the proposal is concrete, memory-aware, and evidence-seeking.",
        payload={"verification_notes": notes, "headline": next_step.headline},
    )


def record_trial_result(
    campaign_memory: dict,
    trial: dict,
    result: dict,
    failure_modes: list[dict] | None = None,
) -> CampaignMemory:
    memory = CampaignMemory.model_validate(campaign_memory)
    characterization = CharacterizationResult.model_validate(result)
    inferred_failures = [
        FailureMode.model_validate(item)
        for item in (failure_modes or _infer_failure_modes(characterization))
    ]
    record = TrialRecord(
        trial=CrystallizationTrial.model_validate(trial),
        result=characterization,
        failure_modes=inferred_failures,
    )
    updated = memory.model_copy(deep=True)
    updated.completed_trials.append(record)
    learned = _learned_preferences_from_record(record)
    updated.learned_preferences.extend(preference for preference in learned if preference not in updated.learned_preferences)
    return updated


def build_trial_for_policy(
    candidate: CrystalFormHypothesis,
    round_index: int,
    priority: int,
    memory: CampaignMemory,
) -> CrystallizationTrial:
    conditions = _base_conditions(candidate)
    rationale = [
        f"Targets {candidate.form_label} form of {candidate.molecule_label}.",
        f"Balances realizability {candidate.realizability_score:.2f} against polymorph risk {candidate.polymorph_risk:.2f}.",
    ]

    if _memory_penalty(memory, "competing_polymorph") > 0 and candidate.candidate_id == "cand-beta":
        conditions.append("Bias toward tighter solvent windows and seeded repeats to reduce polymorph drift.")
        rationale.append("Campaign memory indicates prior competing-polymorph behavior.")

    if _memory_penalty(memory, "dense_phase") > 0 and candidate.candidate_id == "cand-alpha":
        conditions.append("Capture intermediate aliquots before full evaporation to detect premature dense packing.")
        rationale.append("Campaign memory suggests early dense packing is a recurrent failure mode.")

    if _memory_penalty(memory, "solvate_trap") > 0 and candidate.candidate_id == "cand-gamma":
        conditions.append("Preserve solvated material and defer activation until phase identity is clearer.")
        rationale.append("Campaign memory indicates solvate trapping should be explicitly checked.")

    return CrystallizationTrial(
        trial_id=f"round-{round_index:02d}-priority-{priority}",
        candidate_id=candidate.candidate_id,
        round_index=round_index,
        priority=priority,
        goal=f"Test whether {candidate.form_label} can be obtained reproducibly as the next most informative crystal form.",
        conditions=conditions,
        rationale=rationale,
        expected_information_gain=(
            "Clarifies whether the candidate is experimentally accessible and whether failure, if any, "
            "comes from polymorph competition, dense packing, or solvate sensitivity."
        ),
        verification_requirements=[
            "Collect PXRD on the isolated solid before extended aging.",
            "Pair PXRD with DSC or TGA for phase-change evidence.",
            "Record whether crystals are single-crystal candidates, powder only, or amorphous.",
        ],
    )


def _base_conditions(candidate: CrystalFormHypothesis) -> list[str]:
    if candidate.candidate_id == "cand-alpha":
        return [
            "Start with slow vapor diffusion using a moderately volatile antisolvent.",
            "Run room-temperature evaporation before adding temperature cycling.",
            "Keep concentration moderate to avoid immediate dense precipitation.",
        ]
    if candidate.candidate_id == "cand-beta":
        return [
            "Use a narrow solvent panel with controlled hydrogen-bond competition.",
            "Compare seeded and unseeded runs.",
            "Limit condition changes to one main variable in this round.",
        ]
    return [
        "Treat this as a benchmark probe rather than a primary optimization lead.",
        "Maintain mother-liquor characterization to detect solvent-stabilized phases.",
        "Avoid aggressive activation in the same round.",
    ]


def _memory_penalty(memory: CampaignMemory, failure_kind: str) -> int:
    return sum(
        1
        for record in memory.completed_trials
        for failure in record.failure_modes
        if failure.kind == failure_kind
    )


def _learned_preferences_from_record(record: TrialRecord) -> list[str]:
    preferences: list[str] = []
    for failure in record.failure_modes:
        if failure.kind == "dense_phase":
            preferences.append("Favor earlier PXRD checkpoints before complete solvent loss.")
        elif failure.kind == "competing_polymorph":
            preferences.append("Bias future trials toward narrower solvent variation and seeding controls.")
        elif failure.kind == "solvate_trap":
            preferences.append("Treat solvent retention as a tracked variable in subsequent rounds.")
        elif failure.kind == "poor_crystallinity":
            preferences.append("Reduce condition spread and emphasize crystal growth diagnostics in future rounds.")
        elif failure.kind == "inconclusive_characterization":
            preferences.append("Require stronger paired characterization before declaring phase identity.")
    return preferences


def _infer_failure_modes(result: CharacterizationResult) -> list[dict]:
    inferred: list[dict] = []
    pxrd_status = result.pxrd.status if result.pxrd else None
    thermal_status = result.thermal.status if result.thermal else None
    mass_loss = result.thermal.observed_mass_loss_percent if result.thermal else None
    sc_status = result.single_crystal.status if result.single_crystal else None

    if thermal_status == "decomposition_signal":
        inferred.append(
            {
                "kind": "decomposition",
                "confidence": max(0.75, result.confidence),
                "rationale": "Thermal analysis indicates decomposition during heating or handling.",
            }
        )

    if thermal_status == "solvent_loss" or (
        mass_loss is not None and mass_loss >= 3.0 and result.outcome_category in {"solvate", "inconclusive"}
    ):
        inferred.append(
            {
                "kind": "solvate_trap",
                "confidence": max(0.65, result.confidence),
                "rationale": "Thermal behavior is consistent with significant retained solvent or guest loss.",
            }
        )

    if pxrd_status in {"mismatch", "mixed", "partial_match"}:
        inferred.append(
            {
                "kind": "competing_polymorph" if pxrd_status != "partial_match" else "inconclusive_characterization",
                "confidence": max(0.6, result.confidence),
                "rationale": "PXRD does not cleanly support the intended target phase.",
            }
        )

    if sc_status in {"powder_only", "none_observed"} and result.outcome_category in {"amorphous", "inconclusive"}:
        inferred.append(
            {
                "kind": "poor_crystallinity",
                "confidence": max(0.6, result.confidence),
                "rationale": "No useful single-crystal candidates were observed and phase order appears poor.",
            }
        )

    if result.outcome_category == "target_like":
        return _dedupe_failures(inferred)
    if result.outcome_category == "competing_polymorph":
        inferred.append(
            {
                "kind": "competing_polymorph",
                "confidence": max(0.6, result.confidence),
                "rationale": "Characterization suggests a non-target crystalline phase.",
            }
        )
        return _dedupe_failures(inferred)
    if result.outcome_category == "solvate":
        inferred.append(
            {
                "kind": "solvate_trap",
                "confidence": max(0.6, result.confidence),
                "rationale": "Observed phase appears strongly solvent-associated.",
            }
        )
        return _dedupe_failures(inferred)
    if result.outcome_category == "amorphous":
        inferred.append(
            {
                "kind": "poor_crystallinity",
                "confidence": max(0.6, result.confidence),
                "rationale": "Recovered material lacks convincing crystalline order.",
            }
        )
        return _dedupe_failures(inferred)
    if result.outcome_category == "decomposition":
        inferred.append(
            {
                "kind": "decomposition",
                "confidence": max(0.7, result.confidence),
                "rationale": "Material appears chemically degraded during or after crystallization.",
            }
        )
        return _dedupe_failures(inferred)
    inferred.append(
        {
            "kind": "inconclusive_characterization",
            "confidence": max(0.5, result.confidence),
            "rationale": "Available evidence does not conclusively identify the phase outcome.",
        }
    )
    return _dedupe_failures(inferred)


def _dedupe_failures(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        kind = str(item["kind"])
        if kind in seen:
            continue
        seen.add(kind)
        deduped.append(item)
    return deduped
