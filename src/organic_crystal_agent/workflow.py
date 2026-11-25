from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from organic_crystal_agent.config import AgentConfig
from organic_crystal_agent.persistence import load_campaign_memory, load_proposal, save_campaign_memory, save_proposal
from organic_crystal_agent.schemas import (
    CampaignMemory,
    CharacterizationResult,
    FailureMode,
    NextExperimentProposal,
    Observation,
    PXRDAssessment,
    SingleCrystalAssessment,
    ThermalAssessment,
    ToolCall,
)
from organic_crystal_agent.state import AgentState
from organic_crystal_agent.tools.base import ToolRegistry
from organic_crystal_agent.tools.organic_crystal import (
    enumerate_candidate_forms,
    propose_trial_batch,
    record_trial_result,
    retrieve_external_research_brief,
    score_realizability,
    verify_next_experiment,
)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("retrieve_external_research_brief", retrieve_external_research_brief)
    registry.register("enumerate_candidate_forms", enumerate_candidate_forms)
    registry.register("score_realizability", score_realizability)
    registry.register("propose_trial_batch", propose_trial_batch)
    registry.register("verify_next_experiment", verify_next_experiment)
    return registry


def interpret_goal_node(state: AgentState) -> AgentState:
    config = AgentConfig()
    state["pending_steps"] = [
        ToolCall(
            tool_name="retrieve_external_research_brief",
            tool_input={"query": state["request"], "objective": state["objective"]},
        ),
    ]
    return state


def tool_node(state: AgentState) -> AgentState:
    registry = build_registry()
    step = state["pending_steps"].pop(0)

    if step.tool_name == "score_realizability":
        candidate_obs = _find_observation(state["observations"], "enumerate_candidate_forms")
        step.tool_input = {
            "candidates": candidate_obs.payload["candidates"],
            "campaign_memory": state["campaign_memory"].model_dump(),
        }
    elif step.tool_name == "propose_trial_batch":
        score_obs = _find_observation(state["observations"], "score_realizability")
        research_obs = _find_observation(state["observations"], "retrieve_external_research_brief")
        step.tool_input = {
            "candidates": score_obs.payload["candidates"],
            "campaign_memory": state["campaign_memory"].model_dump(),
            "research_hints": research_obs.payload["hints"],
            "batch_size": AgentConfig().default_batch_size,
        }

    observation = registry.execute(step.tool_name, **step.tool_input)
    state["observations"].append(observation)

    if step.tool_name == "retrieve_external_research_brief":
        state["pending_steps"].append(
            ToolCall(
                tool_name="enumerate_candidate_forms",
                tool_input={
                    "request": state["request"],
                    "objective": state["objective"],
                    "campaign_memory": state["campaign_memory"].model_dump(),
                    "research_hints": observation.payload["hints"],
                    "max_candidates": AgentConfig().max_candidates,
                },
            )
        )
    elif step.tool_name == "enumerate_candidate_forms":
        state["pending_steps"].append(ToolCall(tool_name="score_realizability"))
    elif step.tool_name == "score_realizability":
        state["pending_steps"].append(ToolCall(tool_name="propose_trial_batch"))

    return state


def should_continue_tools(state: AgentState) -> str:
    return "tools" if state["pending_steps"] else "synthesize"


def synthesize_node(state: AgentState) -> AgentState:
    proposal_obs = _find_observation(state["observations"], "propose_trial_batch")
    research_obs = _find_observation(state["observations"], "retrieve_external_research_brief")
    payload = proposal_obs.payload
    verification_notes: list[str] = []
    warning = research_obs.payload.get("warning")
    if warning:
        verification_notes.append(warning)
    proposal = NextExperimentProposal(
        headline="Run one memory-aware crystallization trial before expanding the search.",
        summary=(
            "This platform now prioritizes the single next experiment that best balances "
            "realizability, polymorph risk, and expected information gain. Backup trials are "
            "retained for contingency, but the primary recommendation is intentionally batch-size-one."
        ),
        primary_trial=type_validate_trial(payload["primary_trial"]),
        backup_trials=[type_validate_trial(raw) for raw in payload["backup_trials"]],
        ranked_candidates=[type_validate_candidate(raw) for raw in payload["ranked_candidates"]],
        research_hints=list(payload["research_hints"]),
        verification_notes=verification_notes,
    )
    state["proposal"] = proposal
    if AgentConfig().enable_verification:
        state["pending_steps"] = [ToolCall(tool_name="verify_next_experiment")]
    return state


def verify_or_end(state: AgentState) -> str:
    return "verify" if state["pending_steps"] else END


def verify_node(state: AgentState) -> AgentState:
    registry = build_registry()
    step = state["pending_steps"].pop(0)
    if state["proposal"] is None:
        raise ValueError("Cannot verify before a proposal exists.")
    step.tool_input = {
        "proposal": state["proposal"].model_dump(),
        "campaign_memory": state["campaign_memory"].model_dump(),
    }
    observation = registry.execute(step.tool_name, **step.tool_input)
    state["observations"].append(observation)
    state["proposal"].verification_notes = observation.payload["verification_notes"]
    return state


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("interpret_goal", interpret_goal_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify", verify_node)
    graph.add_edge(START, "interpret_goal")
    graph.add_edge("interpret_goal", "tools")
    graph.add_conditional_edges("tools", should_continue_tools, {"tools": "tools", "synthesize": "synthesize"})
    graph.add_conditional_edges("synthesize", verify_or_end, {"verify": "verify", END: END})
    graph.add_edge("verify", END)
    return graph.compile()


def run_campaign_step(
    request: str,
    objective: str | None = None,
    campaign_memory: CampaignMemory | None = None,
) -> NextExperimentProposal:
    config = AgentConfig()
    app = build_workflow()
    final_state = app.invoke(
        {
            "request": request,
            "objective": objective or config.objective,
            "campaign_memory": campaign_memory or CampaignMemory(),
            "pending_steps": [],
            "observations": [],
            "proposal": None,
        }
    )
    proposal = final_state["proposal"]
    if proposal is None:
        raise ValueError("Workflow completed without a next experiment proposal.")
    return proposal


def update_campaign_memory(
    campaign_memory: CampaignMemory,
    trial,
    result: CharacterizationResult,
    failure_modes: list[FailureMode] | None = None,
) -> CampaignMemory:
    serialized_failures = None
    if failure_modes is not None:
        serialized_failures = [mode.model_dump() for mode in failure_modes]
    return record_trial_result(
        campaign_memory=campaign_memory.model_dump(),
        trial=trial.model_dump(),
        result=result.model_dump(),
        failure_modes=serialized_failures,
    )


def plan_with_paths(
    request: str,
    objective: str | None = None,
    campaign_path: str | None = None,
    proposal_path: str | None = None,
) -> NextExperimentProposal:
    if campaign_path and Path(campaign_path).exists():
        memory = load_campaign_memory(campaign_path)
    else:
        memory = CampaignMemory()
    proposal = run_campaign_step(request=request, objective=objective, campaign_memory=memory)
    if proposal_path:
        save_proposal(proposal_path, proposal)
    if campaign_path:
        save_campaign_memory(campaign_path, memory)
    return proposal


def record_with_paths(
    campaign_path: str,
    proposal_path: str,
    outcome_category: str,
    summary: str,
    evidence: list[str] | None = None,
    confidence: float = 0.7,
    failure_kinds: list[str] | None = None,
    pxrd_status: str | None = None,
    pxrd_note: list[str] | None = None,
    thermal_status: str | None = None,
    thermal_note: list[str] | None = None,
    mass_loss_percent: float | None = None,
    onset_temperature_c: float | None = None,
    single_crystal_status: str | None = None,
    single_crystal_note: list[str] | None = None,
) -> CampaignMemory:
    memory = load_campaign_memory(campaign_path)
    proposal = load_proposal(proposal_path)
    failure_modes = (
        [
            FailureMode(
                kind=kind,
                confidence=confidence,
                rationale=f"Explicitly recorded from CLI outcome as {kind}.",
            )
            for kind in failure_kinds
        ]
        if failure_kinds
        else None
    )
    updated = update_campaign_memory(
        campaign_memory=memory,
        trial=proposal.primary_trial,
        result=CharacterizationResult(
            trial_id=proposal.primary_trial.trial_id,
            outcome_category=outcome_category,
            summary=summary,
            evidence=evidence or [],
            confidence=confidence,
            pxrd=PXRDAssessment(status=pxrd_status, notes=pxrd_note or []) if pxrd_status else None,
            thermal=(
                ThermalAssessment(
                    status=thermal_status,
                    observed_mass_loss_percent=mass_loss_percent,
                    onset_temperature_c=onset_temperature_c,
                    notes=thermal_note or [],
                )
                if thermal_status
                else None
            ),
            single_crystal=(
                SingleCrystalAssessment(status=single_crystal_status, notes=single_crystal_note or [])
                if single_crystal_status
                else None
            ),
        ),
        failure_modes=failure_modes,
    )
    save_campaign_memory(campaign_path, updated)
    return updated


def _find_observation(observations: list[Observation], tool_name: str) -> Observation:
    for observation in observations:
        if observation.tool_name == tool_name:
            return observation
    raise KeyError(f"Missing observation from tool: {tool_name}")


def type_validate_candidate(raw: dict):
    from organic_crystal_agent.schemas import CrystalFormHypothesis

    return CrystalFormHypothesis.model_validate(raw)


def type_validate_trial(raw: dict):
    from organic_crystal_agent.schemas import CrystallizationTrial

    return CrystallizationTrial.model_validate(raw)
