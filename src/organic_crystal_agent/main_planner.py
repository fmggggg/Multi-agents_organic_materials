from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from organic_crystal_agent.campaign_summary import CampaignSummary, build_campaign_summary
from organic_crystal_agent.config import resolve_openai_model_env
from organic_crystal_agent.memory_store import MemoryStore, resolve_campaign_root
from organic_crystal_agent.orchestrator import CampaignOrchestrator
from organic_crystal_agent.planner_tools import PlannerToolRegistry, build_planner_tool_registry
from organic_crystal_agent.schemas import DiscoveryCampaign, PlannerTrace


@dataclass
class PlannerConfig:
    literature_hypotheses: int = 5
    molecule_batch_size: int = 5
    sample_count: int = 1000
    z_value: int | None = None
    space_group: int | None = None
    use_real_sg_ratio: bool = False
    checkpoint_path: str = ""
    auto_decide: bool = True
    auto_expand: bool = True


@dataclass
class PlannerActionSelection:
    actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    action_rationales: list[str] = field(default_factory=list)


class MainPlannerAgent(Protocol):
    def choose_actions(
        self,
        campaign: DiscoveryCampaign,
        campaign_summary: CampaignSummary,
        memory_bundle: dict[str, str],
        available_tools: list[str],
        config: PlannerConfig,
    ) -> PlannerActionSelection:
        ...


class HeuristicMainPlannerAgent:
    """Default single-agent planner choosing tool actions from campaign state and memory."""

    def choose_actions(
        self,
        campaign: DiscoveryCampaign,
        campaign_summary: CampaignSummary,
        memory_bundle: dict[str, str],
        available_tools: list[str],
        config: PlannerConfig,
    ) -> PlannerActionSelection:
        actions: list[str] = []
        rationales: list[str] = []
        notes = ["Heuristic planner used campaign state and file-backed memory."]

        # If recent molecule proposal attempts have repeatedly failed to register any
        # new candidates, temporarily suppress automatic expansion to avoid churning
        # identical planner rounds. Do not let a stale historical failure permanently
        # disable future proposal attempts.
        proposal_failure_marker = "No new molecules were registered because the planner did not produce a full valid batch"
        recent_failure_streak = self._trailing_note_matches(campaign.notes, proposal_failure_marker)
        auto_expand_ok = config.auto_expand and recent_failure_streak < 3
        if recent_failure_streak:
            notes.append(
                f"Recent proposal failure streak: {recent_failure_streak}; auto-expand remains enabled until three consecutive failed proposal notes accumulate."
            )
        property_mode = self._campaign_wants_property_evaluation(campaign)
        property_tracking_candidates = [
            molecule
            for molecule in campaign.molecules
            if molecule.status in {"landscape_ready", "selected_for_followup", "property_ready"}
            and molecule.molecule_path
            and not self._property_evaluation_is_skipped(molecule)
        ]
        if not campaign.literature_hypotheses and "collect_literature" in available_tools:
            actions.append("collect_literature")
            rationales.append("No literature hypotheses exist yet, so literature collection is required first.")
        if not campaign.molecules and "propose_molecules" in available_tools:
            actions.append("propose_molecules")
            rationales.append("No molecules are registered yet, so the planner proposes a first batch.")

        active_jobs = [job for job in campaign.csp_jobs if job.status in {"queued", "running"}]
        active_property_jobs = [job for job in campaign.property_jobs if job.status in {"queued", "running"}]
        has_property_jobs = bool(campaign.property_jobs)
        property_missing = property_mode and any(
            self._needs_property_evaluation(campaign, molecule.molecule_id)
            for molecule in property_tracking_candidates
        )

        if property_mode and property_missing and "submit_property_jobs" in available_tools:
            if active_jobs and "refresh_jobs" in available_tools:
                actions.append("refresh_jobs")
                rationales.append("Active CSP jobs are still running, so job state should be refreshed first.")
            actions.append("submit_property_jobs")
            rationales.append(
                "Property-focused campaigns should start evaluating completed landscape representatives "
                "without waiting for every remaining CSP job to finish."
            )
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        property_results_ready = any(self._job_has_useful_property_metrics(job) for job in campaign.property_jobs)
        has_proposed = any(molecule.status == "proposed" for molecule in campaign.molecules)
        if property_mode and property_results_ready and active_jobs and not has_proposed and "propose_molecules" in available_tools:
            if "refresh_jobs" in available_tools:
                actions.append("refresh_jobs")
                rationales.append("Active jobs remain, so refresh current CSP and property status first.")
            actions.append("propose_molecules")
            rationales.append(
                "Property results from completed landscapes are already available, so the next molecule batch can "
                "be drafted while remaining CSP jobs continue."
            )
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        if active_jobs and "refresh_jobs" in available_tools:
            actions.append("refresh_jobs")
            rationales.append("Active CSP jobs are present, so the highest-priority action is to refresh job state.")
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)
        if active_property_jobs and "refresh_jobs" in available_tools:
            actions.append("refresh_jobs")
            rationales.append("Active property jobs are present, so the planner refreshes ORCA job state.")
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        submittable = [
            molecule
            for molecule in campaign.molecules
            if molecule.status == "proposed" and molecule.molecule_path
        ]
        if len(submittable) >= config.molecule_batch_size and "submit_ready_molecules" in available_tools:
            actions.append("submit_ready_molecules")
            rationales.append(
                f"A full batch of {config.molecule_batch_size} proposed molecules with concrete input files is ready for CSP submission."
            )
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        if 0 < len(submittable) < config.molecule_batch_size and "propose_molecules" in available_tools:
            actions.append("propose_molecules")
            rationales.append(
                f"Only {len(submittable)} valid/materialized molecules are ready; propose more candidates until a full batch of "
                f"{config.molecule_batch_size} is available for CSP."
            )
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        if campaign.status == "analysis_ready" and config.auto_decide and "decide_next_round" in available_tools:
            actions.append("decide_next_round")
            rationales.append("Completed landscapes are ready for review, so the planner records a decision.")
            if auto_expand_ok and "propose_molecules" in available_tools:
                actions.append("propose_molecules")
                rationales.append("Auto-expand is enabled, so the planner immediately drafts the next molecule batch.")
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        completed_without_metrics = [
            job for job in campaign.csp_jobs if job.status == "succeeded" and job.metrics is None
        ]
        if completed_without_metrics and "refresh_jobs" in available_tools:
            actions.append("refresh_jobs")
            rationales.append("Completed jobs are missing parsed metrics, so results must be refreshed and summarized.")
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        if (
            [
                molecule
                for molecule in property_tracking_candidates
                if molecule.status in {"landscape_ready", "selected_for_followup"}
            ]
            and ((not has_property_jobs) or property_missing)
            and "submit_property_jobs" in available_tools
        ):
            actions.append("submit_property_jobs")
            rationales.append(
                "Landscape-ready molecules should proceed to target-aligned property evaluation on the most relevant structural representatives."
            )
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        packing_ready_candidates = [
            molecule
            for molecule in campaign.molecules
            if molecule.status in {"landscape_ready", "property_ready", "selected_for_followup"}
        ]
        if packing_ready_candidates and not campaign.packing_evaluations and "submit_packing_jobs" in available_tools:
            actions.append("submit_packing_jobs")
            rationales.append("Packing-aware pair and cluster calculations should follow successful crystal landscapes.")
            return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

        if auto_expand_ok and not has_proposed and "propose_molecules" in available_tools:
            if "winning patterns" in memory_bundle.get("winning_patterns.md", "").lower() or campaign.decisions:
                actions.append("propose_molecules")
                rationales.append("No draft molecules remain and prior rounds provide enough signal to expand the search.")

        return PlannerActionSelection(actions=actions, notes=notes, action_rationales=rationales)

    @staticmethod
    def _campaign_wants_property_evaluation(campaign: DiscoveryCampaign) -> bool:
        desired = " ".join(campaign.desired_properties).lower()
        target = campaign.target_description.lower()
        combined = f"{desired} {target}"
        property_terms = ("adsorption", "uptake", "capture", "separation", "selectivity", "working_capacity")
        material_terms = ("methane", "ch4", "co2", "carbon dioxide", "h2o", "water", "humid", "humidity")
        return any(term in combined for term in property_terms) and any(term in combined for term in material_terms)

    @staticmethod
    def _needs_property_evaluation(campaign: DiscoveryCampaign, molecule_id: str) -> bool:
        relevant_jobs = [
            job
            for job in campaign.property_jobs
            if job.molecule_id == molecule_id
        ]
        if not relevant_jobs:
            return True
        for job in relevant_jobs:
            if job.status in {"queued", "running"}:
                return False
            if job.status == "succeeded" and HeuristicMainPlannerAgent._job_has_useful_property_metrics(job):
                return False
        return True

    @staticmethod
    def _job_has_useful_property_metrics(job) -> bool:
        metrics = getattr(job, "metrics", None)
        if metrics is None:
            return False
        values = [
            getattr(metrics, "property_score", None),
            getattr(metrics, "methane_uptake_absolute_mmol_g", None),
            getattr(metrics, "methane_uptake_excess_mmol_g", None),
            getattr(metrics, "co2_uptake_absolute_mmol_g", None),
            getattr(metrics, "co2_uptake_excess_mmol_g", None),
            getattr(metrics, "h2o_uptake_absolute_mmol_g", None),
            getattr(metrics, "h2o_uptake_excess_mmol_g", None),
        ]
        return any(value is not None for value in values)

    @staticmethod
    def _property_evaluation_is_skipped(molecule) -> bool:
        return any(str(note).startswith("Skipped adsorption:") for note in getattr(molecule, "notes", []))

    @staticmethod
    def _trailing_note_matches(notes: list[str], marker: str) -> int:
        count = 0
        for note in reversed(notes):
            if marker in str(note):
                count += 1
                continue
            break
        return count


class OpenAIMainPlannerAgent:
    """Optional structured main planner that selects tools from file-backed memory and state."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or resolve_openai_model_env(
            "ORGANIC_CRYSTAL_AGENT_OPENAI_MAIN_PLANNER_MODEL",
            specific_default="gpt-5.4-mini",
        )

    def choose_actions(
        self,
        campaign: DiscoveryCampaign,
        campaign_summary: CampaignSummary,
        memory_bundle: dict[str, str],
        available_tools: list[str],
        config: PlannerConfig,
    ) -> PlannerActionSelection:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI main planner requires `openai` to be installed.") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI main planner requires OPENAI_API_KEY.")

        prompt = (
            "You are the single main planner for an automated organic crystal discovery runtime. "
            "Select only from the allowed tool names. Return strict JSON with keys: "
            "`actions` (ordered list of tool names), `action_rationales` (parallel short explanations), "
            "and `notes` (high-level planner notes).\n\n"
            f"Allowed tools: {available_tools}\n"
            f"Structured campaign summary:\n{campaign_summary.model_dump_json(indent=2)}\n\n"
            f"Application spec:\n{memory_bundle.get('application_spec.md', '')[:3000]}\n\n"
            f"Literature memory:\n{memory_bundle.get('literature.md', '')[:4000]}\n\n"
            f"Open-access local literature:\n{memory_bundle.get('open_access_literature.md', '')[:4000]}\n\n"
            f"Design memory:\n{memory_bundle.get('molecule_design_rules.md', '')[:4000]}\n\n"
            f"Failure patterns:\n{memory_bundle.get('failure_patterns.md', '')[:3000]}\n\n"
            f"Winning patterns:\n{memory_bundle.get('winning_patterns.md', '')[:3000]}\n\n"
            f"Knowledge insights:\n{memory_bundle.get('knowledge_insights.md', '')[:3000]}\n\n"
            f"Search bias:\n{memory_bundle.get('search_bias.md', '')[:2500]}\n"
        )
        client = OpenAI()
        response = client.responses.create(model=self.model, input=prompt)
        text = getattr(response, "output_text", "") or ""
        return self._parse_actions(text, available_tools)

    def _parse_actions(self, text: str, available_tools: list[str]) -> PlannerActionSelection:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        payload = json.loads(stripped)
        actions = [str(item) for item in payload.get("actions", [])]
        filtered_actions = [item for item in actions if item in available_tools]
        rationales = [str(item) for item in payload.get("action_rationales", [])][: len(filtered_actions)]
        notes = [str(item) for item in payload.get("notes", [])]
        return PlannerActionSelection(
            actions=filtered_actions,
            action_rationales=rationales,
            notes=notes,
        )


class MainPlanner:
    """Single high-capability planner coordinating tools through file-backed memory."""

    def __init__(
        self,
        orchestrator: CampaignOrchestrator | None = None,
        config: PlannerConfig | None = None,
        agent: MainPlannerAgent | None = None,
    ) -> None:
        self.orchestrator = orchestrator or CampaignOrchestrator()
        self.config = config or PlannerConfig()
        self.registry: PlannerToolRegistry = build_planner_tool_registry(self.orchestrator)
        self.agent = agent or get_default_main_planner_agent()

    def step(self, campaign: DiscoveryCampaign, campaign_path: str | None = None) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        store = self._memory_store(campaign_path, updated)
        store.write_campaign_memory(updated)
        memory_bundle = store.load_context_bundle()
        campaign_summary = build_campaign_summary(updated)
        planner_mode = self.agent.__class__.__name__
        selection = self.agent.choose_actions(
            campaign=updated,
            campaign_summary=campaign_summary,
            memory_bundle=memory_bundle,
            available_tools=self.registry.names(),
            config=self.config,
        )
        updated.planner_traces.append(
            PlannerTrace(
                round_index=updated.round_index,
                planner_mode=planner_mode,
                available_tools=self.registry.names(),
                chosen_actions=selection.actions,
                action_rationales=selection.action_rationales,
                notes=[
                    "Planner step executed from file-backed memory.",
                    f"Memory root: {store.root}",
                ]
                + selection.notes,
            )
        )
        store.write_campaign_memory(updated)

        for action in selection.actions:
            updated = self.registry.execute(
                action,
                campaign=updated,
                max_hypotheses=self.config.literature_hypotheses,
                max_molecules=self.config.molecule_batch_size,
                sample_count=self.config.sample_count,
                z_value=self.config.z_value,
                space_group=self.config.space_group,
                checkpoint_path=self.config.checkpoint_path,
                use_real_sg_ratio=self.config.use_real_sg_ratio,
            )
            updated = self.orchestrator.materialize_molecule_inputs(
                updated,
                campaign_root=store.root,
            )
            store.write_campaign_memory(updated)

        return updated

    def _memory_store(self, campaign_path: str | None, campaign: DiscoveryCampaign) -> MemoryStore:
        return MemoryStore(resolve_campaign_root(campaign_path, campaign.campaign_id))


def get_default_main_planner_agent() -> MainPlannerAgent:
    mode = os.getenv("ORGANIC_CRYSTAL_AGENT_MAIN_PLANNER", "heuristic").lower()
    if mode == "openai":
        return OpenAIMainPlannerAgent()
    return HeuristicMainPlannerAgent()
