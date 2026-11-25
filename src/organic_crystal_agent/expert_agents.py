from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from organic_crystal_agent.application_specs import build_application_spec
from organic_crystal_agent.config import resolve_openai_model_env
from organic_crystal_agent.design_critic import critique_molecule_design
from organic_crystal_agent.memory_store import MemoryStore, resolve_campaign_root
from organic_crystal_agent.molecule_quality import assess_molecule_quality
from organic_crystal_agent.naming import normalize_molecule_label
from organic_crystal_agent.orchestrator import CampaignOrchestrator
from organic_crystal_agent.planning_agent import (
    _record_core_arm_symmetry_preference,
    build_openai_molecule_design_prompt,
)
from organic_crystal_agent.rdkit_utils import canonicalize_and_validate_registration_smiles
from organic_crystal_agent.schemas import DiscoveryCampaign, MoleculeRecord, PlannerTrace


class ExpertActionReport(BaseModel):
    agent_name: str
    action: str
    status: str = "succeeded"
    details: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExpertStepReport(BaseModel):
    campaign_id: str
    round_index: int
    status_before: str
    status_after: str
    actions: list[ExpertActionReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass
class ExpertSupervisorConfig:
    molecule_batch_size: int = 5
    max_total_molecules: int = 100
    sample_count: int = 1000
    z_value: int | None = None
    space_group: int | None = None
    checkpoint_path: str = ""
    use_real_sg_ratio: bool = False
    auto_decide: bool = True
    auto_expand: bool = True


class LiteratureExpert:
    name = "LiteratureExpert"

    def __init__(self, orchestrator: CampaignOrchestrator) -> None:
        self.orchestrator = orchestrator

    def ensure_hypotheses(self, campaign: DiscoveryCampaign, *, max_hypotheses: int = 5) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        before = len(campaign.literature_hypotheses)
        updated = self.orchestrator.auto_add_literature_hypotheses(campaign, max_hypotheses=max_hypotheses)
        after = len(updated.literature_hypotheses)
        return updated, ExpertActionReport(
            agent_name=self.name,
            action="ensure_literature_hypotheses",
            details=[
                f"literature_hypotheses: {before} -> {after}",
                f"knowledge_inputs=csp_succeeded:{_count_succeeded_csp_jobs(updated)} property_succeeded:{_count_succeeded_property_jobs(updated)}",
            ],
        )


class MoleculeProposalExpert:
    name = "MoleculeProposalExpert"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or resolve_openai_model_env(
            "ORGANIC_CRYSTAL_AGENT_OPENAI_MOLECULE_MODEL",
            specific_default="gpt-5.5",
        )

    def propose(
        self,
        campaign: DiscoveryCampaign,
        *,
        max_molecules: int,
        campaign_path: str | Path | None = None,
    ) -> tuple[list[MoleculeRecord], ExpertActionReport]:
        report = ExpertActionReport(agent_name=self.name, action="propose_molecules")
        if not os.getenv("OPENAI_API_KEY"):
            report.status = "failed"
            report.errors.append("OPENAI_API_KEY is not set.")
            return [], report

        try:
            from openai import OpenAI
        except ImportError as exc:
            report.status = "failed"
            report.errors.append(f"OpenAI package is unavailable: {exc}")
            return [], report

        client = OpenAI()
        prompt = build_openai_molecule_design_prompt(campaign, max_molecules=max(max_molecules * 4, max_molecules + 5))
        if campaign_path is not None:
            store = MemoryStore(resolve_campaign_root(str(campaign_path), campaign.campaign_id))
            bundle = store.load_context_bundle()
            prompt += (
                "\n\nDistilled campaign knowledge:\n"
                + bundle.get("knowledge_insights.md", "")[:3500]
                + "\n\nNext-round search bias rules:\n"
                + bundle.get("search_bias.md", "")[:2500]
            )
        feedback = _recent_critic_feedback(campaign)
        if feedback:
            prompt += (
                "\n\nRecent MoleculeCriticExpert feedback from this campaign:\n"
                + "\n".join(f"- {item}" for item in feedback)
                + "\nRecovery instruction: do not repeat clearly invalid or self-defeating patterns, but also do not let prior "
                "rejections collapse the search into only the safest scaffold families. In the next batch, keep most candidates "
                "modular and synthetically manageable, while still reserving meaningful room for one or two more ambitious, "
                "literature-forward HOF tectons if they are chemically coherent and distinct.\n"
            )
        if _consecutive_zero_acceptance_rounds(campaign) >= 3:
            prompt += (
                "\n\nBALANCED FEASIBILITY RECOVERY MODE is active because several proposal rounds accepted zero molecules. "
                "For this round, the first priority is to recover a usable batch of RDKit-valid, campaign-relevant molecules, "
                "but do not over-correct into only trivial safe motifs. Keep most candidates modular and synthetically "
                "manageable, while explicitly preserving at least one or two higher-upside literature-forward branches such as "
                "dibenzofuran-like, triptycene-like, quinoxaline/phenazine-like fused aza-aromatics, or other rigid cavity-forming "
                "motifs when they remain chemically coherent. Use the humid-CO2 XLSX fragment seeds as substituent/head-group "
                "guidance, but keep the whole tecton modular.\n"
            )
        try:
            response = self._create_response(client, prompt)
            text = getattr(response, "output_text", "") or ""
            payload = _loads_jsonish(text)
            molecules = _molecules_from_payload(payload, max_molecules=max_molecules * 4)
        except Exception as exc:
            report.status = "failed"
            report.errors.append(f"Proposal output could not be parsed: {type(exc).__name__}: {exc}")
            return [], report

        report.details.append(f"drafted_candidates={len(molecules)}")
        return molecules, report

    def _create_response(self, client: Any, prompt: str) -> Any:
        try:
            return client.responses.create(
                model=self.model,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
        except TypeError:
            return client.responses.create(model=self.model, input=prompt)


class MoleculeCriticExpert:
    name = "MoleculeCriticExpert"

    def review(
        self,
        campaign: DiscoveryCampaign,
        drafted: list[MoleculeRecord],
        *,
        max_molecules: int,
    ) -> tuple[list[MoleculeRecord], ExpertActionReport]:
        spec = build_application_spec(campaign)
        report = ExpertActionReport(agent_name=self.name, action="review_molecules")
        existing_labels = {molecule.molecule_label for molecule in campaign.molecules if molecule.molecule_label}
        existing_smiles = {molecule.smiles for molecule in campaign.molecules if molecule.smiles}
        accepted: list[MoleculeRecord] = []

        for molecule in drafted:
            label = molecule.molecule_label.strip()
            if not label or label in existing_labels:
                report.details.append(f"rejected duplicate/empty label: {label or '<empty>'}")
                continue
            if not molecule.smiles:
                report.details.append(f"rejected missing SMILES: {label}")
                continue
            try:
                molecule.smiles = canonicalize_and_validate_registration_smiles(molecule.smiles)
            except ValueError as exc:
                report.details.append(f"rejected invalid SMILES {label}: {exc}")
                continue
            if molecule.smiles in existing_smiles or "." in molecule.smiles:
                report.details.append(f"rejected duplicate or disconnected SMILES: {label}")
                continue

            quality = assess_molecule_quality(molecule, spec)
            molecule.notes.extend(
                [f"Quality score: {quality.quality_score}", f"Quality pass: {quality.passes_hard_filters}"]
                + [f"Quality strength: {item}" for item in quality.strengths]
                + [f"Quality concern: {item}" for item in quality.concerns]
                + [f"Quality tag: {item}" for item in quality.derived_tags]
            )
            if not quality.passes_hard_filters:
                report.details.append(f"rejected quality gate: {label}")
                continue

            _record_core_arm_symmetry_preference(molecule, spec.application_mode)

            critique = critique_molecule_design(molecule, spec, quality)
            molecule.notes.extend(
                [f"Critic reject: {critique.should_reject}"]
                + [
                    "Critic synthetic complexity: "
                    f"{critique.synthesis_complexity_label} (score={critique.synthesis_complexity_score})"
                ]
                + [f"Critic synthetic reason: {item}" for item in critique.synthesis_complexity_reasons]
                + [f"Critic positive: {item}" for item in critique.positive_signals]
                + [f"Critic concern: {item}" for item in critique.concerns]
                + [f"Critic red flag: {item}" for item in critique.red_flag_hits]
                + [f"Critic revision: {item}" for item in critique.suggested_revisions]
            )
            if critique.should_reject:
                report.details.append(
                    "rejected critic gate: "
                    f"{label} | smiles={molecule.smiles} | synthetic_complexity="
                    f"{critique.synthesis_complexity_label}:{critique.synthesis_complexity_score}"
                )
                continue

            accepted.append(molecule)
            existing_labels.add(label)
            existing_smiles.add(molecule.smiles)
            if len(accepted) >= max_molecules:
                break

        report.details.append(f"accepted_candidates={len(accepted)}")
        return accepted, report


class CSPLandscapeExpert:
    name = "CSPLandscapeExpert"

    def __init__(self, orchestrator: CampaignOrchestrator) -> None:
        self.orchestrator = orchestrator

    def refresh(self, campaign: DiscoveryCampaign) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        updated = self.orchestrator.refresh_campaign(campaign)
        return updated, ExpertActionReport(
            agent_name=self.name,
            action="refresh_csp_and_property_jobs",
            details=[f"campaign_status={updated.status}"],
        )

    def submit_ready(
        self,
        campaign: DiscoveryCampaign,
        *,
        max_molecules: int,
        sample_count: int,
        z_value: int | None,
        space_group: int | None,
        checkpoint_path: str,
        use_real_sg_ratio: bool,
    ) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        updated = campaign.model_copy(deep=True)
        submitted: list[str] = []
        submittable = [
            molecule
            for molecule in updated.molecules
            if molecule.status == "proposed" and molecule.molecule_path
        ]
        for molecule in sorted(submittable, key=lambda item: item.molecule_id)[:max_molecules]:
            updated = self.orchestrator.submit_molecule_for_csp(
                campaign=updated,
                molecule_id=molecule.molecule_id,
                sample_count=sample_count,
                z_value=z_value,
                space_group=space_group,
                checkpoint_path=checkpoint_path,
                use_real_sg_ratio=use_real_sg_ratio,
            )
            submitted.append(molecule.molecule_id)
        return updated, ExpertActionReport(
            agent_name=self.name,
            action="submit_ready_molecules_to_csp",
            details=[f"submitted={submitted}"],
        )


class PropertyExpert:
    name = "PropertyExpert"

    def __init__(self, orchestrator: CampaignOrchestrator) -> None:
        self.orchestrator = orchestrator

    def submit_missing(self, campaign: DiscoveryCampaign) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        before = len(campaign.property_jobs)
        updated = self.orchestrator.auto_submit_property_jobs(campaign)
        after = len(updated.property_jobs)
        profiles = sorted({job.property_profile for job in updated.property_jobs[before:]})
        return updated, ExpertActionReport(
            agent_name=self.name,
            action="submit_target_property_jobs",
            details=[f"property_jobs: {before} -> {after}", f"new_profiles={profiles}"],
        )


class KnowledgeExpert:
    name = "KnowledgeExpert"

    def __init__(self, orchestrator: CampaignOrchestrator) -> None:
        self.orchestrator = orchestrator

    def distill(
        self,
        campaign: DiscoveryCampaign,
        *,
        campaign_path: str | Path | None,
        auto_decide: bool = True,
    ) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        updated = campaign.model_copy(deep=True)
        root = resolve_campaign_root(str(campaign_path), updated.campaign_id) if campaign_path is not None else resolve_campaign_root(None, updated.campaign_id)
        store = MemoryStore(root)
        store.write_campaign_memory(updated)
        bundle = store.load_context_bundle()
        insight_count = sum(
            1
            for line in bundle.get("knowledge_insights.md", "").splitlines()
            if line.strip().startswith("- ")
        )
        bias_count = sum(
            1
            for line in bundle.get("search_bias.md", "").splitlines()
            if line.strip().startswith("- ")
        )
        details = [
            f"knowledge_root={root}",
            f"insight_bullets={insight_count}",
            f"search_bias_rules={bias_count}",
            f"knowledge_inputs=csp_succeeded:{_count_succeeded_csp_jobs(updated)} property_succeeded:{_count_succeeded_property_jobs(updated)}",
        ]
        if updated.status == "analysis_ready" and auto_decide:
            before = len(updated.decisions)
            updated = self.orchestrator.record_decision(
                campaign=updated,
                selected_molecule_ids=[],
                rejected_molecule_ids=[],
                next_actions=self._next_actions_from_bundle(bundle),
                rationale=self._rationale_from_bundle(bundle),
            )
            details.append(f"decisions: {before} -> {len(updated.decisions)}")
        return updated, ExpertActionReport(
            agent_name=self.name,
            action="distill_campaign_knowledge",
            details=details,
        )

    @staticmethod
    def _rationale_from_bundle(bundle: dict[str, str], *, limit: int = 6) -> list[str]:
        lines: list[str] = []
        for source_name in ("knowledge_insights.md", "search_bias.md", "winning_patterns.md", "failure_patterns.md"):
            for raw in bundle.get(source_name, "").splitlines():
                stripped = raw.strip()
                if stripped.startswith("- "):
                    lines.append(stripped[2:].strip())
                if len(lines) >= limit:
                    return lines
        return lines or ["Completed results were distilled into campaign knowledge for the next proposal round."]

    @staticmethod
    def _next_actions_from_bundle(bundle: dict[str, str]) -> list[str]:
        biases = [
            raw.strip()[2:].strip()
            for raw in bundle.get("search_bias.md", "").splitlines()
            if raw.strip().startswith("- ")
        ]
        actions = ["Generate the next molecule batch using the updated search bias and distilled campaign knowledge."]
        if biases:
            actions.extend([f"Search bias: {item}" for item in biases[:3]])
        return actions


class ExpertSupervisor:
    """Crash-isolated campaign supervisor composed from specialist agents."""

    def __init__(
        self,
        *,
        orchestrator: CampaignOrchestrator | None = None,
        config: ExpertSupervisorConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator or CampaignOrchestrator()
        self.config = config or ExpertSupervisorConfig()
        self.literature = LiteratureExpert(self.orchestrator)
        self.proposer = MoleculeProposalExpert()
        self.critic = MoleculeCriticExpert()
        self.csp = CSPLandscapeExpert(self.orchestrator)
        self.property = PropertyExpert(self.orchestrator)
        self.knowledge = KnowledgeExpert(self.orchestrator)

    def step(self, campaign: DiscoveryCampaign, *, campaign_path: str | Path | None = None) -> tuple[DiscoveryCampaign, ExpertStepReport]:
        status_before = campaign.status
        updated = campaign.model_copy(deep=True)
        report = ExpertStepReport(
            campaign_id=updated.campaign_id,
            round_index=updated.round_index,
            status_before=status_before,
            status_after=updated.status,
        )

        updated, action = self._run(updated, "refresh", lambda: self.csp.refresh(updated))
        report.actions.append(action)

        if _needs_literature_refresh(updated):
            updated, action = self._run(updated, "literature", lambda: self.literature.ensure_hypotheses(updated))
            report.actions.append(action)

        if _needs_knowledge_refresh(updated):
            updated, action = self._run(
                updated,
                "knowledge",
                lambda: self.knowledge.distill(
                    updated,
                    campaign_path=campaign_path,
                    auto_decide=self.config.auto_decide,
                ),
            )
            report.actions.append(action)

        if _has_property_candidates(updated):
            updated, action = self._run(updated, "property", lambda: self.property.submit_missing(updated))
            report.actions.append(action)

        molecule_cap_reached = len(updated.molecules) >= self.config.max_total_molecules
        if molecule_cap_reached:
            report.notes.append(
                f"Molecule cap reached ({len(updated.molecules)}/{self.config.max_total_molecules}); supervisor will not propose additional molecules."
            )

        inflight_count = _count_inflight_molecules(updated)
        missing_slots = max(0, self.config.molecule_batch_size - inflight_count)
        if missing_slots > 0 and self.config.auto_expand and not molecule_cap_reached:
            updated, proposal_reports = self._proposal_pipeline(
                updated,
                campaign_path=campaign_path,
                requested_count=missing_slots,
            )
            report.actions.extend(proposal_reports)
            inflight_count = _count_inflight_molecules(updated)

        submittable = [molecule for molecule in updated.molecules if molecule.status == "proposed" and molecule.molecule_path]
        if submittable:
            updated, action = self._run(
                updated,
                "csp_submit",
                lambda: self.csp.submit_ready(
                    updated,
                    max_molecules=len(submittable),
                    sample_count=self.config.sample_count,
                    z_value=self.config.z_value,
                    space_group=self.config.space_group,
                    checkpoint_path=self.config.checkpoint_path,
                    use_real_sg_ratio=self.config.use_real_sg_ratio,
                ),
            )
            report.actions.append(action)
            return self._finalize(updated, report, campaign_path)

        if _has_active_jobs(updated):
            report.notes.append(
                "Active Slurm-backed jobs remain; supervisor refreshes, distills, submits ready property jobs, and refills the rolling molecule queue toward the target batch size."
            )
            return self._finalize(updated, report, campaign_path)

        if molecule_cap_reached:
            updated.status = "completed"
            report.notes.append("Campaign completed because the configured molecule cap has been reached and no active jobs remain.")
            return self._finalize(updated, report, campaign_path)

        report.notes.append("No specialist action was needed beyond refresh.")
        return self._finalize(updated, report, campaign_path)

    def _proposal_pipeline(
        self,
        campaign: DiscoveryCampaign,
        *,
        campaign_path: str | Path | None,
        requested_count: int,
    ) -> tuple[DiscoveryCampaign, list[ExpertActionReport]]:
        reports: list[ExpertActionReport] = []
        drafted, proposal_report = self.proposer.propose(
            campaign,
            max_molecules=requested_count,
            campaign_path=campaign_path,
        )
        reports.append(proposal_report)
        if proposal_report.status != "succeeded":
            updated = campaign.model_copy(deep=True)
            _append_unique_note(updated, f"Expert proposer failed without stopping supervisor: {'; '.join(proposal_report.errors)}")
            return updated, reports

        accepted, critique_report = self.critic.review(campaign, drafted, max_molecules=requested_count)
        reports.append(critique_report)
        updated = campaign.model_copy(deep=True)
        if not accepted:
            _append_unique_note(updated, "Expert critic accepted no new molecules; supervisor will retry on a later step.")
            return updated, reports

        for item in accepted:
            updated = self.orchestrator.add_molecule(
                campaign=updated,
                molecule_label=item.molecule_label,
                rationale=item.rationale,
                molecule_path=item.molecule_path,
                smiles=item.smiles,
                source_agent="expert-proposer+critic",
                design_family=item.design_family,
                evidence=item.evidence,
                target_properties=item.target_properties,
                notes=item.notes,
            )
        if campaign_path is not None:
            updated = self.orchestrator.materialize_molecule_inputs(
                updated,
                campaign_root=resolve_campaign_root(str(campaign_path), updated.campaign_id),
            )
        reports.append(
            ExpertActionReport(
                agent_name="Supervisor",
                action="register_reviewed_molecules",
                details=[f"registered={len(accepted)}"],
            )
        )
        return updated, reports

    def _run(self, current: DiscoveryCampaign, action_name: str, fn: Any) -> tuple[DiscoveryCampaign, ExpertActionReport]:
        try:
            return fn()
        except Exception as exc:
            report = ExpertActionReport(
                agent_name="Supervisor",
                action=action_name,
                status="failed",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            _append_unique_note(current, f"Expert action `{action_name}` failed without stopping supervisor: {report.errors[0]}")
            return current, report

    def _finalize(
        self,
        campaign: DiscoveryCampaign,
        report: ExpertStepReport,
        campaign_path: str | Path | None,
    ) -> tuple[DiscoveryCampaign, ExpertStepReport]:
        report.status_after = campaign.status
        campaign.planner_traces.append(
            PlannerTrace(
                round_index=campaign.round_index,
                planner_mode="ExpertSupervisor",
                available_tools=[
                    "LiteratureExpert",
                    "MoleculeProposalExpert",
                    "MoleculeCriticExpert",
                    "CSPLandscapeExpert",
                    "PropertyExpert",
                    "KnowledgeExpert",
                ],
                chosen_actions=[f"{item.agent_name}:{item.action}" for item in report.actions],
                action_rationales=[
                    detail
                    for item in report.actions
                    for detail in (item.details or item.errors)
                ],
                notes=report.notes + [f"expert_status_after={report.status_after}"],
            )
        )
        if campaign_path is not None:
            store = MemoryStore(resolve_campaign_root(str(campaign_path), campaign.campaign_id))
            store.write_campaign_memory(campaign)
        return campaign, report


def _loads_jsonish(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not start_candidates:
        raise ValueError("No JSON object or array found in model output.")
    start = min(start_candidates)
    candidate = stripped[start:]
    end = _balanced_json_end(candidate)
    if end is not None:
        candidate = candidate[:end]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


def _recent_critic_feedback(campaign: DiscoveryCampaign, *, limit: int = 18) -> list[str]:
    feedback: list[str] = []
    for trace in reversed(campaign.planner_traces):
        if "MoleculeCriticExpert:review_molecules" not in trace.chosen_actions:
            continue
        for item in trace.action_rationales:
            if item.startswith("rejected retrosynthesis gate:"):
                continue
            if item.startswith("rejected critic gate:") and "synthetic_complexity=" in item:
                continue
            if item.startswith("accepted_candidates=0"):
                continue
            if item.startswith("rejected "):
                feedback.append(item)
            if len(feedback) >= limit:
                return feedback
    return feedback


def _consecutive_zero_acceptance_rounds(campaign: DiscoveryCampaign) -> int:
    count = 0
    for trace in reversed(campaign.planner_traces):
        if "MoleculeCriticExpert:review_molecules" not in trace.chosen_actions:
            continue
        if any(item == "accepted_candidates=0" for item in trace.action_rationales):
            count += 1
            continue
        break
    return count


def _balanced_json_end(text: str) -> int | None:
    stack: list[str] = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index + 1
    return None


def _molecules_from_payload(payload: Any, *, max_molecules: int) -> list[MoleculeRecord]:
    if isinstance(payload, dict):
        items = payload.get("molecules") or payload.get("candidates") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    molecules: list[MoleculeRecord] = []
    for idx, item in enumerate(items[:max_molecules], start=1):
        if not isinstance(item, dict):
            continue
        molecules.append(
            MoleculeRecord(
                molecule_id=f"draft-{idx:03d}",
                molecule_label=normalize_molecule_label(
                    str(item.get("molecule_label", f"expert-candidate-{idx}")).strip()
                ),
                smiles=(str(item.get("smiles", "")).strip() or None),
                rationale=str(item.get("rationale", "")).strip(),
                design_family=str(item.get("design_family", "")).strip(),
                evidence=[str(value) for value in item.get("evidence", []) if str(value).strip()],
                target_properties=[str(value) for value in item.get("target_properties", []) if str(value).strip()],
                notes=[str(value) for value in item.get("notes", []) if str(value).strip()],
            )
        )
    return molecules


def _has_active_jobs(campaign: DiscoveryCampaign) -> bool:
    return any(job.status in {"queued", "running"} for job in campaign.csp_jobs + campaign.property_jobs)


def _has_knowledge_signal(campaign: DiscoveryCampaign) -> bool:
    if campaign.decisions or campaign.property_jobs:
        return True
    return any(job.metrics is not None for job in campaign.csp_jobs)


def _count_succeeded_csp_jobs(campaign: DiscoveryCampaign) -> int:
    return sum(1 for job in campaign.csp_jobs if job.status == "succeeded" and job.metrics is not None)


def _count_succeeded_property_jobs(campaign: DiscoveryCampaign) -> int:
    return sum(1 for job in campaign.property_jobs if job.status == "succeeded" and job.metrics is not None)


def _latest_trace_with_action(campaign: DiscoveryCampaign, action_name: str) -> PlannerTrace | None:
    target = f":{action_name}"
    for trace in reversed(campaign.planner_traces):
        if any(target in action for action in trace.chosen_actions):
            return trace
    return None


def _extract_knowledge_input_counts(trace: PlannerTrace | None) -> tuple[int, int] | None:
    if trace is None:
        return None
    for item in trace.action_rationales:
        if not item.startswith("knowledge_inputs="):
            continue
        match = re.search(r"csp_succeeded:(\d+)\s+property_succeeded:(\d+)", item)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _needs_knowledge_refresh(campaign: DiscoveryCampaign) -> bool:
    if not _has_knowledge_signal(campaign):
        return False
    current = (_count_succeeded_csp_jobs(campaign), _count_succeeded_property_jobs(campaign))
    previous = _extract_knowledge_input_counts(_latest_trace_with_action(campaign, "distill_campaign_knowledge"))
    return previous is None or current > previous


def _needs_literature_refresh(campaign: DiscoveryCampaign) -> bool:
    if not campaign.literature_hypotheses:
        return True
    if not _has_knowledge_signal(campaign):
        return False
    current = (_count_succeeded_csp_jobs(campaign), _count_succeeded_property_jobs(campaign))
    previous = _extract_knowledge_input_counts(_latest_trace_with_action(campaign, "ensure_literature_hypotheses"))
    return previous is None or current > previous


def _has_property_candidates(campaign: DiscoveryCampaign) -> bool:
    profile = _target_property_profile(campaign)
    submitted = {
        (job.molecule_id, job.property_profile)
        for job in campaign.property_jobs
        if job.status in {"queued", "running", "succeeded"}
    }
    for molecule in campaign.molecules:
        if molecule.status not in {"landscape_ready", "selected_for_followup", "property_ready"}:
            continue
        if any(str(note).startswith("Skipped adsorption:") for note in molecule.notes):
            continue
        if (molecule.molecule_id, profile) not in submitted:
            return True
    return False


def _count_inflight_molecules(campaign: DiscoveryCampaign) -> int:
    return sum(1 for molecule in campaign.molecules if not _is_molecule_work_complete(campaign, molecule))


def _is_molecule_work_complete(campaign: DiscoveryCampaign, molecule: MoleculeRecord) -> bool:
    if molecule.status in {"rejected", "selected_for_followup"}:
        return True
    if _target_property_profile(campaign) in {"humid_co2_capture", "methane_adsorption"}:
        if molecule.status == "property_ready":
            return True
        if any(str(note).startswith("Skipped adsorption:") for note in molecule.notes):
            return True
        return False
    return molecule.status in {"landscape_ready", "property_ready"}


def _target_property_profile(campaign: DiscoveryCampaign) -> str:
    combined = f"{campaign.target_description} {' '.join(campaign.desired_properties)}".lower()
    if any(term in combined for term in ("humid", "co2", "carbon dioxide", "h2o", "water")):
        return "humid_co2_capture"
    if any(term in combined for term in ("methane", "ch4")):
        return "methane_adsorption"
    return "frontier"


def _append_unique_note(campaign: DiscoveryCampaign, note: str) -> None:
    stamped = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {note}"
    if stamped not in campaign.notes:
        campaign.notes.append(stamped)
