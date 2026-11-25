from __future__ import annotations

import json
import os
from typing import Protocol

from organic_crystal_agent.application_specs import build_application_spec
from organic_crystal_agent.config import resolve_openai_model_env
from organic_crystal_agent.knowledge import SEED_KNOWLEDGE
from organic_crystal_agent.local_literature import local_evidence_cards, resolve_local_literature_hypotheses
from organic_crystal_agent.research import get_default_research_backend, is_openai_backend_available
from organic_crystal_agent.schemas import DiscoveryCampaign, LiteratureHypothesis


class LiteratureAgent(Protocol):
    def generate_hypotheses(
        self,
        campaign: DiscoveryCampaign,
        max_hypotheses: int = 5,
    ) -> list[LiteratureHypothesis]:
        ...


class HeuristicLiteratureAgent:
    """Research-backed hypothesis generator with structured outputs."""

    def generate_hypotheses(
        self,
        campaign: DiscoveryCampaign,
        max_hypotheses: int = 5,
    ) -> list[LiteratureHypothesis]:
        spec = build_application_spec(campaign)
        local_hypotheses = resolve_local_literature_hypotheses(campaign, max_hypotheses=max_hypotheses)
        if len(local_hypotheses) >= max_hypotheses:
            return local_hypotheses[:max_hypotheses]

        backend = get_default_research_backend()
        hints: list[str] = []
        for query in spec.search_queries[: max(1, min(3, max_hypotheses))]:
            hints.extend(
                backend.search(
                    query=query,
                    objective=", ".join(campaign.desired_properties) or campaign.target_description,
                )
            )
        hypotheses: list[LiteratureHypothesis] = []
        hypotheses.extend(local_hypotheses)

        remaining_slots = max(0, max_hypotheses - len(hypotheses))
        for idx, hint in enumerate(hints[:remaining_slots], start=len(hypotheses) + 1):
            hypotheses.append(
                LiteratureHypothesis(
                    hypothesis_id=f"lit-auto-{idx:03d}",
                    title=f"Research hypothesis {idx}",
                    summary=hint,
                    evidence=[hint],
                    design_implications=self._design_implications(hint, campaign),
                    risk_flags=self._risk_flags(hint, campaign),
                    confidence=0.68,
                )
            )

        if len(hypotheses) < max_hypotheses:
            hypotheses.append(
                LiteratureHypothesis(
                    hypothesis_id=f"lit-spec-001",
                    title=f"{spec.application_mode} design spec",
                    summary=(
                        f"Application mode `{spec.application_mode}` requires: "
                        + "; ".join(spec.hard_constraints[:3])
                    ),
                    evidence=spec.hard_constraints[:3],
                    design_implications=spec.soft_preferences[:4],
                    risk_flags=self._risk_flags(" ".join(spec.red_flags), campaign),
                    confidence=0.72 if spec.evidence_source == "built_in_prior" else 0.55,
                )
            )

        remaining = max_hypotheses - len(hypotheses)
        if remaining > 0:
            lowered = campaign.target_description.lower()
            knowledge_rows = list(SEED_KNOWLEDGE)
            if "porous" in lowered:
                knowledge_rows.sort(key=lambda row: 0 if "porous" in row["topic"] else 1)
            for row in knowledge_rows[:remaining]:
                hypotheses.append(
                    LiteratureHypothesis(
                        hypothesis_id=f"lit-seed-{len(hypotheses) + 1:03d}",
                        title=row["topic"],
                        summary=row["insight"],
                        evidence=[row["insight"]],
                        design_implications=self._design_implications(row["insight"], campaign),
                        risk_flags=self._risk_flags(row["insight"], campaign),
                        confidence=0.58,
                    )
                )

        return hypotheses[:max_hypotheses]

    def _design_implications(self, text: str, campaign: DiscoveryCampaign) -> list[str]:
        spec = build_application_spec(campaign)
        lowered = f"{text} {campaign.target_description} {' '.join(campaign.desired_properties)}".lower()
        implications: list[str] = []
        implications.extend(spec.soft_preferences[:2])
        if "porous" in lowered or "low-density" in lowered or "low_density" in lowered:
            implications.append("Bias molecule proposals toward contorted, shape-persistent, weakly interdigitating scaffolds.")
        if "solvent" in lowered or "solvate" in lowered:
            implications.append("Track solvent-trapping risk explicitly when ranking candidates and validating low-density forms.")
        if spec.application_mode == "charge_transport" and ("transport" in lowered or "stack" in lowered):
            implications.append("Favor planar pi-surfaces and anisotropic packing hypotheses for transport-oriented screens.")
        elif "stack" in lowered and spec.application_mode != "charge_transport":
            implications.append("Treat pi-stacking only as one packing motif among many; do not convert it into a transport-screening objective.")
        if "hydrogen-bond" in lowered or "hydrogen bond" in lowered:
            implications.append("Include directional donor/acceptor motifs that can stabilize open packing without immediate collapse.")
        if not implications:
            implications.append("Use this hint to refine scaffold family selection rather than directly overfitting one structure.")
        return implications

    def _risk_flags(self, text: str, campaign: DiscoveryCampaign) -> list[str]:
        spec = build_application_spec(campaign)
        lowered = f"{text} {campaign.target_description} {' '.join(spec.red_flags)}".lower()
        flags: list[str] = []
        if "solvate" in lowered or "solvent" in lowered:
            flags.append("solvate_trap")
        if "collapse" in lowered:
            flags.append("collapse_on_activation")
        if "polymorph" in lowered or "dense" in lowered:
            flags.append("competing_polymorph")
        if "poor crystallinity" in lowered:
            flags.append("poor_crystallinity")
        return flags


class OpenAILiteratureAgent:
    """Optional structured literature reasoner using OpenAI Responses + web search."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or resolve_openai_model_env(
            "ORGANIC_CRYSTAL_AGENT_OPENAI_LITERATURE_MODEL",
            specific_default="gpt-5.5",
        )

    def generate_hypotheses(
        self,
        campaign: DiscoveryCampaign,
        max_hypotheses: int = 5,
    ) -> list[LiteratureHypothesis]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI literature agent requires `openai` to be installed.") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI literature agent requires OPENAI_API_KEY.")

        local_cards = local_evidence_cards(campaign)[:8]
        local_card_text = "\n".join(
            f"- {card.title}: {card.summary} | implications: {', '.join(card.design_implications[:3])}"
            for card in local_cards
        )
        prompt = (
            "You are a literature specialist for automated organic crystal discovery. "
            "Use web search and return strict JSON with key `hypotheses`, a list of objects. "
            "Each object must contain: title, summary, evidence (list of short strings), "
            "design_implications (list), risk_flags (list), confidence (0-1).\n\n"
            f"Target: {campaign.target_description}\n"
            f"Desired properties: {', '.join(campaign.desired_properties)}\n"
            f"Max hypotheses: {max_hypotheses}\n"
            "The local corpus already contains a humid-CO2 macrocycle paper, its supplementary information, "
            "and several HOF reviews. Treat the macrocycle paper as evidence about shape-selective aromatic pi-surfaces "
            "and weak multi-contact CO2 binding, not only nitrile/fluoro head-group preference.\n"
            "Use the review perspective to surface whole-tecton and framework-design implications, including HOF assembly, "
            "shape persistence, anti-collapse behavior, contorted aromatic faces, and gas-separation-relevant packing motifs.\n"
            f"Local literature cards:\n{local_card_text or '- None'}"
        )
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        text = getattr(response, "output_text", "") or ""
        return self._parse(text, max_hypotheses=max_hypotheses)

    def _parse(self, text: str, max_hypotheses: int) -> list[LiteratureHypothesis]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        payload = json.loads(stripped)
        items = payload.get("hypotheses", [])
        hypotheses: list[LiteratureHypothesis] = []
        for idx, item in enumerate(items[:max_hypotheses], start=1):
            hypotheses.append(
                LiteratureHypothesis(
                    hypothesis_id=f"lit-openai-{idx:03d}",
                    title=str(item.get("title", f"Hypothesis {idx}")),
                    summary=str(item.get("summary", "")),
                    evidence=[str(x) for x in item.get("evidence", []) if str(x).strip()],
                    design_implications=[str(x) for x in item.get("design_implications", []) if str(x).strip()],
                    risk_flags=[str(x) for x in item.get("risk_flags", []) if str(x).strip()],
                    confidence=float(item.get("confidence", 0.6)),
                )
            )
        return hypotheses


def get_default_literature_agent() -> LiteratureAgent:
    mode = os.getenv("ORGANIC_CRYSTAL_AGENT_LITERATURE_AGENT", "auto").lower()
    if mode == "openai":
        return OpenAILiteratureAgent()
    if mode == "auto" and is_openai_backend_available():
        return OpenAILiteratureAgent()
    return HeuristicLiteratureAgent()
