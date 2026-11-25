from __future__ import annotations

import json
import os
from typing import Protocol

from organic_crystal_agent.config import resolve_openai_model_env


class ResearchBackend(Protocol):
    """Abstract research service for web/research-backed context gathering."""

    def search(self, query: str, objective: str) -> list[str]:
        ...


class MockOpenAIResearchBackend:
    """Deterministic placeholder for an external research-capable model backend."""

    def search(self, query: str, objective: str) -> list[str]:
        hints = [
            "Prefer experiment-by-experiment reasoning over large initial batches when uncertainty is high.",
            "Treat solvent choice as a control variable for both polymorph selection and solvate trapping.",
            "Early PXRD plus thermal analysis is usually more informative than repeating nearly identical screens.",
        ]
        lowered = f"{query} {objective}".lower()
        if "transport" in lowered or "semiconductor" in lowered:
            hints.append(
                "Planar aromatic cores with slipped pi-stacking are reasonable first-pass hypotheses for transport-oriented campaigns."
            )
        if "porous" in lowered or "hof" in lowered:
            hints.append(
                "Low-density porous forms should be treated as high-information benchmarks, not default first synthesis targets."
            )
        return hints


class OpenAIResponsesResearchBackend:
    """OpenAI Responses API backend using the built-in web search tool.

    Molecule design can use a larger model, while research utility calls default
    to a cheaper mini model unless overridden.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or resolve_openai_model_env(
            "ORGANIC_CRYSTAL_AGENT_OPENAI_RESEARCH_MODEL",
            specific_default="gpt-5.4-mini",
        )

    def search(self, query: str, objective: str) -> list[str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI research backend selected but the openai package is not installed. "
                "Install with `pip install -e .[openai]`."
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OpenAI research backend selected but OPENAI_API_KEY is not set."
            )

        client = OpenAI()
        prompt = (
            "You are assisting an organic crystal experimental campaign. "
            "Use web search to gather only the most useful practical experimental hints. "
            "Return strict JSON with one key: `hints`, whose value is a list of 3 to 6 short strings. "
            "Focus on crystallization strategy, polymorph risk, characterization, and failure interpretation.\n\n"
            f"Objective: {objective}\n"
            f"Request: {query}"
        )

        response = client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        text = getattr(response, "output_text", "") or ""
        return _parse_hints_from_output(text)


def get_default_research_backend() -> ResearchBackend:
    preference = os.getenv("ORGANIC_CRYSTAL_AGENT_RESEARCH_BACKEND", "auto").lower()

    if preference == "mock":
        return MockOpenAIResearchBackend()

    if preference == "openai":
        if is_openai_backend_available():
            return OpenAIResponsesResearchBackend()
        return MockOpenAIResearchBackend()

    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIResponsesResearchBackend()
        except RuntimeError:
            return MockOpenAIResearchBackend()

    return MockOpenAIResearchBackend()


def is_openai_backend_available() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_hints_from_output(text: str) -> list[str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        payload = json.loads(stripped)
        hints = payload.get("hints", [])
        if isinstance(hints, list):
            cleaned = [str(item).strip() for item in hints if str(item).strip()]
            if cleaned:
                return cleaned
    except json.JSONDecodeError:
        pass

    fallback = [line.strip("- ").strip() for line in stripped.splitlines() if line.strip()]
    return fallback[:6] if fallback else [
        "Gather crystallization and verification hints before committing to a broad batch."
    ]
