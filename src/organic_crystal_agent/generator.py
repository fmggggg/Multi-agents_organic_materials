from __future__ import annotations

from typing import Protocol

from organic_crystal_agent.schemas import CampaignMemory, CrystalFormHypothesis


class CrystalFormGenerator(Protocol):
    """Interface for producing candidate crystal-form hypotheses."""

    def generate(
        self,
        request: str,
        objective: str,
        campaign_memory: CampaignMemory,
        research_hints: list[str],
        max_candidates: int = 3,
    ) -> list[CrystalFormHypothesis]:
        ...


class TemplateCrystalFormGenerator:
    """Deterministic baseline generator for early platform development."""

    def generate(
        self,
        request: str,
        objective: str,
        campaign_memory: CampaignMemory,
        research_hints: list[str],
        max_candidates: int = 3,
    ) -> list[CrystalFormHypothesis]:
        lowered = f"{request} {objective}".lower()
        candidates = [
            CrystalFormHypothesis(
                candidate_id="cand-alpha",
                molecule_label="Rigid aromatic donor-acceptor core",
                form_label="alpha",
                packing_hypothesis="slipped pi-stack with moderate anisotropy",
                expected_advantage="balanced transport and crystallization tractability",
                notes=[
                    "Favors planar cores",
                    "Compatible with low-complexity initial screening",
                ],
            ),
            CrystalFormHypothesis(
                candidate_id="cand-beta",
                molecule_label="Hydrogen-bond-assisted aromatic core",
                form_label="beta",
                packing_hypothesis="herringbone network with competitive dense polymorph",
                expected_advantage="good control case for polymorph-sensitive crystallization",
                notes=["Useful when solvent competition is explicitly varied"],
            ),
            CrystalFormHypothesis(
                candidate_id="cand-gamma",
                molecule_label="Porous molecular crystal motif",
                form_label="gamma-solvate-sensitive",
                packing_hypothesis="open packing stabilized by guest or solvent inclusion",
                expected_advantage="high-information porous benchmark",
                notes=["Retain for benchmarking, not immediate first-pass optimization"],
            ),
        ]

        if "porous" in lowered or "hof" in lowered:
            candidates[2].notes.append("Request explicitly mentions porous behavior.")
        if "transport" in lowered or "semiconductor" in lowered:
            candidates[0].notes.append("Request emphasizes charge-transport-relevant packing.")
        if campaign_memory.completed_trials:
            candidates[1].notes.append("Secondary control informed by prior trial history.")
        if research_hints:
            candidates[0].notes.append("Research brief available for downstream ranking.")

        return candidates[:max_candidates]


def get_default_generator() -> CrystalFormGenerator:
    return TemplateCrystalFormGenerator()
