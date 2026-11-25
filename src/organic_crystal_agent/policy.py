from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from organic_crystal_agent.schemas import CampaignMemory, CrystalFormHypothesis, CrystallizationTrial


class ExperimentPolicy(Protocol):
    """Policy interface for selecting the next experiment(s)."""

    def propose_trials(
        self,
        candidates: list[CrystalFormHypothesis],
        campaign_memory: CampaignMemory,
        batch_size: int = 1,
    ) -> tuple[CrystallizationTrial, list[CrystallizationTrial]]:
        ...


@dataclass
class ReasoningPolicy:
    """Deterministic baseline policy for experiment-by-experiment planning."""

    def propose_trials(
        self,
        candidates: list[CrystalFormHypothesis],
        campaign_memory: CampaignMemory,
        batch_size: int = 1,
    ) -> tuple[CrystallizationTrial, list[CrystallizationTrial]]:
        from organic_crystal_agent.tools.organic_crystal import build_trial_for_policy

        primary = build_trial_for_policy(candidates[0], campaign_memory.round_index, 1, campaign_memory)
        backups = [
            build_trial_for_policy(candidate, campaign_memory.round_index, idx + 2, campaign_memory)
            for idx, candidate in enumerate(candidates[1 : max(1, batch_size + 1)])
        ]
        return primary, backups
