from __future__ import annotations

from typing import TypedDict

from organic_crystal_agent.schemas import CampaignMemory, NextExperimentProposal, Observation, ToolCall


class AgentState(TypedDict):
    request: str
    objective: str
    campaign_memory: CampaignMemory
    pending_steps: list[ToolCall]
    observations: list[Observation]
    proposal: NextExperimentProposal | None
