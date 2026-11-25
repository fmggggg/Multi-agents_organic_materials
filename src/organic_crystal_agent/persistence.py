from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from organic_crystal_agent.schemas import CampaignMemory, DiscoveryCampaign, NextExperimentProposal

T = TypeVar("T", bound=BaseModel)


def save_model(path: str | Path, model: BaseModel) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.model_dump(), indent=2), encoding="utf-8")


def load_model(path: str | Path, model_type: type[T]) -> T:
    raw = Path(path).read_text(encoding="utf-8")
    return model_type.model_validate(json.loads(raw))


def load_campaign_memory(path: str | Path) -> CampaignMemory:
    return load_model(path, CampaignMemory)


def save_campaign_memory(path: str | Path, memory: CampaignMemory) -> None:
    save_model(path, memory)


def load_proposal(path: str | Path) -> NextExperimentProposal:
    return load_model(path, NextExperimentProposal)


def save_proposal(path: str | Path, proposal: NextExperimentProposal) -> None:
    save_model(path, proposal)


def load_discovery_campaign(path: str | Path) -> DiscoveryCampaign:
    return load_model(path, DiscoveryCampaign)


def save_discovery_campaign(path: str | Path, campaign: DiscoveryCampaign) -> None:
    save_model(path, campaign)
