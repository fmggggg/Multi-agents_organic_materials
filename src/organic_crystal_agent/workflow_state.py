from __future__ import annotations

import json
from pathlib import Path


def stage_state_path(workdir: str | Path) -> Path:
    return Path(workdir) / "stage_state.json"


def initialize_stage_state(workdir: str | Path, stages: list[str]) -> dict[str, str]:
    state = {stage: "pending" for stage in stages}
    write_stage_state(workdir, state)
    return state


def load_stage_state(workdir: str | Path) -> dict[str, str]:
    path = stage_state_path(workdir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage_state(workdir: str | Path, state: dict[str, str]) -> None:
    path = stage_state_path(workdir)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def update_stage_state(workdir: str | Path, stage: str, status: str) -> dict[str, str]:
    state = load_stage_state(workdir)
    state[stage] = status
    write_stage_state(workdir, state)
    return state
