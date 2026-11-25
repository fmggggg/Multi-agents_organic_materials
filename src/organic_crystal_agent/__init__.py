"""Organic crystal agent package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_campaign_step", "update_campaign_memory"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        workflow = import_module("organic_crystal_agent.workflow")
        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
