from __future__ import annotations

from collections.abc import Callable

from organic_crystal_agent.schemas import Observation


ToolFunction = Callable[..., Observation]


class ToolRegistry:
    """Minimal structured registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, fn: ToolFunction) -> None:
        self._tools[name] = fn

    def execute(self, name: str, **kwargs) -> Observation:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)
