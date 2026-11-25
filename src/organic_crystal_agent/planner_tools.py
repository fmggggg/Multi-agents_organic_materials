from __future__ import annotations

from collections.abc import Callable

from organic_crystal_agent.orchestrator import CampaignOrchestrator
from organic_crystal_agent.schemas import DiscoveryCampaign


PlannerTool = Callable[..., DiscoveryCampaign]


class PlannerToolRegistry:
    """Registry of high-level campaign actions available to the main planner."""

    def __init__(self) -> None:
        self._tools: dict[str, PlannerTool] = {}

    def register(self, name: str, fn: PlannerTool) -> None:
        self._tools[name] = fn

    def execute(self, name: str, **kwargs) -> DiscoveryCampaign:
        if name not in self._tools:
            raise KeyError(f"Unknown planner tool: {name}")
        return self._tools[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)


def build_planner_tool_registry(orchestrator: CampaignOrchestrator) -> PlannerToolRegistry:
    registry = PlannerToolRegistry()

    def collect_literature(
        campaign: DiscoveryCampaign,
        max_hypotheses: int,
        **_: object,
    ) -> DiscoveryCampaign:
        return orchestrator.auto_add_literature_hypotheses(campaign, max_hypotheses=max_hypotheses)

    def propose_molecules(
        campaign: DiscoveryCampaign,
        max_molecules: int,
        **_: object,
    ) -> DiscoveryCampaign:
        return orchestrator.auto_add_molecule_batch(campaign, max_molecules=max_molecules)

    def refresh_jobs(
        campaign: DiscoveryCampaign,
        **_: object,
    ) -> DiscoveryCampaign:
        return orchestrator.refresh_campaign(campaign)

    def decide_next_round(
        campaign: DiscoveryCampaign,
        **_: object,
    ) -> DiscoveryCampaign:
        return orchestrator.auto_decide_next_round(campaign)

    def submit_property_jobs(
        campaign: DiscoveryCampaign,
        **_: object,
    ) -> DiscoveryCampaign:
        return orchestrator.auto_submit_property_jobs(campaign)

    def submit_packing_jobs(
        campaign: DiscoveryCampaign,
        **_: object,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        existing = {record.molecule_id for record in updated.packing_evaluations}
        candidates = [
            molecule
            for molecule in updated.molecules
            if molecule.status in {"landscape_ready", "selected_for_followup", "property_ready"}
        ]
        for molecule in candidates:
            if molecule.molecule_id in existing:
                continue
            updated = orchestrator.submit_molecule_for_packing_analysis(updated, molecule.molecule_id)
        return updated

    def submit_ready_molecules(
        campaign: DiscoveryCampaign,
        sample_count: int,
        max_molecules: int,
        z_value: int | None,
        space_group: int | None,
        checkpoint_path: str,
        use_real_sg_ratio: bool,
        **_: object,
    ) -> DiscoveryCampaign:
        updated = campaign.model_copy(deep=True)
        submittable = [
            molecule
            for molecule in updated.molecules
            if molecule.status == "proposed" and molecule.molecule_path
        ]
        if len(submittable) < max_molecules:
            return updated
        for molecule in sorted(submittable, key=lambda item: item.molecule_id)[:max_molecules]:
            updated = orchestrator.submit_molecule_for_csp(
                campaign=updated,
                molecule_id=molecule.molecule_id,
                sample_count=sample_count,
                z_value=z_value,
                space_group=space_group,
                checkpoint_path=checkpoint_path,
                use_real_sg_ratio=use_real_sg_ratio,
            )
        return updated

    registry.register("collect_literature", collect_literature)
    registry.register("propose_molecules", propose_molecules)
    registry.register("refresh_jobs", refresh_jobs)
    registry.register("submit_ready_molecules", submit_ready_molecules)
    registry.register("submit_property_jobs", submit_property_jobs)
    registry.register("submit_packing_jobs", submit_packing_jobs)
    registry.register("decide_next_round", decide_next_round)
    return registry
