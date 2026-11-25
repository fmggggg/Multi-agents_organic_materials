from __future__ import annotations

import time
from dataclasses import dataclass

from organic_crystal_agent.main_planner import MainPlanner, PlannerConfig
from organic_crystal_agent.orchestrator import CampaignOrchestrator
from organic_crystal_agent.persistence import load_discovery_campaign, save_discovery_campaign
from organic_crystal_agent.schemas import DiscoveryCampaign
from organic_crystal_agent.campaign_summary import build_campaign_summary


@dataclass
class RunnerConfig:
    literature_hypotheses: int = 5
    molecule_batch_size: int = 5
    sample_count: int = 1000
    z_value: int | None = None
    space_group: int | None = None
    use_real_sg_ratio: bool = False
    checkpoint_path: str = ""
    auto_decide: bool = True
    auto_expand: bool = True

    def to_planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            literature_hypotheses=self.literature_hypotheses,
            molecule_batch_size=self.molecule_batch_size,
            sample_count=self.sample_count,
            z_value=self.z_value,
            space_group=self.space_group,
            use_real_sg_ratio=self.use_real_sg_ratio,
            checkpoint_path=self.checkpoint_path,
            auto_decide=self.auto_decide,
            auto_expand=self.auto_expand,
        )


class CampaignRunner:
    """Event-driven runtime delegating high-level reasoning to MainPlanner."""

    def __init__(
        self,
        orchestrator: CampaignOrchestrator | None = None,
        planner: MainPlanner | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.planner = planner or MainPlanner(
            orchestrator=orchestrator,
            config=self.config.to_planner_config(),
        )

    def run_once(self, campaign: DiscoveryCampaign, campaign_path: str | None = None) -> DiscoveryCampaign:
        return self.planner.step(campaign=campaign, campaign_path=campaign_path)

    def run_loop(
        self,
        campaign_path: str,
        poll_seconds: int = 300,
        max_cycles: int = 0,
    ) -> DiscoveryCampaign:
        cycles = 0
        while True:
            campaign = load_discovery_campaign(campaign_path)
            updated = self.run_once(campaign, campaign_path=campaign_path)
            save_discovery_campaign(campaign_path, updated)
            cycles += 1
            if max_cycles and cycles >= max_cycles:
                return updated
            active_jobs = [job for job in updated.csp_jobs if job.status in {"queued", "running"}]
            active_jobs.extend(job for job in updated.property_jobs if job.status in {"queued", "running"})
            if not active_jobs and updated.status not in {"waiting_on_jobs"}:
                return updated
            time.sleep(poll_seconds)


def summarize_campaign(campaign: DiscoveryCampaign) -> dict:
    summary = build_campaign_summary(campaign)
    return {
        "campaign_id": campaign.campaign_id,
        "status": campaign.status,
        "round_index": campaign.round_index,
        "desired_properties": campaign.desired_properties,
        "literature_hypotheses": len(campaign.literature_hypotheses),
        "molecule_counts": {
            "total": len(campaign.molecules),
            "proposed": sum(1 for molecule in campaign.molecules if molecule.status == "proposed"),
            "queued_for_csp": sum(1 for molecule in campaign.molecules if molecule.status == "queued_for_csp"),
            "running_csp": sum(1 for molecule in campaign.molecules if molecule.status == "running_csp"),
            "landscape_ready": sum(1 for molecule in campaign.molecules if molecule.status == "landscape_ready"),
            "queued_for_property": sum(1 for molecule in campaign.molecules if molecule.status == "queued_for_property"),
            "running_property": sum(1 for molecule in campaign.molecules if molecule.status == "running_property"),
            "property_ready": sum(1 for molecule in campaign.molecules if molecule.status == "property_ready"),
            "selected_for_followup": sum(
                1 for molecule in campaign.molecules if molecule.status == "selected_for_followup"
            ),
            "rejected": sum(1 for molecule in campaign.molecules if molecule.status == "rejected"),
        },
        "job_counts": {
            "total": len(campaign.csp_jobs),
            "queued": sum(1 for job in campaign.csp_jobs if job.status == "queued"),
            "running": sum(1 for job in campaign.csp_jobs if job.status == "running"),
            "succeeded": sum(1 for job in campaign.csp_jobs if job.status == "succeeded"),
            "failed": sum(1 for job in campaign.csp_jobs if job.status == "failed"),
            "cancelled": sum(1 for job in campaign.csp_jobs if job.status == "cancelled"),
        },
        "property_job_counts": {
            "total": len(campaign.property_jobs),
            "queued": sum(1 for job in campaign.property_jobs if job.status == "queued"),
            "running": sum(1 for job in campaign.property_jobs if job.status == "running"),
            "succeeded": sum(1 for job in campaign.property_jobs if job.status == "succeeded"),
            "failed": sum(1 for job in campaign.property_jobs if job.status == "failed"),
            "cancelled": sum(1 for job in campaign.property_jobs if job.status == "cancelled"),
        },
        "decision_count": len(campaign.decisions),
        "packing_evaluation_count": len(campaign.packing_evaluations),
        "hbond_mode_counts": summary.hbond_mode_counts,
        "hbond_mode_highlights": summary.hbond_mode_highlights,
    }
