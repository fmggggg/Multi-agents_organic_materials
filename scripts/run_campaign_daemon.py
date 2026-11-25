#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from organic_crystal_agent.campaign_runner import CampaignRunner, RunnerConfig, summarize_campaign
from organic_crystal_agent.persistence import load_discovery_campaign, save_discovery_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the organic crystal campaign loop as a daemon-like process.")
    parser.add_argument("--campaign-path", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--z-value", type=int, default=None)
    parser.add_argument("--space-group", type=int, default=None)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--use-real-sg-ratio", action="store_true")
    parser.add_argument("--molecule-batch-size", type=int, default=5)
    parser.add_argument("--literature-hypotheses", type=int, default=5)
    args = parser.parse_args()

    runner = CampaignRunner(
        config=RunnerConfig(
            literature_hypotheses=args.literature_hypotheses,
            molecule_batch_size=args.molecule_batch_size,
            sample_count=args.sample_count,
            z_value=args.z_value,
            space_group=args.space_group,
            checkpoint_path=args.checkpoint_path,
            use_real_sg_ratio=args.use_real_sg_ratio,
        )
    )

    cycles = 0
    campaign_path = Path(args.campaign_path)
    while True:
        campaign = load_discovery_campaign(campaign_path)
        updated = runner.run_once(campaign, campaign_path=str(campaign_path))
        save_discovery_campaign(campaign_path, updated)
        print(json.dumps(summarize_campaign(updated), indent=2), flush=True)
        cycles += 1
        if args.max_cycles and cycles >= args.max_cycles:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
