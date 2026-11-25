from __future__ import annotations

import csv
import json
from pathlib import Path

from organic_crystal_agent.application_specs import build_application_spec
from organic_crystal_agent.campaign_summary import build_campaign_summary
from organic_crystal_agent.local_literature import render_open_access_literature
from organic_crystal_agent.schemas import DiscoveryCampaign


def resolve_campaign_root(campaign_path: str | Path | None, campaign_id: str) -> Path:
    if campaign_path:
        return Path(campaign_path).resolve().parent
    return Path("campaigns") / campaign_id


class MemoryStore:
    """Workspace-first memory layout inspired by modern agent runtimes."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.root = Path(campaign_root)
        self.memory_dir = self.root / "memory"
        self.reports_dir = self.root / "reports"
        self.jobs_dir = self.root / "jobs"
        self.molecules_dir = self.root / "molecules"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.molecules_dir.mkdir(parents=True, exist_ok=True)

    def write_campaign_memory(self, campaign: DiscoveryCampaign) -> None:
        self.ensure()
        application_spec = build_application_spec(campaign)
        self._write_text("application_spec.md", self._render_application_spec(campaign))
        self._write_text("literature.md", self._render_literature(campaign))
        self._write_text("literature_intake_template.md", self._render_literature_intake_template(campaign))
        self._write_text("open_access_literature.md", render_open_access_literature(campaign))
        self._write_text("molecule_design_rules.md", self._render_design_rules(campaign))
        self._write_text("hbond_mode_summary.md", self._render_hbond_mode_summary(campaign))
        self._write_text("failure_patterns.md", self._render_failure_patterns(campaign))
        self._write_text("winning_patterns.md", self._render_winning_patterns(campaign))
        self._write_text("knowledge_insights.md", self._render_knowledge_insights(campaign))
        self._write_text("search_bias.md", self._render_search_bias(campaign))
        self._write_text(f"round_{campaign.round_index:03d}.md", self._render_round_snapshot(campaign))
        (self.reports_dir / "application_spec.json").write_text(
            application_spec.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.reports_dir / "campaign_summary.json").write_text(
            build_campaign_summary(campaign).model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.reports_dir / "campaign_snapshot.json").write_text(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "status": campaign.status,
                    "round_index": campaign.round_index,
                    "molecules": len(campaign.molecules),
                    "jobs": len(campaign.csp_jobs),
                    "decisions": len(campaign.decisions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.reports_dir / "knowledge_report.json").write_text(
            json.dumps(self._build_knowledge_report(campaign), indent=2),
            encoding="utf-8",
        )

    def load_context_bundle(self) -> dict[str, str]:
        self.ensure()
        bundle = {}
        for name in [
            "application_spec.md",
            "literature.md",
            "literature_intake_template.md",
            "open_access_literature.md",
            "molecule_design_rules.md",
            "hbond_mode_summary.md",
            "failure_patterns.md",
            "winning_patterns.md",
            "knowledge_insights.md",
            "search_bias.md",
        ]:
            path = self.memory_dir / name
            bundle[name] = path.read_text(encoding="utf-8") if path.exists() else ""
        return bundle

    def _write_text(self, name: str, content: str) -> None:
        (self.memory_dir / name).write_text(content.strip() + "\n", encoding="utf-8")

    def _render_application_spec(self, campaign: DiscoveryCampaign) -> str:
        spec = build_application_spec(campaign)
        lines = [
            "# Application Spec",
            "",
            f"Campaign: {campaign.campaign_id}",
            f"Application mode: {spec.application_mode}",
            f"Evidence source: {spec.evidence_source}",
            "",
            "## Hard Constraints",
            "",
        ]
        lines.extend(f"- {item}" for item in spec.hard_constraints or ["none"])
        lines.extend(["", "## Soft Preferences", ""])
        lines.extend(f"- {item}" for item in spec.soft_preferences or ["none"])
        lines.extend(["", "## Red Flags", ""])
        lines.extend(f"- {item}" for item in spec.red_flags or ["none"])
        lines.extend(["", "## Evaluation Criteria", ""])
        lines.extend(f"- {item}" for item in spec.evaluation_criteria or ["none"])
        lines.extend(["", "## Synthesis Heuristics", ""])
        lines.extend(f"- {item}" for item in spec.synthesis_heuristics or ["none"])
        lines.extend(["", "## Search Queries", ""])
        lines.extend(f"- {item}" for item in spec.search_queries or ["none"])
        return "\n".join(lines)

    def _render_literature(self, campaign: DiscoveryCampaign) -> str:
        lines = [f"# Literature Memory", f"", f"Campaign: {campaign.campaign_id}", ""]
        if not campaign.literature_hypotheses:
            lines.append("No literature hypotheses recorded.")
            return "\n".join(lines)
        for item in campaign.literature_hypotheses:
            lines.extend(
                [
                    f"## {item.title}",
                    item.summary,
                    f"Confidence: {item.confidence:.2f}",
                    f"Evidence: {', '.join(item.evidence) if item.evidence else 'none'}",
                    f"Design implications: {', '.join(item.design_implications) if item.design_implications else 'none'}",
                    f"Risk flags: {', '.join(item.risk_flags) if item.risk_flags else 'none'}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _render_literature_intake_template(self, campaign: DiscoveryCampaign) -> str:
        target_text = campaign.target_description.lower()
        humid_co2 = "co2" in target_text and ("humid" in target_text or "h2o" in target_text or "water" in target_text)
        methane = "methane" in target_text or "ch4" in target_text
        relevance_hint = (
            "humid CO2 capture / CO2-vs-H2O competition / porous HOF design / activation stability / crystal engineering / negative result"
            if humid_co2
            else "methane adsorption / porous HOF design / activation stability / crystal engineering / negative result"
            if methane
            else "target property / porous HOF design / activation stability / crystal engineering / negative result"
        )
        gas_hint = (
            "CO2 with H2O / humid feed (and any co-adsorbates)"
            if humid_co2
            else "CH4 (and any co-adsorbates)"
            if methane
            else "target gas or co-adsorbate system"
        )
        metric_hint = (
            "CO2 uptake, H2O uptake, selectivity, working capacity, or note that none were reported"
            if humid_co2
            else "uptake, working capacity, selectivity, deliverable capacity, or note that none were reported"
        )
        return "\n".join(
            [
                "# Literature Intake Template",
                "",
                f"Campaign: {campaign.campaign_id}",
                "",
                "Use this template when adding a new paper, preprint, thesis chapter, or supplemental note to the campaign memory.",
                "Keep entries short and concrete so the planner can reuse them directly.",
                "",
                "## Intake Fields",
                "",
                "- Citation: full title, authors, journal or preprint source, year, DOI if available",
                "- Access status: full text available / abstract only / paywalled",
                f"- Relevance: {relevance_hint}",
                "- Scaffold family: core topology and tecton family",
                "- Functional groups: donor or acceptor groups that actually matter",
                "- Porosity outcome: porous / non-porous / collapses on activation / unclear",
                f"- Adsorption evidence: explicit values/conditions for {metric_hint}",
                "- Structural lesson: what about the crystal packing seems to help or hurt the target property",
                "- What to emulate: the one or two design ideas worth reusing",
                "- What to avoid: motifs, substituents, solvent traps, dense spikes, or collapse patterns to stop repeating",
                "- Confidence: high / medium / low",
                "",
                "## Example Entry Skeleton",
                "",
                "### Paper Title Here",
                "- Citation:",
                "- Access status:",
                "- Relevance:",
                "- Scaffold family:",
                "- Functional groups:",
                "- Porosity outcome:",
                f"- Adsorption evidence ({gas_hint}):",
                "- Structural lesson:",
                "- What to emulate:",
                "- What to avoid:",
                "- Confidence:",
            ]
        )

    def _render_design_rules(self, campaign: DiscoveryCampaign) -> str:
        spec = build_application_spec(campaign)
        lines = ["# Molecule Design Rules", ""]
        lines.append(f"Target description: {campaign.target_description}")
        lines.append(
            f"Desired properties: {', '.join(campaign.desired_properties) if campaign.desired_properties else 'none'}"
        )
        lines.append(f"Application mode: {spec.application_mode} ({spec.evidence_source})")
        lines.append("")
        lines.append("## Current Application Prior")
        lines.append("")
        lines.extend(f"- hard: {rule}" for rule in spec.hard_constraints)
        lines.extend(f"- prefer: {rule}" for rule in spec.soft_preferences[:3])
        lines.extend(f"- avoid: {rule}" for rule in spec.red_flags[:3])
        lines.append("")
        implications = []
        for item in campaign.literature_hypotheses:
            implications.extend(self._normalize_humid_design_implication(rule) for rule in item.design_implications)
        if not implications:
            lines.append("No explicit design rules recorded yet.")
        else:
            for rule in sorted(dict.fromkeys(implications)):
                lines.append(f"- {rule}")
        lines.append("")
        lines.append("## Proposed Molecules")
        lines.append("")
        for molecule in campaign.molecules:
            lines.append(
                f"- {molecule.molecule_id}: {molecule.molecule_label} | family={molecule.design_family or 'unspecified'} | status={molecule.status}"
            )
        return "\n".join(lines)

    def _render_failure_patterns(self, campaign: DiscoveryCampaign) -> str:
        lines = ["# Failure Patterns", ""]
        findings = []
        for job in campaign.csp_jobs:
            if job.metrics and job.metrics.failure_fraction is not None:
                findings.append(
                    f"{job.molecule_id}: failure_fraction={job.metrics.failure_fraction:.2f}, notes={'; '.join(job.metrics.notes)}"
                )
        if not findings:
            lines.append("No failure patterns observed yet.")
        else:
            lines.extend(f"- {item}" for item in findings)
        return "\n".join(lines)

    def _render_hbond_mode_summary(self, campaign: DiscoveryCampaign) -> str:
        summary = build_campaign_summary(campaign)
        lines = ["# Hydrogen-Bonding Mode Summary", ""]
        if not summary.hbond_mode_highlights:
            lines.append("No explicit hydrogen-bonding modes have been identified yet.")
            return "\n".join(lines)
        lines.append("## Mode Counts")
        lines.append("")
        for mode, count in sorted(summary.hbond_mode_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {mode}: {count}")
        lines.append("")
        lines.append("## Mode Highlights")
        lines.append("")
        lines.extend(f"- {item}" for item in summary.hbond_mode_highlights)
        return "\n".join(lines)

    def _render_winning_patterns(self, campaign: DiscoveryCampaign) -> str:
        lines = ["# Winning Patterns", ""]
        if "humid" in campaign.target_description.lower() or any(
            term in " ".join(campaign.desired_properties).lower() for term in ("co2", "h2o", "humid")
        ):
            lines.extend(
                [
                    "Humid-capture note: `property_score` = `CO2 uptake × sqrt((CO2+1e-3)/(H2O+1e-3))`.",
                    "",
                ]
            )
        findings = []
        property_findings: list[tuple[float, str]] = []
        for job in campaign.csp_jobs:
            if job.metrics and (job.metrics.low_density_spike_score or 0.0) >= 5.0:
                findings.append(
                    f"{job.molecule_id}: spike={job.metrics.low_density_spike_score:.2f}, porous_survival={job.metrics.porous_survival_rate or 0.0:.2f}"
                )
        official_properties = self._load_official_humid_property_summary(campaign)
        if official_properties:
            for item in official_properties:
                ranking_value = float(item["property_score"] or 0.0)
                summary = (
                    f"{item['molecule_id']}: "
                    f"co2_abs={float(item['co2_uptake_absolute_mmol_g'] or 0.0):.3f} mmol/g, "
                    f"h2o_abs={float(item['h2o_uptake_absolute_mmol_g'] or 0.0):.3f} mmol/g, "
                    f"score={ranking_value:.3f}"
                )
                property_findings.append((ranking_value, summary))
        else:
            for job in campaign.property_jobs:
                metrics = job.metrics
                if not metrics:
                    continue
                summary: str | None = None
                ranking_value: float | None = None
                if metrics.co2_uptake_absolute_mmol_g is not None or metrics.h2o_uptake_absolute_mmol_g is not None:
                    ranking_value = float(metrics.property_score or metrics.co2_uptake_absolute_mmol_g or 0.0)
                    summary = (
                        f"{job.molecule_id}: co2_abs={(metrics.co2_uptake_absolute_mmol_g or 0.0):.3f} mmol/g, "
                        f"co2_exc={(metrics.co2_uptake_excess_mmol_g or 0.0):.3f} mmol/g, "
                        f"h2o_abs={(metrics.h2o_uptake_absolute_mmol_g or 0.0):.3f} mmol/g, "
                        f"h2o_exc={(metrics.h2o_uptake_excess_mmol_g or 0.0):.3f} mmol/g, "
                        f"score={(metrics.property_score or 0.0):.3f}"
                    )
                elif metrics.methane_uptake_absolute_mmol_g is not None:
                    ranking_value = float(metrics.property_score or metrics.methane_uptake_absolute_mmol_g or 0.0)
                    summary = (
                        f"{job.molecule_id}: methane_abs={metrics.methane_uptake_absolute_mmol_g:.3f} mmol/g, "
                        f"methane_exc={(metrics.methane_uptake_excess_mmol_g or 0.0):.3f} mmol/g, "
                        f"score={(metrics.property_score or 0.0):.3f}"
                    )
                elif metrics.property_score is not None:
                    ranking_value = float(metrics.property_score)
                    summary = f"{job.molecule_id}: property_score={metrics.property_score:.3f}"
                if summary is not None and ranking_value is not None:
                    property_findings.append((ranking_value, summary))
        if not findings and not property_findings:
            lines.append("No winning patterns identified yet.")
        else:
            if findings:
                lines.append("## CSP Landscape Winners")
                lines.append("")
                lines.extend(f"- {item}" for item in findings)
                lines.append("")
            if property_findings:
                lines.append("## Property Winners")
                lines.append("")
                property_findings.sort(reverse=True, key=lambda item: item[0])
                lines.extend(f"- {item}" for _, item in property_findings)
                lines.append("")
        return "\n".join(lines)

    def _render_round_snapshot(self, campaign: DiscoveryCampaign) -> str:
        lines = [
            f"# Round {campaign.round_index}",
            "",
            f"Campaign status: {campaign.status}",
            f"Molecule count: {len(campaign.molecules)}",
            f"Job count: {len(campaign.csp_jobs)}",
            f"Decision count: {len(campaign.decisions)}",
            f"Planner traces: {len(campaign.planner_traces)}",
            "",
        ]
        if campaign.planner_traces:
            trace = campaign.planner_traces[-1]
            lines.extend(
                [
                    "## Latest Planner Trace",
                    f"Planner mode: {trace.planner_mode}",
                    f"Available tools: {', '.join(trace.available_tools) if trace.available_tools else 'none'}",
                    f"Chosen actions: {', '.join(trace.chosen_actions) if trace.chosen_actions else 'none'}",
                    f"Action rationales: {' | '.join(trace.action_rationales) if trace.action_rationales else 'none'}",
                    f"Notes: {', '.join(trace.notes) if trace.notes else 'none'}",
                    "",
                ]
            )
        if campaign.decisions:
            decision = campaign.decisions[-1]
            lines.extend(
                [
                    "## Latest Decision",
                    f"Selected: {', '.join(decision.selected_molecule_ids) if decision.selected_molecule_ids else 'none'}",
                    f"Rejected: {', '.join(decision.rejected_molecule_ids) if decision.rejected_molecule_ids else 'none'}",
                    f"Actions: {', '.join(decision.next_actions) if decision.next_actions else 'none'}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _build_knowledge_report(self, campaign: DiscoveryCampaign) -> dict:
        summary = build_campaign_summary(campaign)
        top_landscapes: list[dict] = []
        for job in campaign.csp_jobs:
            if not job.metrics:
                continue
            score = float(job.metrics.low_density_spike_score or 0.0) + 2.0 * float(job.metrics.porous_survival_rate or 0.0)
            top_landscapes.append(
                {
                    "molecule_id": job.molecule_id,
                    "molecule_label": job.molecule_label,
                    "design_family": self._molecule_family(campaign, job.molecule_id),
                    "landscape_score_proxy": round(score, 4),
                    "low_density_spike_score": job.metrics.low_density_spike_score,
                    "porous_survival_rate": job.metrics.porous_survival_rate,
                    "collapse_risk_score": job.metrics.collapse_risk_score,
                    "failure_fraction": job.metrics.failure_fraction,
                }
            )
        top_landscapes.sort(key=lambda item: item["landscape_score_proxy"], reverse=True)

        top_properties = self._load_official_humid_property_summary(campaign)
        if not top_properties:
            top_properties = []
            for job in campaign.property_jobs:
                if not job.metrics:
                    continue
                metrics = job.metrics
                co2 = float(metrics.co2_uptake_absolute_mmol_g or 0.0)
                h2o = float(metrics.h2o_uptake_absolute_mmol_g or 0.0)
                selectivity_proxy = round(co2 / max(h2o, 1e-6), 6) if (co2 > 0.0 or h2o > 0.0) else None
                top_properties.append(
                    {
                        "molecule_id": job.molecule_id,
                        "molecule_label": job.molecule_label,
                        "design_family": self._molecule_family(campaign, job.molecule_id),
                        "property_profile": metrics.property_profile,
                        "property_score": metrics.property_score,
                        "co2_uptake_absolute_mmol_g": metrics.co2_uptake_absolute_mmol_g,
                        "h2o_uptake_absolute_mmol_g": metrics.h2o_uptake_absolute_mmol_g,
                        "methane_uptake_absolute_mmol_g": metrics.methane_uptake_absolute_mmol_g,
                        "selectivity_proxy_co2_over_h2o": selectivity_proxy,
                    }
                )
        top_properties.sort(
            key=lambda item: (
                float(item["property_score"] or -1.0),
                float(item["selectivity_proxy_co2_over_h2o"] or -1.0),
            ),
            reverse=True,
        )

        dominant_families: dict[str, int] = {}
        for molecule in campaign.molecules:
            family = molecule.design_family.strip() or "unspecified"
            dominant_families[family] = dominant_families.get(family, 0) + 1

        failure_signals = self._knowledge_failure_signals(campaign)
        search_bias = self._knowledge_search_bias(campaign, top_landscapes=top_landscapes, top_properties=top_properties)
        next_round_hypotheses = self._knowledge_next_round_hypotheses(campaign)

        return {
            "campaign_id": campaign.campaign_id,
            "round_index": campaign.round_index,
            "status": campaign.status,
            "dominant_design_families": [
                {"design_family": family, "count": count}
                for family, count in sorted(dominant_families.items(), key=lambda item: (-item[1], item[0]))
            ],
            "top_landscape_candidates": top_landscapes[:5],
            "top_property_candidates": top_properties[:5],
            "failure_signals": failure_signals,
            "search_bias": search_bias,
            "next_round_hypotheses": next_round_hypotheses,
            "campaign_summary": summary.model_dump(),
        }

    def _load_official_humid_property_summary(self, campaign: DiscoveryCampaign) -> list[dict]:
        repo_reports = Path(__file__).resolve().parents[2] / "reports"
        candidates = [
            self.reports_dir / "humid_co2_pareto_summary.csv",
            repo_reports / "humid_co2_pareto_summary.csv",
        ]
        rows = None
        for summary_path in candidates:
            if not summary_path.exists():
                continue
            try:
                with summary_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except Exception:
                rows = None
            if rows:
                break
        if not rows:
            return []
        top_properties: list[dict] = []
        for row in rows:
            molecule_id = (row.get("molecule_id") or "").strip()
            if not molecule_id:
                continue
            top_properties.append(
                {
                    "molecule_id": molecule_id,
                    "molecule_label": molecule_id,
                    "design_family": self._molecule_family(campaign, molecule_id),
                    "property_profile": "humid_co2_capture",
                    "property_score": row.get("score"),
                    "co2_uptake_absolute_mmol_g": row.get("co2_mmol_g"),
                    "h2o_uptake_absolute_mmol_g": row.get("h2o_mmol_g"),
                    "methane_uptake_absolute_mmol_g": None,
                    "selectivity_proxy_co2_over_h2o": row.get("selectivity_proxy"),
                }
            )
        return top_properties

    def _render_knowledge_insights(self, campaign: DiscoveryCampaign) -> str:
        report = self._build_knowledge_report(campaign)
        lines = [
            "# Knowledge Insights",
            "",
            f"Campaign: {campaign.campaign_id}",
            f"Round: {campaign.round_index}",
            f"Status: {campaign.status}",
            "",
        ]
        if "humid" in campaign.target_description.lower() or any(
            term in " ".join(campaign.desired_properties).lower() for term in ("co2", "h2o", "humid")
        ):
            lines.extend(
                [
                    "Humid-capture score formula: `CO2 uptake × sqrt((CO2+1e-3)/(H2O+1e-3))`.",
                    "",
                ]
            )
        lines.extend(["## Top Landscape Signals", ""])
        top_landscapes = report["top_landscape_candidates"]
        if not top_landscapes:
            lines.append("No completed landscape signals yet.")
        else:
            for item in top_landscapes:
                lines.append(
                    "- "
                    + f"{item['molecule_id']} ({item['design_family']}): "
                    + f"spike={item['low_density_spike_score']}, "
                    + f"porous_survival={item['porous_survival_rate']}, "
                    + f"collapse_risk={item['collapse_risk_score']}"
                )
        lines.extend(["", "## Top Property Signals", ""])
        top_properties = report["top_property_candidates"]
        if not top_properties:
            lines.append("No completed property signals yet.")
        else:
            for item in top_properties:
                lines.append(
                    "- "
                    + f"{item['molecule_id']} ({item['design_family']}): "
                    + f"profile={item['property_profile']}, "
                    + f"score={item['property_score']}, "
                    + f"CO2={item['co2_uptake_absolute_mmol_g']}, "
                    + f"H2O={item['h2o_uptake_absolute_mmol_g']}, "
                    + f"selectivity_proxy={item['selectivity_proxy_co2_over_h2o']}"
                )
        lines.extend(["", "## Failure Signals", ""])
        failure_signals = report["failure_signals"]
        if not failure_signals:
            lines.append("No durable failure pattern has been extracted yet.")
        else:
            lines.extend(f"- {item}" for item in failure_signals)
        lines.extend(["", "## Next-Round Hypotheses", ""])
        next_round_hypotheses = report["next_round_hypotheses"]
        if not next_round_hypotheses:
            lines.append("No next-round hypothesis has been distilled yet.")
        else:
            lines.extend(f"- {item}" for item in next_round_hypotheses)
        return "\n".join(lines)

    def _render_search_bias(self, campaign: DiscoveryCampaign) -> str:
        report = self._build_knowledge_report(campaign)
        lines = [
            "# Search Bias",
            "",
            "These rules are intended for the next molecule-design round. They are derived from actual campaign outcomes, not only static literature priors.",
            "",
        ]
        biases = report["search_bias"]
        if not biases:
            lines.append("No strong search bias extracted yet.")
            return "\n".join(lines)
        lines.extend(f"- {item}" for item in biases)
        return "\n".join(lines)

    def _knowledge_failure_signals(self, campaign: DiscoveryCampaign) -> list[str]:
        signals: list[str] = []
        collapse_heavy = [
            job for job in campaign.csp_jobs
            if job.metrics and (job.metrics.collapse_risk_score or 0.0) >= 0.7
        ]
        if collapse_heavy:
            labels = ", ".join(job.molecule_id for job in collapse_heavy[:4])
            signals.append(
                f"Several CSP outcomes are collapse-prone or dense-packed ({labels}); shape persistence is a major bottleneck."
            )

        poor_survival = [
            job for job in campaign.csp_jobs
            if job.metrics and (job.metrics.porous_survival_rate or 0.0) < 0.05
        ]
        if poor_survival:
            labels = ", ".join(job.molecule_id for job in poor_survival[:4])
            signals.append(
                f"Porous survival is weak for multiple molecules ({labels}); merely reaching a low-density spike is not enough."
            )

        synthesis_rejects = 0
        for trace in campaign.planner_traces:
            synthesis_rejects += sum(
                1
                for item in trace.action_rationales
                if item.startswith("rejected retrosynthesis gate:") or item.startswith("rejected critic gate:")
            )
        if synthesis_rejects:
            signals.append(
                f"Synthetic-feasibility screening is a persistent gatekeeper: {synthesis_rejects} drafted candidates were rejected before registration, so the search should stay practical without collapsing into only the safest motifs."
            )

        humid_failures = []
        for job in campaign.property_jobs:
            metrics = job.metrics
            if not metrics:
                continue
            co2 = float(metrics.co2_uptake_absolute_mmol_g or 0.0)
            h2o = float(metrics.h2o_uptake_absolute_mmol_g or 0.0)
            if h2o > 0.0 and co2 > 0.0 and h2o >= co2:
                humid_failures.append(job.molecule_id)
        if humid_failures:
            signals.append(
                f"Some humid-capture candidates show H2O uptake comparable to or larger than CO2 ({', '.join(humid_failures[:4])}); water-lean pore chemistry still matters."
            )
        return signals

    def _knowledge_search_bias(
        self,
        campaign: DiscoveryCampaign,
        *,
        top_landscapes: list[dict],
        top_properties: list[dict],
    ) -> list[str]:
        biases: list[str] = []
        family_counts: dict[str, int] = {}
        for molecule in campaign.molecules:
            family = molecule.design_family.strip() or "unspecified"
            family_counts[family] = family_counts.get(family, 0) + 1
        dominant_families = [family for family, count in family_counts.items() if count >= 3 and family != "unspecified"]
        if dominant_families:
            biases.append(
                "Diversify away from overrepresented design families in the next round: "
                + ", ".join(sorted(dominant_families[:4]))
                + "."
            )

        if any((item.get("collapse_risk_score") or 0.0) >= 0.7 for item in top_landscapes) or any(
            job.metrics and (job.metrics.porous_survival_rate or 0.0) < 0.05 for job in campaign.csp_jobs
        ):
            biases.append(
                "Bias toward more shape-persistent, less interdigitated tectons; weak porous survival suggests the search should reward rigidity and anti-collapse packing features."
            )

        if any(
            ("rejected retrosynthesis gate:" in item) or ("rejected critic gate:" in item)
            for trace in campaign.planner_traces
            for item in trace.action_rationales
        ):
            biases.append(
                "Keep most proposals modular and avoid heavily decorated repeated-arm scaffolds, but preserve a minority branch of higher-upside literature-forward motifs instead of collapsing into only the safest families."
            )

        humid_ranked = [
            item for item in top_properties
            if item.get("co2_uptake_absolute_mmol_g") is not None or item.get("h2o_uptake_absolute_mmol_g") is not None
        ]
        if humid_ranked:
            best = humid_ranked[0]
            ratio = best.get("selectivity_proxy_co2_over_h2o")
            try:
                ratio_value = float(ratio) if ratio is not None else None
            except (TypeError, ValueError):
                ratio_value = None
            if ratio_value is not None and ratio_value < 1.0:
                biases.append(
                    "Do not chase raw CO2 affinity with strongly hydrophilic pore linings; the current best humid-capture selectivity proxy is still weak."
                )
            else:
                biases.append(
                    "Keep selectivity as the primary objective over absolute uptake, but do not anchor the next round on any single top-scoring scaffold."
                )

        if "humid" in campaign.target_description.lower() or any(
            term in " ".join(campaign.desired_properties).lower() for term in ("co2", "h2o", "humid")
        ):
            biases.append(
                "Keep using the Nat Chem / XLSX fragment evidence as head-group guidance, especially for CO2-vs-H2O balance rather than dry-gas intuition."
            )
        return list(dict.fromkeys(biases))

    def _knowledge_next_round_hypotheses(self, campaign: DiscoveryCampaign) -> list[str]:
        hypotheses: list[str] = []
        for item in campaign.literature_hypotheses[:4]:
            if item.design_implications:
                hypotheses.append(self._normalize_humid_design_implication(item.design_implications[0]))
        for decision in campaign.decisions[-2:]:
            if decision.rationale:
                hypotheses.append(self._normalize_humid_design_implication(decision.rationale[0]))
        return list(dict.fromkeys(hypotheses))[:6]

    @staticmethod
    def _molecule_family(campaign: DiscoveryCampaign, molecule_id: str) -> str:
        for molecule in campaign.molecules:
            if molecule.molecule_id == molecule_id:
                return molecule.design_family or "unspecified"
        return "unspecified"

    @staticmethod
    def _normalize_humid_design_implication(text: str) -> str:
        replacements = {
            "Bias molecule proposals toward aromatic clip-like surfaces with distributed weak pi contacts.":
                "Favor broad aromatic pi-surfaces with distributed weak contacts, but maintain scaffold diversity across fused aza-aromatics, polyaryls, and contorted rigid tectons.",
            "Prefer bent or concave aromatic surfaces over strongly hydrophilic pore-lining motifs.":
                "Prefer low-water aromatic pore environments over strongly hydrophilic pore-lining motifs, without collapsing the search onto one cavity shape.",
            "Actively explore five-membered heteroaromatic or pentagonal cavity-forming motifs, especially when they can be expressed through HOF-credible tectons.":
                "Keep five-membered heteroaromatic or pentagonal cavity-forming motifs as one optional branch among several HOF-credible tecton families.",
            "Treat five-membered ring and pentagonal cavity motifs as a favored route when they can still support HOF assembly.":
                "Treat five-membered ring and pentagonal cavity motifs as one plausible route when they support HOF assembly, but do not let them dominate proposal space.",
            "Actively test pyrazole orientation for pentagonal or five-membered cavity pockets rather than treating pyrazole only as a hydrogen-bond synthon.":
                "Test pyrazole orientation when it improves pore shape or framework assembly, but do not force every branch back into pentagonal or five-membered cavity motifs.",
            "Add a shape filter for CO2-sized pentagonal pockets and reject candidates where water can form continuous H-bond chains through the pore.":
                "Add a shape filter for CO2-sized pockets and reject candidates where water can form continuous H-bond chains through the pore.",
        }
        return replacements.get(text, text)
