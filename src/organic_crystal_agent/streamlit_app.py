from __future__ import annotations

from pathlib import Path

import streamlit as st

from organic_crystal_agent.persistence import load_campaign_memory, load_proposal
from organic_crystal_agent.workflow import plan_with_paths, record_with_paths


def main() -> None:
    st.set_page_config(page_title="Organic Crystal Agent", layout="wide")
    st.title("Organic Crystal Agent")
    st.caption("Plan the next experiment, record results, and inspect campaign memory.")

    with st.sidebar:
        st.header("Campaign")
        campaign_path = st.text_input("Campaign path", value="campaigns/default_campaign.json")
        proposal_path = st.text_input("Proposal path", value="campaigns/latest_proposal.json")
        st.divider()
        st.header("Planning")
        request = st.text_area(
            "Request",
            value="Find a promising organic crystal starting point for charge transport",
            height=100,
        )
        objective = st.text_input(
            "Objective",
            value="Propose the single most informative next organic crystal experiment.",
        )
        if st.button("Plan Next Experiment", use_container_width=True):
            proposal = plan_with_paths(
                request=request,
                objective=objective,
                campaign_path=campaign_path,
                proposal_path=proposal_path,
            )
            st.session_state["latest_proposal"] = proposal.model_dump()
            st.success("Planned the next experiment and persisted the proposal.")

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("Latest Proposal")
        proposal = _safe_load_proposal(proposal_path)
        if proposal is None:
            st.info("No proposal found yet. Use 'Plan Next Experiment' first.")
        else:
            st.markdown(f"**Headline:** {proposal.headline}")
            st.write(proposal.summary)

            with st.expander("Primary Trial", expanded=True):
                st.markdown(f"**Trial ID:** `{proposal.primary_trial.trial_id}`")
                st.markdown(f"**Candidate:** `{proposal.primary_trial.candidate_id}`")
                st.markdown(f"**Goal:** {proposal.primary_trial.goal}")
                st.markdown("**Conditions**")
                for item in proposal.primary_trial.conditions:
                    st.markdown(f"- {item}")
                st.markdown("**Rationale**")
                for item in proposal.primary_trial.rationale:
                    st.markdown(f"- {item}")
                st.markdown("**Verification requirements**")
                for item in proposal.primary_trial.verification_requirements:
                    st.markdown(f"- {item}")

            with st.expander("Backup Trials"):
                if proposal.backup_trials:
                    for trial in proposal.backup_trials:
                        st.markdown(f"**{trial.trial_id}**: {trial.goal}")
                else:
                    st.caption("No backup trials recorded.")

            with st.expander("Ranked Candidates"):
                for candidate in proposal.ranked_candidates:
                    st.markdown(
                        f"**{candidate.candidate_id}** | realizability `{candidate.realizability_score:.2f}` | "
                        f"polymorph risk `{candidate.polymorph_risk:.2f}`"
                    )
                    st.caption(candidate.packing_hypothesis)

            with st.expander("Research Hints"):
                for hint in proposal.research_hints:
                    st.markdown(f"- {hint}")

            with st.expander("Verification Notes"):
                for note in proposal.verification_notes:
                    st.markdown(f"- {note}")

    with right:
        st.subheader("Record Experimental Result")
        if proposal is None:
            st.info("Plan a proposal before recording a result.")
        else:
            with st.form("record_result_form"):
                outcome_category = st.selectbox(
                    "Outcome category",
                    [
                        "target_like",
                        "competing_polymorph",
                        "solvate",
                        "amorphous",
                        "decomposition",
                        "inconclusive",
                    ],
                    index=2,
                )
                summary = st.text_area("Summary", height=100)
                evidence_text = st.text_area("Evidence (one line per item)", height=100)
                confidence = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

                st.markdown("**PXRD**")
                pxrd_status = st.selectbox(
                    "PXRD status",
                    ["", "target_match", "partial_match", "mismatch", "mixed", "weak_pattern", "not_collected"],
                )
                pxrd_note_text = st.text_area("PXRD notes", height=80)

                st.markdown("**Thermal**")
                thermal_status = st.selectbox(
                    "Thermal status",
                    ["", "consistent", "phase_change", "solvent_loss", "decomposition_signal", "not_collected"],
                )
                thermal_note_text = st.text_area("Thermal notes", height=80)
                mass_loss_percent = st.number_input("Mass loss percent", min_value=0.0, max_value=100.0, value=0.0)
                onset_temperature_c = st.number_input("Onset temperature (C)", min_value=0.0, max_value=500.0, value=0.0)

                st.markdown("**Single-crystal availability**")
                single_crystal_status = st.selectbox(
                    "Single-crystal status",
                    ["", "good_candidate", "microcrystalline", "powder_only", "none_observed", "not_checked"],
                )
                single_crystal_note_text = st.text_area("Single-crystal notes", height=80)

                submitted = st.form_submit_button("Record Result", use_container_width=True)
                if submitted:
                    updated = record_with_paths(
                        campaign_path=campaign_path,
                        proposal_path=proposal_path,
                        outcome_category=outcome_category,
                        summary=summary,
                        evidence=_split_lines(evidence_text),
                        confidence=confidence,
                        pxrd_status=pxrd_status or None,
                        pxrd_note=_split_lines(pxrd_note_text),
                        thermal_status=thermal_status or None,
                        thermal_note=_split_lines(thermal_note_text),
                        mass_loss_percent=mass_loss_percent if thermal_status else None,
                        onset_temperature_c=onset_temperature_c if thermal_status else None,
                        single_crystal_status=single_crystal_status or None,
                        single_crystal_note=_split_lines(single_crystal_note_text),
                    )
                    st.session_state["latest_campaign"] = updated.model_dump()
                    st.success("Recorded the result and updated the campaign memory.")

        st.subheader("Campaign Memory")
        memory = _safe_load_campaign(campaign_path)
        if memory is None:
            st.info("No campaign memory found yet.")
        else:
            st.markdown(f"**Campaign ID:** `{memory.campaign_id}`")
            st.markdown(f"**Completed trials:** {len(memory.completed_trials)}")
            if memory.learned_preferences:
                st.markdown("**Learned preferences**")
                for item in memory.learned_preferences:
                    st.markdown(f"- {item}")
            for index, record in enumerate(memory.completed_trials, start=1):
                with st.expander(f"Trial {index}: {record.trial.trial_id}"):
                    st.markdown(f"**Outcome:** `{record.result.outcome_category}`")
                    st.write(record.result.summary)
                    if record.result.evidence:
                        st.markdown("**Evidence**")
                        for item in record.result.evidence:
                            st.markdown(f"- {item}")
                    if record.failure_modes:
                        st.markdown("**Failure modes**")
                        for failure in record.failure_modes:
                            st.markdown(f"- `{failure.kind}`: {failure.rationale}")


def _safe_load_campaign(path: str):
    target = Path(path)
    if not target.exists():
        return None
    return load_campaign_memory(target)


def _safe_load_proposal(path: str):
    target = Path(path)
    if not target.exists():
        return None
    return load_proposal(target)


def _split_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


if __name__ == "__main__":
    main()
