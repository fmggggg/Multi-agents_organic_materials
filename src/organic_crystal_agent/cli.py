from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from organic_crystal_agent.campaign_summary import build_campaign_summary
from organic_crystal_agent.campaign_runner import CampaignRunner, RunnerConfig, summarize_campaign
from organic_crystal_agent.crystal_packing import extract_packing_features
from organic_crystal_agent.adsorption_workflow import (
    AdsorptionInput,
    SlurmAdsorptionWorkflow,
)
from organic_crystal_agent.expert_agents import ExpertSupervisor, ExpertSupervisorConfig
from organic_crystal_agent.landscape_vision import analyze_landscape_with_openai
from organic_crystal_agent.lammps_workflow import LAMMPSRelaxInput, LAMMPSRelaxSummary, SlurmLAMMPSWorkflow
from organic_crystal_agent.main_planner import MainPlanner, PlannerConfig
from organic_crystal_agent.memory_store import MemoryStore, resolve_campaign_root
from organic_crystal_agent.orchestrator import CampaignOrchestrator, load_campaign, save_campaign
from organic_crystal_agent.orca_workflow import (
    ORCAPropertyInput,
    ORCAPropertySummary,
    SlurmORCAWorkflow,
)
from organic_crystal_agent.packing_lammps import refresh_cluster_relaxation_batch, submit_cluster_relaxation_batch
from organic_crystal_agent.packing_orca import refresh_pair_interaction_batch, submit_pair_interaction_batch
from organic_crystal_agent.persistence import load_campaign_memory, load_proposal
from organic_crystal_agent.property_analysis import rank_campaign_property_candidates
from organic_crystal_agent.postprocess import export_spike_representative_structures, write_landscape_svg
from organic_crystal_agent.retrosynthesis import (
    ensure_buyable_stock_hdf5,
    extract_buyable_smiles,
    retrosynthesis_status,
    write_aizynthfinder_config,
)
from organic_crystal_agent.remote_workflow import (
    CrystalLandscapeSummary,
    RemoteWorkflowJob,
    RemoteWorkflowInput,
    SlurmCSPWorkflow,
    load_job_from_path,
    save_job_to_path,
)
from organic_crystal_agent.workflow import plan_with_paths, record_with_paths, run_campaign_step

app = typer.Typer(help="Organic crystal agent CLI.")
console = Console()


@app.command()
def run(request: str, objective: str | None = None) -> None:
    """Run one closed-loop campaign planning step without persistence."""
    proposal = run_campaign_step(request=request, objective=objective)
    console.print_json(json.dumps(proposal.model_dump(), indent=2))


@app.command()
def plan(
    request: str,
    objective: str | None = None,
    campaign_path: str = typer.Option("campaigns/default_campaign.json"),
    proposal_path: str = typer.Option("campaigns/latest_proposal.json"),
) -> None:
    """Plan the next experiment and persist campaign/proposal JSON."""
    proposal = plan_with_paths(
        request=request,
        objective=objective,
        campaign_path=campaign_path,
        proposal_path=proposal_path,
    )
    console.print_json(json.dumps(proposal.model_dump(), indent=2))


@app.command()
def record(
    outcome_category: str = typer.Option(..., help="Outcome category for the completed trial."),
    summary: str = typer.Option(..., help="Short experimental outcome summary."),
    campaign_path: str = typer.Option("campaigns/default_campaign.json"),
    proposal_path: str = typer.Option("campaigns/latest_proposal.json"),
    evidence: list[str] | None = typer.Option(None, help="Repeatable evidence entries."),
    confidence: float = typer.Option(0.7),
    failure_kind: list[str] | None = typer.Option(None, help="Optional explicit failure kinds."),
    pxrd_status: str | None = typer.Option(None, help="Structured PXRD status."),
    pxrd_note: list[str] | None = typer.Option(None, help="Repeatable PXRD notes."),
    thermal_status: str | None = typer.Option(None, help="Structured thermal-analysis status."),
    thermal_note: list[str] | None = typer.Option(None, help="Repeatable thermal notes."),
    mass_loss_percent: float | None = typer.Option(None, help="Observed mass loss percentage from TGA."),
    onset_temperature_c: float | None = typer.Option(None, help="Onset temperature in C for a key thermal event."),
    single_crystal_status: str | None = typer.Option(None, help="Single-crystal observation status."),
    single_crystal_note: list[str] | None = typer.Option(None, help="Repeatable single-crystal notes."),
) -> None:
    """Record the result of the latest proposed primary trial back into campaign memory."""
    updated = record_with_paths(
        campaign_path=campaign_path,
        proposal_path=proposal_path,
        outcome_category=outcome_category,
        summary=summary,
        evidence=evidence,
        confidence=confidence,
        failure_kinds=failure_kind,
        pxrd_status=pxrd_status,
        pxrd_note=pxrd_note,
        thermal_status=thermal_status,
        thermal_note=thermal_note,
        mass_loss_percent=mass_loss_percent,
        onset_temperature_c=onset_temperature_c,
        single_crystal_status=single_crystal_status,
        single_crystal_note=single_crystal_note,
    )
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command()
def show_campaign(
    campaign_path: str = typer.Option("campaigns/default_campaign.json"),
) -> None:
    """Show the persisted campaign memory."""
    memory = load_campaign_memory(campaign_path)
    console.print_json(json.dumps(memory.model_dump(), indent=2))


@app.command()
def show_proposal(
    proposal_path: str = typer.Option("campaigns/latest_proposal.json"),
) -> None:
    """Show the persisted latest proposal."""
    proposal = load_proposal(proposal_path)
    console.print_json(json.dumps(proposal.model_dump(), indent=2))


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port for the local Streamlit app."),
) -> None:
    """Launch the local visual interface."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter(
            "Streamlit is not installed. Install with `pip install -e .[ui]`."
        ) from exc

    app_path = Path(__file__).with_name("streamlit_app.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
        ],
        check=True,
    )


@app.command("csp-submit")
def csp_submit(
    molecule_path: str = typer.Option(..., help="Path to an input .xyz or .mol2 file."),
    molecule_label: str = typer.Option(..., help="Human-readable molecule label."),
    smiles: str | None = typer.Option(None, help="Optional canonical SMILES to persist with the job."),
    objective: str = typer.Option("Generate a CSP landscape for planning."),
    sample_count: int = typer.Option(1000, help="Total number of sampled crystal structures."),
    z_value: int | None = typer.Option(None, help="Target Z value for AlphaPack Z-aware sampling."),
    space_group: int | None = typer.Option(None, help="Optional fixed space group."),
    checkpoint_path: str = typer.Option("", help="Override AlphaPack checkpoint path."),
    use_real_sg_ratio: bool = typer.Option(False, help="Allocate samples by empirical SG ratio."),
    jobs_root: str | None = typer.Option(None, help="Optional working root for submitted CSP job directories."),
    save_job_path: str | None = typer.Option(None, help="Optional path to persist submitted job metadata."),
) -> None:
    """Submit an AlphaPack + BOMLIP CSP workflow to Slurm."""
    workflow = SlurmCSPWorkflow()
    if jobs_root:
        workflow = SlurmCSPWorkflow(
            workflow.settings.model_copy(update={"jobs_root": jobs_root})
        )
    job = workflow.submit(
        RemoteWorkflowInput(
            molecule_label=molecule_label,
            molecule_path=molecule_path,
            smiles=smiles,
            objective=objective,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
        )
    )
    if save_job_path:
        save_job_to_path(save_job_path, job)
    console.print_json(json.dumps(job.model_dump(), indent=2))


@app.command("csp-preview")
def csp_preview(
    molecule_path: str = typer.Option(..., help="Path to an input .xyz or .mol2 file."),
    molecule_label: str = typer.Option(..., help="Human-readable molecule label."),
    smiles: str | None = typer.Option(None, help="Optional canonical SMILES to persist with the preview."),
    objective: str = typer.Option("Generate a CSP landscape for planning."),
    sample_count: int = typer.Option(1000, help="Total number of sampled crystal structures."),
    z_value: int | None = typer.Option(None, help="Target Z value for AlphaPack Z-aware sampling."),
    space_group: int | None = typer.Option(None, help="Optional fixed space group."),
    checkpoint_path: str = typer.Option("", help="Override AlphaPack checkpoint path."),
    use_real_sg_ratio: bool = typer.Option(False, help="Allocate samples by empirical SG ratio."),
    jobs_root: str | None = typer.Option(None, help="Optional working root for preview CSP job directories."),
) -> None:
    """Prepare a CSP job directory and print the generated Slurm script without submitting."""
    workflow = SlurmCSPWorkflow()
    if jobs_root:
        workflow = SlurmCSPWorkflow(
            workflow.settings.model_copy(update={"jobs_root": jobs_root})
        )
    job = workflow.prepare(
        RemoteWorkflowInput(
            molecule_label=molecule_label,
            molecule_path=molecule_path,
            smiles=smiles,
            objective=objective,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
        )
    )
    script_path = Path(job.working_directory) / "submit_slurm.sh"
    payload = {
        "job": job.model_dump(),
        "script_path": str(script_path),
        "script": script_path.read_text(encoding="utf-8"),
    }
    console.print_json(json.dumps(payload, indent=2))


@app.command("csp-status")
def csp_status(
    job_path: str = typer.Option(..., help="Path to a saved job metadata JSON file."),
    save: bool = typer.Option(True, help="Write the refreshed status back to the same JSON file."),
) -> None:
    """Refresh Slurm status for a previously submitted CSP job."""
    workflow = SlurmCSPWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.get_status(job)
    if save:
        save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("csp-requeue")
def csp_requeue(
    job_path: str = typer.Option(..., help="Path to a saved CSP job metadata JSON file."),
) -> None:
    """Requeue a resume-aware CSP job using the existing working directory."""
    workflow = SlurmCSPWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.requeue(job)
    save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("csp-fetch")
def csp_fetch(
    job_path: str | None = typer.Option(None, help="Path to a saved job metadata JSON file."),
    job_dir: str | None = typer.Option(None, help="Working directory for a submitted job."),
) -> None:
    """Summarize CSP landscape outputs from a submitted job directory."""
    workflow = SlurmCSPWorkflow()
    if job_path:
        job = load_job_from_path(job_path)
    elif job_dir:
        job = RemoteWorkflowJob(
            workflow_name="alphapack-bomlip-slurm",
            job_id="manual",
            status="succeeded",
            server_label=workflow.settings.server_label,
            working_directory=job_dir,
        )
    else:
        raise typer.BadParameter("Either --job-path or --job-dir is required.")

    summary: CrystalLandscapeSummary = workflow.fetch_results(job)
    console.print_json(json.dumps(summary.model_dump(), indent=2))


@app.command("landscape-plot")
def landscape_plot(
    csv_path: str = typer.Option(..., help="CSV containing density and energy_kj columns."),
    output_path: str = typer.Option(..., help="Path to write the SVG plot."),
    title: str = typer.Option("Crystal Landscape", help="Plot title."),
) -> None:
    """Generate a density vs relative-energy landscape SVG."""
    point_count = write_landscape_svg(csv_path, output_path, title=title)
    payload = {
        "csv_path": str(Path(csv_path).expanduser().resolve()),
        "output_path": str(Path(output_path).expanduser().resolve()),
        "point_count": point_count,
    }
    console.print_json(json.dumps(payload, indent=2))


@app.command("landscape-export-spikes")
def landscape_export_spikes(
    job_dir: str = typer.Option(..., help="CSP job directory containing optimized landscape CSV and relax artifacts."),
    max_relative_energy_kj: float = typer.Option(100.0, help="Relative-energy window used to identify spike representatives."),
    density_cluster_window: float = typer.Option(0.10, help="Density clustering window for spike representatives."),
    output_dir_name: str = typer.Option("spikes", help="Folder name to create inside the CSP job directory."),
) -> None:
    """Export representative spike CIFs into a stable CSP-job subfolder."""
    manifest = export_spike_representative_structures(
        job_dir,
        max_relative_energy_kj=max_relative_energy_kj,
        density_cluster_window=density_cluster_window,
        output_dir_name=output_dir_name,
    )
    console.print_json(json.dumps(manifest, indent=2))


@app.command("landscape-vision")
def landscape_vision(
    png_path: str = typer.Option(..., help="PNG landscape image to send as visual input."),
    csv_path: str = typer.Option(..., help="CSV containing density and energy_kj columns."),
    cif_dir: str | None = typer.Option(None, help="Optional directory containing named CIF files for candidate points."),
    molecule_label: str | None = typer.Option(None, help="Optional molecule/job label for prompt context."),
    model: str | None = typer.Option(None, help="Optional OpenAI model override."),
    max_relative_energy_kj: float = typer.Option(100.0, help="Relative-energy window used to preselect candidate points."),
    max_chars_per_cif: int = typer.Option(4000, help="Max CIF characters to include per candidate structure."),
    output_path: str | None = typer.Option(None, help="Optional JSON path for saving the analysis result."),
) -> None:
    """Run explicit image-based landscape analysis with OpenAI vision."""
    report = analyze_landscape_with_openai(
        png_path=png_path,
        csv_path=csv_path,
        cif_dir=cif_dir,
        molecule_label=molecule_label,
        model=model,
        max_relative_energy_kj=max_relative_energy_kj,
        max_chars_per_cif=max_chars_per_cif,
    )
    payload = report.model_dump()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else Path(png_path).expanduser().resolve().with_name("landscape_vision_report.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["saved_report_path"] = str(target)
    console.print_json(json.dumps(payload, indent=2))


@app.command("orca-submit")
def orca_submit(
    structure_path: str = typer.Option(..., help="Path to an input .xyz structure file."),
    molecule_label: str = typer.Option(..., help="Human-readable molecule label."),
    objective: str = typer.Option("Estimate molecular electronic properties."),
    property_profile: str = typer.Option("frontier", help="One of: frontier, hole_reorg."),
    calculation_mode: str = typer.Option("sp", help="One of: sp, opt, freq, optfreq."),
    charge: int = typer.Option(0),
    multiplicity: int = typer.Option(1),
    fragment_a_path: str = typer.Option("", help="Required for pair_interaction profile."),
    fragment_b_path: str = typer.Option("", help="Required for pair_interaction profile."),
    method: str = typer.Option("B3LYP"),
    basis: str = typer.Option("def2-SVP"),
    save_job_path: str | None = typer.Option(None, help="Optional path to persist submitted job metadata."),
) -> None:
    """Submit an ORCA property workflow to Slurm."""
    workflow = SlurmORCAWorkflow()
    job = workflow.submit(
        ORCAPropertyInput(
            molecule_label=molecule_label,
            structure_path=structure_path,
            objective=objective,
            property_profile=property_profile,  # type: ignore[arg-type]
            calculation_mode=calculation_mode,  # type: ignore[arg-type]
            charge=charge,
            multiplicity=multiplicity,
            fragment_a_path=fragment_a_path,
            fragment_b_path=fragment_b_path,
            method=method,
            basis=basis,
        )
    )
    if save_job_path:
        save_job_to_path(save_job_path, job)
    console.print_json(json.dumps(job.model_dump(), indent=2))


@app.command("adsorb-submit")
def adsorb_submit(
    structure_path: str = typer.Option(..., help="Path to a single CIF structure file."),
    molecule_label: str = typer.Option(..., help="Human-readable molecule label."),
    objective: str = typer.Option("Estimate adsorption capacity."),
    property_profile: str = typer.Option(
        "humid_co2_capture",
        help="Adsorption profile, e.g. humid_co2_capture or methane_adsorption.",
    ),
    source_csp_workdir: str = typer.Option("", help="Optional source CSP working directory."),
    source_csp_job_id: str = typer.Option("", help="Optional source CSP job id."),
    save_job_path: str | None = typer.Option(None, help="Optional path to persist submitted job metadata."),
) -> None:
    """Submit an adsorption workflow to Slurm via RASPA2."""
    workflow = SlurmAdsorptionWorkflow()
    job = workflow.submit(
        AdsorptionInput(
            molecule_label=molecule_label,
            structure_path=structure_path,
            source_csp_workdir=source_csp_workdir,
            source_csp_job_id=source_csp_job_id,
            objective=objective,
            property_profile=property_profile,
        )
    )
    if save_job_path:
        save_job_to_path(save_job_path, job)
    console.print_json(json.dumps(job.model_dump(), indent=2))


@app.command("adsorb-status")
def adsorb_status(
    job_path: str = typer.Option(..., help="Path to a saved adsorption job metadata JSON file."),
    save: bool = typer.Option(True, help="Write the refreshed status back to the same JSON file."),
) -> None:
    """Refresh Slurm status for a previously submitted adsorption job."""
    workflow = SlurmAdsorptionWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.get_status(job)
    if save:
        save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("adsorb-fetch")
def adsorb_fetch(
    job_path: str | None = typer.Option(None, help="Path to a saved adsorption job metadata JSON file."),
    job_dir: str | None = typer.Option(None, help="Working directory for a submitted adsorption job."),
) -> None:
    """Summarize adsorption outputs from a submitted job directory."""
    workflow = SlurmAdsorptionWorkflow()
    if job_path:
        job = load_job_from_path(job_path)
    elif job_dir:
        job = RemoteWorkflowJob(
            workflow_name="raspa-adsorption-slurm",
            job_id="manual",
            status="succeeded",
            server_label=workflow.settings.server_label,
            working_directory=job_dir,
        )
    else:
        raise typer.BadParameter("Either --job-path or --job-dir is required.")

    summary = workflow.fetch_results(job)
    console.print_json(json.dumps(summary.model_dump(), indent=2))


@app.command("orca-status")
def orca_status(
    job_path: str = typer.Option(..., help="Path to a saved ORCA job metadata JSON file."),
    save: bool = typer.Option(True, help="Write the refreshed status back to the same JSON file."),
) -> None:
    """Refresh Slurm status for a previously submitted ORCA job."""
    workflow = SlurmORCAWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.get_status(job)
    if save:
        save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("orca-requeue")
def orca_requeue(
    job_path: str = typer.Option(..., help="Path to a saved ORCA job metadata JSON file."),
) -> None:
    """Requeue a resume-aware ORCA job using the existing working directory."""
    workflow = SlurmORCAWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.requeue(job)
    save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("orca-fetch")
def orca_fetch(
    job_path: str | None = typer.Option(None, help="Path to a saved ORCA job metadata JSON file."),
    job_dir: str | None = typer.Option(None, help="Working directory for a submitted ORCA job."),
) -> None:
    """Summarize ORCA property outputs from a submitted job directory."""
    workflow = SlurmORCAWorkflow()
    if job_path:
        job = load_job_from_path(job_path)
    elif job_dir:
        job = RemoteWorkflowJob(
            workflow_name="orca-slurm",
            job_id="manual",
            status="succeeded",
            server_label=workflow.settings.server_label,
            working_directory=job_dir,
        )
    else:
        raise typer.BadParameter("Either --job-path or --job-dir is required.")

    summary: ORCAPropertySummary = workflow.fetch_results(job)
    console.print_json(json.dumps(summary.model_dump(), indent=2))


@app.command("campaign-init")
def campaign_init(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    campaign_id: str = typer.Option(..., help="Stable campaign identifier."),
    target_description: str = typer.Option(..., help="High-level target such as porous crystal discovery."),
    desired_property: list[str] | None = typer.Option(None, help="Repeatable desired-property tags."),
    note: list[str] | None = typer.Option(None, help="Optional campaign notes."),
) -> None:
    """Create a long-running discovery campaign state file."""
    orchestrator = CampaignOrchestrator()
    campaign = orchestrator.create_campaign(
        campaign_id=campaign_id,
        target_description=target_description,
        desired_properties=desired_property or [],
        notes=note or [],
    )
    save_campaign(campaign_path, campaign)
    console.print_json(json.dumps(campaign.model_dump(), indent=2))


@app.command("retrosyn-status")
def retrosyn_status_cmd() -> None:
    """Show the current AiZynthFinder/buyable-stock configuration status."""
    payload = retrosynthesis_status().model_dump()
    console.print_json(json.dumps(payload, indent=2))


@app.command("retrosyn-prepare-stock")
def retrosyn_prepare_stock(
    buyables_jsonl: str = typer.Option(..., help="JSONL containing canonical buyable SMILES records."),
    output_smiles: str = typer.Option(..., help="Output newline-delimited SMILES file."),
    output_stock_hdf5: str = typer.Option(..., help="Output AiZynthFinder stock HDF5 path."),
    output_config: str = typer.Option(..., help="Output AiZynthFinder YAML config path."),
    max_steps: int = typer.Option(4, help="Maximum allowed retrosynthesis depth in the generated config."),
    iteration_limit: int = typer.Option(150, help="MCTS iteration limit written into the generated config."),
    time_limit: int = typer.Option(120, help="Search time limit written into the generated config."),
) -> None:
    """Convert buyable JSONL into AiZynthFinder stock inputs and write a config template."""
    smiles_summary = extract_buyable_smiles(buyables_jsonl, output_smiles)
    stock_summary = None
    stock_error = None
    try:
        stock_summary = ensure_buyable_stock_hdf5(
            smiles_path=output_smiles,
            stock_hdf5=output_stock_hdf5,
        )
    except Exception as exc:  # pragma: no cover - depends on local CLI installation
        stock_error = str(exc)
    config_path = write_aizynthfinder_config(
        output_config,
        stock_hdf5=output_stock_hdf5,
        max_transforms=max_steps,
        iteration_limit=iteration_limit,
        time_limit=time_limit,
    )
    payload = {
        "smiles_summary": smiles_summary,
        "stock_summary": stock_summary,
        "stock_error": stock_error,
        "config_path": str(config_path),
        "status": retrosynthesis_status().model_dump(),
    }
    console.print_json(json.dumps(payload, indent=2))


@app.command("campaign-add-molecule")
def campaign_add_molecule(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_label: str = typer.Option(...),
    rationale: str = typer.Option(...),
    molecule_path: str = typer.Option("", help="Path to .xyz/.mol2 input for CSP."),
    smiles: str | None = typer.Option(None),
    source_agent: str = typer.Option("molecule-design-agent"),
    design_family: str = typer.Option(""),
    evidence: list[str] | None = typer.Option(None, help="Repeatable design evidence entries."),
    target_property: list[str] | None = typer.Option(None, help="Repeatable property tags."),
    note: list[str] | None = typer.Option(None, help="Optional notes."),
) -> None:
    """Register a candidate molecule in the campaign state."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.add_molecule(
        campaign=campaign,
        molecule_label=molecule_label,
        rationale=rationale,
        molecule_path=molecule_path,
        smiles=smiles,
        source_agent=source_agent,
        design_family=design_family,
        evidence=evidence or [],
        target_properties=target_property or [],
        notes=note or [],
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-update-molecule")
def campaign_update_molecule(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_id: str = typer.Option(...),
    molecule_path: str | None = typer.Option(None),
    smiles: str | None = typer.Option(None),
    rationale: str | None = typer.Option(None),
    design_family: str | None = typer.Option(None),
    evidence: list[str] | None = typer.Option(None, help="Optional design evidence to append."),
    note: list[str] | None = typer.Option(None, help="Optional notes to append."),
) -> None:
    """Patch a registered molecule record, e.g. to attach a file path before CSP submission."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.update_molecule(
        campaign=campaign,
        molecule_id=molecule_id,
        molecule_path=molecule_path,
        smiles=smiles,
        rationale=rationale,
        design_family=design_family,
        evidence=evidence or [],
        notes=note or [],
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-add-literature")
def campaign_add_literature(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    title: str = typer.Option(...),
    summary: str = typer.Option(...),
    evidence: list[str] | None = typer.Option(None, help="Repeatable evidence snippets."),
    confidence: float = typer.Option(0.5),
) -> None:
    """Attach a literature-derived hypothesis to the campaign state."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.add_literature_hypothesis(
        campaign=campaign,
        title=title,
        summary=summary,
        evidence=evidence or [],
        confidence=confidence,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-auto-literature")
def campaign_auto_literature(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    max_hypotheses: int = typer.Option(5, help="Maximum number of hypotheses to add."),
) -> None:
    """Generate literature-style hypotheses from the configured research backend."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.auto_add_literature_hypotheses(
        campaign=campaign,
        max_hypotheses=max_hypotheses,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-submit-csp")
def campaign_submit_csp(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_id: str = typer.Option(..., help="Registered molecule id, e.g. mol-001."),
    sample_count: int = typer.Option(1000),
    z_value: int | None = typer.Option(None),
    space_group: int | None = typer.Option(None),
    checkpoint_path: str = typer.Option(""),
    use_real_sg_ratio: bool = typer.Option(False),
    objective: str | None = typer.Option(None),
) -> None:
    """Submit a registered molecule into the Slurm CSP workflow."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.submit_molecule_for_csp(
        campaign=campaign,
        molecule_id=molecule_id,
        sample_count=sample_count,
        z_value=z_value,
        space_group=space_group,
        checkpoint_path=checkpoint_path,
        use_real_sg_ratio=use_real_sg_ratio,
        objective=objective,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-submit-orca")
def campaign_submit_orca(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_id: str = typer.Option(..., help="Registered molecule id, e.g. mol-001."),
    property_profile: str = typer.Option("frontier"),
    calculation_mode: str = typer.Option("sp"),
    charge: int = typer.Option(0),
    multiplicity: int = typer.Option(1),
    method: str = typer.Option("B3LYP"),
    basis: str = typer.Option("def2-SVP"),
    objective: str | None = typer.Option(None),
) -> None:
    """Submit an ORCA property evaluation for a registered molecule."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.submit_molecule_for_property_eval(
        campaign=campaign,
        molecule_id=molecule_id,
        property_profile=property_profile,
        calculation_mode=calculation_mode,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        objective=objective,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-submit-adsorption")
def campaign_submit_adsorption(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_id: str = typer.Option(..., help="Registered molecule id, e.g. mol-001."),
    property_profile: str = typer.Option(
        "humid_co2_capture",
        help="Adsorption profile, e.g. humid_co2_capture or methane_adsorption.",
    ),
    objective: str | None = typer.Option(None),
) -> None:
    """Submit adsorption evaluation for a registered molecule using a spike-representative CIF."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.submit_molecule_for_property_eval(
        campaign=campaign,
        molecule_id=molecule_id,
        property_profile=property_profile,
        calculation_mode="single_point",
        method="",
        basis="",
        objective=objective,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))

@app.command("campaign-property-report")
def campaign_property_report(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Show property-ranked molecules from current ORCA results."""
    campaign = load_campaign(campaign_path)
    console.print_json(json.dumps(rank_campaign_property_candidates(campaign), indent=2))


@app.command("campaign-submit-packing")
def campaign_submit_packing(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_id: str = typer.Option(..., help="Registered molecule id, e.g. mol-001."),
    max_dimers: int = typer.Option(12),
    cluster_size: int = typer.Option(4),
    pair_top_n: int = typer.Option(5),
    cluster_top_n: int = typer.Option(5),
    pair_method: str = typer.Option("B3LYP"),
    pair_basis: str = typer.Option("def2-SVP"),
    cluster_max_iter: int = typer.Option(2000),
) -> None:
    """Extract packing and submit ORCA pair + LAMMPS cluster batches for a molecule."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.submit_molecule_for_packing_analysis(
        campaign=campaign,
        molecule_id=molecule_id,
        max_dimers=max_dimers,
        cluster_size=cluster_size,
        pair_top_n=pair_top_n,
        cluster_top_n=cluster_top_n,
        pair_method=pair_method,
        pair_basis=pair_basis,
        cluster_max_iter=cluster_max_iter,
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-packing-report")
def campaign_packing_report(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Show packing-ranked molecules from current ORCA pair and LAMMPS cluster results."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    console.print_json(json.dumps(orchestrator.packing_ranking(campaign), indent=2))


@app.command("packing-extract")
def packing_extract(
    cif_path: str = typer.Option(..., help="Packed or bonded CIF/MMCIF file."),
    out_dir: str = typer.Option(..., help="Directory to write monomer/dimer/cluster XYZ files."),
    max_dimers: int = typer.Option(12, help="Maximum number of dimer pairs to keep."),
    cluster_size: int = typer.Option(4, help="Number of molecules in each local cluster."),
) -> None:
    """Extract monomers, nearest-neighbor dimers, and local clusters from a packed crystal file."""
    summary = extract_packing_features(
        cif_path=cif_path,
        out_dir=out_dir,
        max_dimers=max_dimers,
        cluster_size=cluster_size,
    )
    console.print_json(summary.model_dump_json(indent=2))


@app.command("packing-submit-pairs")
def packing_submit_pairs(
    packing_summary_path: str = typer.Option(..., help="Path to packing_summary.json from packing-extract."),
    top_n: int = typer.Option(5, help="Number of top dimers by transport proxy to submit."),
    method: str = typer.Option("B3LYP"),
    basis: str = typer.Option("def2-SVP"),
) -> None:
    """Submit ORCA pair-interaction jobs for the top dimers in a packing summary."""
    summary = submit_pair_interaction_batch(
        packing_summary_path=packing_summary_path,
        top_n=top_n,
        method=method,
        basis=basis,
    )
    console.print_json(summary.model_dump_json(indent=2))


@app.command("packing-refresh-pairs")
def packing_refresh_pairs(
    batch_manifest_path: str = typer.Option(..., help="Path to pair_batch_manifest.json."),
) -> None:
    """Refresh and summarize a submitted ORCA pair-interaction batch."""
    summary = refresh_pair_interaction_batch(batch_manifest_path=batch_manifest_path)
    console.print_json(summary.model_dump_json(indent=2))


@app.command("packing-submit-clusters")
def packing_submit_clusters(
    packing_summary_path: str = typer.Option(..., help="Path to packing_summary.json from packing-extract."),
    top_n: int = typer.Option(5, help="Number of clusters to submit."),
    max_iter: int = typer.Option(2000),
) -> None:
    """Submit LAMMPS cluster-relaxation jobs for extracted local clusters."""
    summary = submit_cluster_relaxation_batch(
        packing_summary_path=packing_summary_path,
        top_n=top_n,
        max_iter=max_iter,
    )
    console.print_json(summary.model_dump_json(indent=2))


@app.command("packing-refresh-clusters")
def packing_refresh_clusters(
    batch_manifest_path: str = typer.Option(..., help="Path to cluster_batch_manifest.json."),
) -> None:
    """Refresh and summarize a submitted LAMMPS cluster-relaxation batch."""
    summary = refresh_cluster_relaxation_batch(batch_manifest_path=batch_manifest_path)
    console.print_json(summary.model_dump_json(indent=2))


@app.command("lammps-submit")
def lammps_submit(
    structure_path: str = typer.Option(..., help="Path to a monomer/dimer/cluster xyz file."),
    system_label: str = typer.Option(...),
    objective: str = typer.Option("Run a generic LJ relaxation proxy."),
    max_iter: int = typer.Option(2000),
    save_job_path: str | None = typer.Option(None),
) -> None:
    """Submit a LAMMPS generic LJ proxy relaxation on Slurm."""
    workflow = SlurmLAMMPSWorkflow()
    job = workflow.submit(
        LAMMPSRelaxInput(
            structure_path=structure_path,
            system_label=system_label,
            objective=objective,
            max_iter=max_iter,
        )
    )
    if save_job_path:
        save_job_to_path(save_job_path, job)
    console.print_json(json.dumps(job.model_dump(), indent=2))


@app.command("lammps-status")
def lammps_status(
    job_path: str = typer.Option(..., help="Path to a saved LAMMPS job metadata JSON file."),
    save: bool = typer.Option(True),
) -> None:
    """Refresh Slurm status for a previously submitted LAMMPS job."""
    workflow = SlurmLAMMPSWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.get_status(job)
    if save:
        save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("lammps-requeue")
def lammps_requeue(
    job_path: str = typer.Option(..., help="Path to a saved LAMMPS job metadata JSON file."),
) -> None:
    """Requeue a resume-aware LAMMPS job using the existing working directory."""
    workflow = SlurmLAMMPSWorkflow()
    job = load_job_from_path(job_path)
    updated = workflow.requeue(job)
    save_job_to_path(job_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("lammps-fetch")
def lammps_fetch(
    job_path: str | None = typer.Option(None, help="Path to a saved LAMMPS job metadata JSON file."),
    job_dir: str | None = typer.Option(None, help="Working directory for a submitted LAMMPS job."),
) -> None:
    """Summarize LAMMPS proxy relaxation outputs from a submitted job directory."""
    workflow = SlurmLAMMPSWorkflow()
    if job_path:
        job = load_job_from_path(job_path)
    elif job_dir:
        job = RemoteWorkflowJob(
            workflow_name="lammps-slurm",
            job_id="manual",
            status="succeeded",
            server_label=workflow.settings.server_label,
            working_directory=job_dir,
        )
    else:
        raise typer.BadParameter("Either --job-path or --job-dir is required.")
    summary: LAMMPSRelaxSummary = workflow.fetch_results(job)
    console.print_json(json.dumps(summary.model_dump(), indent=2))


@app.command("campaign-auto-molecules")
def campaign_auto_molecules(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    max_molecules: int = typer.Option(5, help="Maximum number of auto-generated molecules."),
) -> None:
    """Generate a batch of candidate molecules from the planning agent."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.auto_add_molecule_batch(
        campaign=campaign,
        max_molecules=max_molecules,
    )
    updated = orchestrator.materialize_molecule_inputs(
        updated,
        campaign_root=resolve_campaign_root(campaign_path, updated.campaign_id),
    )
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-refresh")
def campaign_refresh(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Refresh all CSP jobs in a campaign and ingest completed landscapes."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.refresh_campaign(campaign)
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-watch")
def campaign_watch(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    poll_seconds: int = typer.Option(300, help="Polling interval in seconds."),
    max_polls: int = typer.Option(0, help="0 means unlimited until jobs finish."),
) -> None:
    """Watch a campaign, polling Slurm until all active jobs finish or max polls is reached."""
    orchestrator = CampaignOrchestrator()
    updated = orchestrator.watch_campaign(
        campaign_path=campaign_path,
        poll_seconds=poll_seconds,
        max_polls=max_polls,
    )
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-decide")
def campaign_decide(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Run the planning agent over completed landscapes and record a round decision."""
    orchestrator = CampaignOrchestrator()
    campaign = load_campaign(campaign_path)
    updated = orchestrator.auto_decide_next_round(campaign)
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("campaign-report")
def campaign_report(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Show a compact summary of campaign counts and state."""
    campaign = load_campaign(campaign_path)
    console.print_json(json.dumps(summarize_campaign(campaign), indent=2))


@app.command("campaign-write-memory")
def campaign_write_memory(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Materialize workspace memory files for a campaign."""
    campaign = load_campaign(campaign_path)
    store = MemoryStore(resolve_campaign_root(campaign_path, campaign.campaign_id))
    store.write_campaign_memory(campaign)
    console.print_json(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "memory_root": str(store.root),
                "memory_files": sorted(path.name for path in store.memory_dir.glob("*.md")),
            },
            indent=2,
        )
    )


@app.command("campaign-run-once")
def campaign_run_once(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    literature_hypotheses: int = typer.Option(5),
    molecule_batch_size: int = typer.Option(5),
    sample_count: int = typer.Option(1000),
    z_value: int | None = typer.Option(None),
    space_group: int | None = typer.Option(None),
    checkpoint_path: str = typer.Option(""),
    use_real_sg_ratio: bool = typer.Option(False),
    auto_decide: bool = typer.Option(True),
    auto_expand: bool = typer.Option(True),
) -> None:
    """Advance the campaign by one orchestration step."""
    runner = CampaignRunner(
        config=RunnerConfig(
            literature_hypotheses=literature_hypotheses,
            molecule_batch_size=molecule_batch_size,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
            auto_decide=auto_decide,
            auto_expand=auto_expand,
        )
    )
    campaign = load_campaign(campaign_path)
    updated = runner.run_once(campaign)
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("planner-step")
def planner_step(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    literature_hypotheses: int = typer.Option(5),
    molecule_batch_size: int = typer.Option(5),
    sample_count: int = typer.Option(1000),
    z_value: int | None = typer.Option(None),
    space_group: int | None = typer.Option(None),
    checkpoint_path: str = typer.Option(""),
    use_real_sg_ratio: bool = typer.Option(False),
    auto_decide: bool = typer.Option(True),
    auto_expand: bool = typer.Option(True),
) -> None:
    """Run one main-planner step and refresh workspace memory."""
    planner = MainPlanner(
        config=PlannerConfig(
            literature_hypotheses=literature_hypotheses,
            molecule_batch_size=molecule_batch_size,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
            auto_decide=auto_decide,
            auto_expand=auto_expand,
        )
    )
    campaign = load_campaign(campaign_path)
    updated = planner.step(campaign=campaign, campaign_path=campaign_path)
    save_campaign(campaign_path, updated)
    console.print_json(json.dumps(updated.model_dump(), indent=2))


@app.command("planner-report")
def planner_report(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Show the latest planner trace and memory root for a campaign."""
    campaign = load_campaign(campaign_path)
    memory_root = str(resolve_campaign_root(campaign_path, campaign.campaign_id))
    latest_trace = campaign.planner_traces[-1].model_dump() if campaign.planner_traces else None
    console.print_json(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "memory_root": memory_root,
                "planner_trace_count": len(campaign.planner_traces),
                "latest_planner_trace": latest_trace,
            },
            indent=2,
        )
    )


@app.command("expert-step")
def expert_step(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    molecule_batch_size: int = typer.Option(5),
    max_total_molecules: int = typer.Option(100),
    sample_count: int = typer.Option(1000),
    z_value: int | None = typer.Option(None),
    space_group: int | None = typer.Option(None),
    checkpoint_path: str = typer.Option(""),
    use_real_sg_ratio: bool = typer.Option(False),
    auto_decide: bool = typer.Option(True),
    auto_expand: bool = typer.Option(True),
) -> None:
    """Run one crash-isolated specialist-agent supervisor step."""
    supervisor = ExpertSupervisor(
        config=ExpertSupervisorConfig(
            molecule_batch_size=molecule_batch_size,
            max_total_molecules=max_total_molecules,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
            auto_decide=auto_decide,
            auto_expand=auto_expand,
        )
    )
    campaign = load_campaign(campaign_path)
    updated, report = supervisor.step(campaign, campaign_path=campaign_path)
    save_campaign(campaign_path, updated)
    console.print_json(
        json.dumps(
            {
                "campaign": updated.model_dump(),
                "expert_report": report.model_dump(),
            },
            indent=2,
        )
    )


@app.command("campaign-summary")
def campaign_summary_cmd(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
) -> None:
    """Show the structured campaign summary used by the main planner."""
    campaign = load_campaign(campaign_path)
    summary = build_campaign_summary(campaign)
    console.print_json(summary.model_dump_json(indent=2))


@app.command("campaign-run-loop")
def campaign_run_loop(
    campaign_path: str = typer.Option(..., help="Path to the campaign JSON file."),
    poll_seconds: int = typer.Option(300),
    max_cycles: int = typer.Option(0),
    literature_hypotheses: int = typer.Option(5),
    molecule_batch_size: int = typer.Option(5),
    sample_count: int = typer.Option(1000),
    z_value: int | None = typer.Option(None),
    space_group: int | None = typer.Option(None),
    checkpoint_path: str = typer.Option(""),
    use_real_sg_ratio: bool = typer.Option(False),
    auto_decide: bool = typer.Option(True),
    auto_expand: bool = typer.Option(True),
) -> None:
    """Run the campaign loop until no active jobs remain or max cycles is reached."""
    runner = CampaignRunner(
        config=RunnerConfig(
            literature_hypotheses=literature_hypotheses,
            molecule_batch_size=molecule_batch_size,
            sample_count=sample_count,
            z_value=z_value,
            space_group=space_group,
            checkpoint_path=checkpoint_path,
            use_real_sg_ratio=use_real_sg_ratio,
            auto_decide=auto_decide,
            auto_expand=auto_expand,
        )
    )
    updated = runner.run_loop(
        campaign_path=campaign_path,
        poll_seconds=poll_seconds,
        max_cycles=max_cycles,
    )
    console.print_json(json.dumps(updated.model_dump(), indent=2))


if __name__ == "__main__":
    app()
