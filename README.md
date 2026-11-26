# Organic Crystal Agent

`Organic Crystal Agent` is a ChemGraph-inspired platform for organic crystal discovery, CSP orchestration, packing extraction, and property evaluation.

> **Projects in this repository**
>
> - [`projects/azapentacene_mobility`](projects/azapentacene_mobility): LLM agent loop (OpenAI or free OpenRouter models)
>   that searches aza-substituted five-ring acenes for high predicted charge mobility, using Clari-H CSP, MACE-OFF
>   energies and the MCRT mobility model.

It is intentionally built around the parts that matter most for this project:

- a typed workflow engine
- a structured tool registry
- an organic-crystal-specific campaign schema
- a memory-aware reasoning policy
- a verification step before next-trial proposals are emitted

## Why this base

After reviewing recent chemistry and materials agent platforms, the best starting point for this repo is:

- **ChemGraph-style workflow orchestration** as the platform backbone
- **Eunomia-style verification and knowledge retrieval** for the knowledge layer
- **ChemToolAgent/ChemMCP-style tool abstraction** for extensibility

This is a better fit than a HOF-only stack because the core problem is broader:

**selecting molecules, crystal forms, and crystallization conditions under real experimental constraints**

## Initial platform scope

This repo does not try to solve full organic crystal discovery on day one.

The first milestone is a platform that can represent and execute the loop:

1. interpret an experimental goal
2. retrieve an external research brief
3. enumerate candidate crystal forms
4. score realizability using campaign memory
5. propose the single next experiment plus backups
6. verify that the proposal is evidence-seeking

The current implementation is rule-based and deterministic by design. That keeps the platform testable while the interfaces for future LLM planners, CSP engines, external research services, and experiment logs are stabilized.

## Layout

```text
src/organic_crystal_agent/
  cli.py
  config.py
  policy.py
  research.py
  schemas.py
  state.py
  workflow.py
  tools/
    base.py
    organic_crystal.py
tests/
  test_workflow.py
```

## Current capabilities

- output a `NextExperimentProposal`
- represent `CampaignMemory`, `CrystallizationTrial`, `CharacterizationResult`, and `FailureMode`
- update memory after a trial result is recorded
- bias the next recommendation using prior failure modes
- switch between `mock` and `openai` research backends
- run AlphaPack + BOMLIP CSP jobs on Slurm
- postprocess CSP outputs with CIF merge, archive, and AMD-based deduplication
- maintain campaign memory, planner traces, and structured campaign summaries
- extract monomers, dimers, and local clusters from packed/bonded CIF files
- submit ORCA property jobs for frontier, hole reorganization, and dimer pair interaction profiles
- batch-submit ORCA pair-interaction jobs from extracted packing dimers
- submit LAMMPS generic LJ relaxation proxy jobs for monomer/dimer/cluster xyz files
- requeue resume-aware CSP, ORCA, and LAMMPS jobs from the same working directory
- batch-submit LAMMPS local-cluster relaxations from extracted packing clusters
- attach packing evaluations back to campaign state and report packing-ranked molecules
- apply application-specific design priors plus inferred specs for unfamiliar applications
- quality-gate molecule proposals before they enter expensive CSP or property workflows

## Reusable AlphaPack benchmark workflow

The humid-CO2 HOF campaign is the most elaborate use of this repo, but the infrastructure is broader than that. A much simpler and already-supported use case is:

1. prepare a set of known molecules with ground-truth crystal structures
2. sample `1000` AlphaPack candidates for each molecule
3. relax the generated crystals with the current relax workflow
4. compare the relaxed candidates against ground-truth CIFs
5. decide whether each target was hit, and record the best RMSD match

This benchmark mode does **not** require the full specialist-agent loop. It reuses only the parts of the stack that matter for a known-target crystal benchmark:

- AlphaPack sampling
- BOMLIP relax
- post-relax CIF collection
- ground-truth structure matching

In other words, for this task OCAgent should be treated as a workflow/orchestration layer, not as a molecule-discovery agent.

### When to use this path

Use this simpler path when:

- the molecules are already known
- the target experimental crystal structures are already available as CIFs
- there is no need for literature reasoning, proposal generation, property selection, or campaign memory
- the scientific question is: "does the current AP model sample a structure close to the known crystal?"

Do **not** use the full HOF agent loop for this benchmark unless you intentionally want autonomous molecule proposal on top of the CSP benchmark.

### Minimum inputs

For each benchmark target, prepare:

- one molecular input representation for AlphaPack sampling
  - typically the same kind of monomer input already used by the current AlphaPack CSP workflow
- one or more ground-truth CIFs for that molecule
  - multiple CIFs are allowed if you want to compare against polymorphs or multiple experimental structures
- a stable target prefix / identifier
  - this matters because the existing matcher groups files by filename prefix

At the batch level, you also need:

- the AP checkpoint you want to test
  - for the current benchmark workflow, the default intended checkpoint is:
    `/users/ffmmgg/scratch/AlphaPack_llm/hydra/singlerun/2026-06-06/llm_extra_new_random_trainval_add_cases/epoch=76-step=346632.ckpt`
  - do not silently fall back to the older humid-CO2 HOF checkpoint from the `2026-03-10/llm_extra_include_0109` run unless a benchmark is explicitly meant to reproduce that older model
- the output root for the sampled and relaxed structures
- the Slurm partition / walltime profile for the sampling and relax stages

### Existing components that already cover the workflow

`Sampling`

- [sample_CSP_Z.py](/users/ffmmgg/scratch/AlphaPack_llm/AlphaPackllm/AP_llm/sample_CSP_Z.py)
- [run_APllm_sample_CSP_Z.sh](/users/ffmmgg/scratch/AlphaPack_llm/run_APllm_sample_CSP_Z.sh)

This is the current AP-side sampling entry point and is already used by the OCAgent CSP workflow.

`Relaxation`

- the current OCAgent CSP jobs already split work into `sampling -> relaxation -> postprocess`
- the relaxed final CIFs are collected under `relax/cif_result_final`
- OCAgent tests and workflow code already expect this artifact layout

`Ground-truth comparison`

- [evaluate_multi_real.py](/users/ffmmgg/scratch/AlphaPack_llm/AlphaPackllm/AP_llm/evaluate_multi_real.py)
- [run_APllm_evaluate_multi_real.sh](/users/ffmmgg/scratch/AlphaPack_llm/run_APllm_evaluate_multi_real.sh)

This script groups real and generated CIFs by filename prefix, runs structure matching, and writes:

- `all_rmsd_pairs__<prefix>.csv`
- `best_match_per_real__<prefix>.csv`

Those are already the right outputs for a "did we hit the ground truth?" benchmark.

### Recommended benchmark layout

The cleanest future-proof layout is:

```text
benchmark_root/
  inputs/
    <target_id>.xyz or other monomer input
  ground_truth/
    <target_id>_exp_01.cif
    <target_id>_exp_02.cif
  sample_output/
  pre_relax/
  relax/
    cif_result_final/
    json_result/
    cif_result_press/
  evaluation/
    all_rmsd_pairs__<target_id>.csv
    best_match_per_real__<target_id>.csv
  sample_artifacts.zip
  pre_relax_artifacts.zip
  relax_artifacts.zip
  benchmark_manifest.json
```

This layout mirrors the current HOF/CSP job outputs closely enough that we can reuse existing postprocess expectations without pretending this is a full autonomous-discovery campaign.

The most important operational rule is:

- one molecule = one working directory

That means each target should own its own:

- input file
- sample output directory
- pre-relax directory when used
- relax directory
- evaluation directory
- manifest / metadata

This benchmark should **not** be run as:

- one giant sampling job for dozens of molecules at once
- one giant relax job over tens of thousands of structures from many unrelated molecules mixed together

The correct granularity is molecule-local. A batch of many targets should therefore be represented as many sibling per-molecule job folders, each following the same mini-workflow independently.

The second operational rule is:

- keep the evaluation outputs, but archive bulky CSP artifacts immediately after evaluation

In practice this should mirror the current HOF workflow:

- preserve the scientific outputs needed for review and benchmarking
- compress `sample_output/`, `pre_relax/`, and `relax/` process artifacts into archives
- delete the expanded high-file-count directories once the archive is verified
- retain a restore path so individual CIFs can still be recovered later if needed

### Recommended execution phases

The intended execution plan for a many-molecule benchmark is:

1. prepare all benchmark inputs and ground-truth CIFs
2. create one benchmark job folder per target molecule
3. run AlphaPack sampling for that target with `sample_count = 1000`
4. run the current relax stage only for that target's generated samples
5. collect the relaxed final CIFs under that target's `relax/cif_result_final`
6. run [evaluate_multi_real.py](/users/ffmmgg/scratch/AlphaPack_llm/AlphaPackllm/AP_llm/evaluate_multi_real.py) for that target's ground-truth and relaxed CIFs
7. archive the bulky CSP / relax process directories for that target
8. summarize hit rate using the best RMSD per target

If many molecules are benchmarked, the outer loop should iterate over benchmark job folders rather than flatten all targets into one mixed sampling/relax/evaluation directory.

The archive step should happen **after** the match evaluation succeeds, because the evaluator reads the relaxed CIFs directly. Once `best_match_per_real__<prefix>.csv` and `all_rmsd_pairs__<prefix>.csv` are written, the job should move into the low-file-count archived state.

### Hit definition

The current matcher writes all pairwise successful matches and the best match per real CIF. For benchmark reporting, the usual summary should include:

- best RMSD to ground truth for each target
- whether a hit occurred under the chosen RMSD threshold
- which relaxed generated structure was the top hit
- optional top-k hit counts if multiple generated structures are close

The exact "hit" threshold should be fixed explicitly in the benchmark report rather than left implicit. That avoids mixing together runs that used different structure-matching tolerances.

### What still needs to be wrapped

The core pieces already exist, but the benchmark still deserves its own thin wrapper script. The missing layer is not chemistry; it is convenience and record-keeping.

The ideal wrapper should:

- read a simple benchmark manifest for dozens of molecules
- create one working directory per molecule
- launch the AP sampling stage
- launch the relax stage
- call the ground-truth matcher automatically
- archive bulky CSP/relax artifacts after successful evaluation
- gather per-target best-RMSD summaries into one benchmark CSV / JSON

The wrapper should also expose the AP checkpoint explicitly and default it to:

- `/users/ffmmgg/scratch/AlphaPack_llm/hydra/singlerun/2026-06-06/llm_extra_new_random_trainval_add_cases/epoch=76-step=346632.ckpt`

That keeps benchmark runs aligned with the intended current AP model and avoids accidental reuse of the older HOF-oriented checkpoint.

The wrapper should reuse the same archive discipline already used in the HOF workflow. The existing OCAgent postprocess layer already knows how to create:

- `sample_artifacts.zip` / `.tar.gz`
- `pre_relax_artifacts.zip` / `.tar.gz`
- `relax_artifacts.zip` / `.tar.gz`

and then remove expanded directories to control file count. The benchmark path should follow the same pattern rather than inventing a second archive convention.

That wrapper can stay much simpler than the HOF agent loop because it does not need:

- literature retrieval
- molecule proposal
- critic review
- property workflows
- campaign memory distillation

### Current extended-CSP runner

The benchmark wrapper now lives at:

- [run_extended_csp_benchmark.py](/mnt/scratch/users/ffmmgg/OCAgent/scripts/run_extended_csp_benchmark.py)

It is intended for datasets such as:

- `/mnt/scratch/users/ffmmgg/AlphaPack_llm/AlphaPackllm/data/extended_csp_cases`

where each target has:

- one monomer input: `<REFCODE>.xyz`
- one experimental crystal: `<REFCODE>.cif`

The runner does the following:

1. discover all paired `xyz/cif` targets
2. create one benchmark job directory per target under the chosen run root
3. submit AlphaPack sampling first
4. evaluate the raw `sample_output` directly against the ground-truth CIF
5. if `best_rmsd <= 0.2`, mark the target as an immediate match and skip relaxation entirely
6. if the raw samples do not hit the threshold, submit the `small` MACE-OFF relax stage
7. after relax/postprocess, restore only the archived `relax/cif_result_final` CIFs into a temporary evaluation folder
8. run [evaluate_multi_real.py](/mnt/scratch/users/ffmmgg/AlphaPack_llm/AlphaPackllm/AP_llm/evaluate_multi_real.py) on the relaxed structures
9. keep the evaluation CSVs and logs, then remove the temporary restored CIF folder
10. preserve per-target timing metadata for sampling, relaxation, and evaluation
11. write a benchmark registry JSON and a flat benchmark summary CSV

For timing and auditability, each completed benchmark target now keeps:

- sampling wall time parsed from `slurm-<sampling_jobid>.out`
- relaxation wall time parsed from `slurm-<relax_jobid>.out`
- direct sample-level evaluation under `sample_evaluation/`
- copied relax internal timing file as `evaluation/relax_internal_timing.csv` when available
- per-target timing summary JSON at `evaluation/timing_summary.json`
- flat timing columns in `benchmark_summary.csv`, including:
  - `match_stage`
  - `sample_best_rmsd`
  - `sampling_wall_seconds`
  - `sampling_wall_hms`
  - `relax_wall_seconds`
  - `relax_wall_hms`
  - `relax_internal_timing_csv`
  - `evaluation_seconds`

The default launch pattern is:

```bash
/mnt/scratch/users/ffmmgg/OCAgent/.venv-rdkit/bin/python \
  /mnt/scratch/users/ffmmgg/OCAgent/scripts/run_extended_csp_benchmark.py \
  --cases-root /mnt/scratch/users/ffmmgg/AlphaPack_llm/AlphaPackllm/data/extended_csp_cases \
  --run-root /mnt/scratch/users/ffmmgg/OCAgent/jobs/extended-csp-cases-2026-06-19-v1 \
  --checkpoint /mnt/scratch/users/ffmmgg/AlphaPack_llm/hydra/singlerun/2026-06-06/llm_extra_new_random_trainval_add_cases/epoch=76-step=346632.ckpt \
  --max-concurrent 10
```

There is also a shell wrapper with the same defaults:

```bash
bash /mnt/scratch/users/ffmmgg/OCAgent/scripts/start_extended_csp_benchmark.sh
```

For a long-running background session, use:

```bash
bash /mnt/scratch/users/ffmmgg/OCAgent/scripts/extended_csp_benchmark_tmux.sh
```

This starts a tmux session named `ocagent-extended-csp` and appends runner output to:

- `/mnt/scratch/users/ffmmgg/OCAgent/logs/extended_csp_benchmark.log`

The script is resumable. Running it again against the same `--run-root` will:

- reload `benchmark_registry.json`
- refresh already-submitted jobs
- auto-advance unfinished targets
- only submit new targets when capacity is available

This keeps the benchmark operationally close to the HOF workflow while staying lightweight enough for pure CSP hit-rate testing.

### Recommendation for future chats

If you open a fresh chat and want another model to continue this benchmark work efficiently, the most important instruction is:

"Use the existing AlphaPack benchmark sub-workflow in `OCAgent`, not the full humid-CO2 HOF agent loop. Reuse AP sampling, current relax/postprocess, and `evaluate_multi_real.py` ground-truth matching."

If the task is a multi-molecule benchmark, preserve the rule:

- one molecule = one folder = one sampling/relax/evaluation chain
- default benchmark checkpoint = `2026-06-06/llm_extra_new_random_trainval_add_cases/epoch=76-step=346632.ckpt`

That is the right abstraction boundary for this task.

## Specialist-agent architecture

The newer campaign loop is organized as a small society of specialist agents rather than one monolithic planner. The goal is reliability as much as chemical intelligence: one malformed molecule proposal, one Slurm status mismatch, or one property backend failure should not kill the entire autonomous discovery campaign.

The implementation entry point is:

```bash
organic-crystal-agent expert-step --campaign-path /path/to/campaign.json
```

For long-running campaigns, use:

```bash
bash openclaw/scripts/run_expert_loop.sh
```

### Why split the agent

The earlier loop mixed several jobs in one control path:

- deciding the next campaign stage
- proposing molecules
- judging molecule quality
- submitting CSP
- refreshing Slurm state
- selecting property calculations
- interpreting results

That design is easy to start, but fragile in production. If the molecule generator returns malformed JSON, the whole loop can crash. If a Slurm job finishes but local job metadata is not refreshed, the campaign can appear stuck. If property selection is hard-coded as "adsorption", the same loop is harder to reuse for ORCA electronic calculations, RASPA2 adsorption, packing interaction jobs, or future property workflows.

The specialist architecture makes each responsibility explicit and gives every step a narrow contract.

### Expert roles

`ExpertSupervisor`

The supervisor is the only component allowed to decide campaign state transitions. It reads the campaign JSON, refreshes running jobs, chooses one next specialist action, records a `PlannerTrace`, and writes file-backed memory. It does not design molecules or interpret crystal landscapes itself.

`LiteratureExpert`

The literature expert maintains campaign-level evidence cards and design implications. For humid CO2 HOF discovery this includes local PDF/XLSX evidence, fragment families, Nat Chem-inspired five-membered or pentagonal cavity motifs, and warnings against overly hydrophilic pore linings.

When OpenAI access is available, the literature layer can also search the web and read open-access literature through the OpenAI Responses web-search path. The default mode is now `auto`: it prefers the OpenAI-backed literature agent when the API key and package are available, and otherwise falls back to the local/heuristic literature path.

The literature expert is no longer one-shot only. It can be re-triggered after new CSP/property evidence appears so later rounds can add supplementary literature hypotheses rather than freezing the initial literature picture forever.

`MoleculeProposalExpert`

The proposal expert only drafts candidate molecules. It uses the active objective, literature cards, prior winners/losers, and fragment evidence. It is deliberately not trusted to approve its own output. Malformed JSON or invalid output is recorded as a failed proposal attempt, not a fatal loop error.

`MoleculeCriticExpert`

The critic expert reviews proposed molecules before campaign registration. It checks RDKit validity, duplicate SMILES, disconnected molecules, HOF-relevant rigidity and donor/acceptor inventory, symmetry preference, synthetic complexity, and design red flags. Instead of trusting a template-driven retrosynthesis gate, the critic now directly judges whether a candidate looks too synthetically overcomplicated for a practical screening campaign. Accepted molecules are then materialized to 3D inputs with RDKit.

The expert loop now treats the batch size as a sliding-window throughput target. It tries to keep up to that many unfinished molecules moving through the pipeline at once. As soon as one molecule finishes its full campaign-relevant evaluation, the supervisor is allowed to draft a replacement instead of waiting for the rest of the old batch to finish.

`CSPLandscapeExpert`

The CSP/landscape expert owns pre-CSP monomer optimization, CSP submission, Slurm refresh, landscape fetching, deduplication summaries, and landscape readiness. It is also the natural home for future geometry sanity checks such as intermolecular clash filtering and false bonded-structure rejection.

`PropertyExpert`

The property expert replaces the narrower idea of an adsorption-only agent. It chooses the target-aligned property workflow:

- RASPA2 humid CO2 / H2O adsorption for humid capture campaigns
- RASPA2 methane adsorption for methane campaigns
- ORCA frontier, gap, dipole, pair interaction, or other electronic properties for molecular electronics or packing-analysis campaigns
- future property calculators through the same `PropertyJobRecord` interface

This keeps the discovery loop application-general while still supporting the current humid CO2 HOF objective.

The property expert is now allowed to act on partial campaign progress. If some molecules are already `landscape_ready` while other CSP jobs are still queued or running, it can still submit the finished molecules to the appropriate property workflow instead of waiting for the entire CSP batch to finish.

`KnowledgeExpert`

The knowledge expert is the campaign-level distillation role. It now supports incremental distillation: whenever new succeeded CSP or property results appear, it can refresh campaign memory even if other Slurm-backed jobs are still active.

Its job is not to propose molecules directly. Instead, it:

- extracts cross-round landscape and property signals
- identifies durable failure modes such as collapse-heavy CSP outcomes, low porous survival, or repeated synthetic-complexity rejection
- writes explicit next-round search biases for the proposer and critic
- keeps the memory layer synchronized so later rounds are influenced by campaign evidence, not only static literature priors

The current memory outputs written by this role include:

- `memory/knowledge_insights.md`
- `memory/search_bias.md`
- `reports/knowledge_report.json`

### State flow

The expert loop follows a conservative state machine:

```text
refresh jobs
  -> collect literature if missing or if new results justify a supplementary literature refresh
  -> distill campaign knowledge when new completed results exist, even if other jobs are still active
  -> submit target property jobs for any molecules already landscape-ready
  -> if fewer than the target number of unfinished molecules remain, propose replacements for the missing slots
  -> submit any ready proposed molecules to pre-CSP DFT / CSP without waiting for a full synchronous batch
  -> wait if active jobs still remain and no further safe asynchronous action is available
  -> record round summary and next-actions through KnowledgeExpert
  -> propose molecules
  -> critique and register accepted molecules
  -> materialize RDKit 3D inputs
  -> repeat
```

The supervisor no longer treats each round as a rigid synchronous five-molecule block. Instead, it keeps a bounded rolling queue of unfinished molecules. Literature supplementation, incremental knowledge distillation, property submission for already-finished landscapes, and replacement proposals for fully completed molecules can all proceed while slower CSP jobs are still running.

### Fault tolerance

Specialist actions are crash-isolated. A failed proposal parse, failed critic gate, or failed property selection is recorded in the campaign notes and planner trace. The next heartbeat can retry or choose another action. The long-running shell loop also keeps going after a failed `expert-step` and logs the failure in:

```text
openclaw/runtime/ocagent_expert_loop.log
```

The campaign JSON remains the source of truth. Every expert step appends a `PlannerTrace` with the specialist names, chosen actions, and action details, making the run auditable for later paper figures and method descriptions.

### Memory products

The expert loop now maintains two layers of memory:

`State memory`

This is the typed campaign JSON itself: molecules, job records, property metrics, decisions, and planner traces.

`File-backed research memory`

This is written under the campaign `memory/` and `reports/` directories so humans and later agents can inspect the campaign as a research program rather than only as a queue of jobs.

Important files include:

- `memory/literature.md`: campaign-specific evidence cards and design implications
- `memory/molecule_design_rules.md`: the active design priors and registered-molecule overview
- `memory/failure_patterns.md`: recurring collapse, workflow, or geometry failure patterns
- `memory/winning_patterns.md`: best landscape/property outcomes observed so far
- `memory/knowledge_insights.md`: cross-round distilled scientific signals
- `memory/search_bias.md`: explicit instructions for how the next round should shift its search
- `reports/campaign_summary.json`: structured summary for dashboards or paper figures
- `reports/knowledge_report.json`: machine-readable distilled knowledge snapshot

The molecule proposer can read the distilled knowledge and search-bias files on later rounds, so the system's learning signal is no longer limited to static literature plus the latest few molecules.

### Does the loop need a restart after code updates?

Usually no full campaign reset is needed.

`run_expert_loop.sh` launches a fresh Python process on every heartbeat:

```bash
python -m organic_crystal_agent.cli expert-step ...
```

That means most code changes are effectively hot-loaded on the next loop cycle. In practice:

- if the tmux loop is still running, the next heartbeat will pick up the new code automatically
- if you want the new behavior immediately, restarting the tmux loop is still a clean option
- you do **not** need to reinitialize the campaign or discard completed jobs just because the expert logic changed

You should only restart from scratch if the campaign state itself is already scientifically compromised and you want a clean experimental record.

### Recommended expert-loop launch

```bash
cd /users/ffmmgg/scratch/OCAgent
module load rdkit/2025_03_5-gcc14.2.0
source .venv-rdkit/bin/activate
export PYTHONPATH=/users/ffmmgg/scratch/OCAgent/src
export OPENAI_API_KEY="..."

tmux new-session -d -s ocagent-expert-humid-co2 \
'cd /users/ffmmgg/scratch/OCAgent && \
OPENCLAW_CAMPAIGN_PATH=/users/ffmmgg/scratch/OCAgent/openclaw/workspace/campaigns/humid-co2-hof-ai-test-2026-04-12-v1/campaign.json \
POLL_SECONDS=180 \
OPENCLAW_SAMPLE_COUNT=1000 \
MOLECULE_BATCH_SIZE=5 \
ORGANIC_CRYSTAL_AGENT_OPENAI_MOLECULE_MODEL=gpt-5.5 \
ORGANIC_CRYSTAL_AGENT_OPENAI_MODEL=gpt-5.5-mini \
ORGANIC_CRYSTAL_AGENT_OPENAI_LITERATURE_MODEL=gpt-5.5 \
ORGANIC_CRYSTAL_AGENT_RETROSYN_ENABLED=1 \
ORGANIC_CRYSTAL_AGENT_RETROSYN_STRICT=1 \
ORGANIC_CRYSTAL_AGENT_RETROSYN_MAX_STEPS=5 \
bash openclaw/scripts/run_expert_loop.sh'
```

Attach and detach safely:

```bash
tmux attach -t ocagent-expert-humid-co2
# detach with Ctrl-b d
```

Monitor without attaching:

```bash
tail -f /users/ffmmgg/scratch/OCAgent/openclaw/runtime/ocagent_expert_loop.log
squeue -u "$USER"
```

## Compute backends

### ORCA

The repo is configured for this cluster's module-based ORCA environment:

```bash
module load orca/6.0.1
```

Important: the module exists, but the cluster currently reports `Permission denied` for the licensed ORCA binary until your account is added to the ORCA working group after registration on:

- https://orcaforum.kofo.mpg.de

### LAMMPS

LAMMPS is available on this cluster and can be loaded with:

```bash
module load lammps/20250722-openmpi5.0.8-gcc14.2.0
```

The current LAMMPS integration is a `generic LJ proxy relaxation`, intended for rapid screening of extracted monomer/dimer/cluster xyz files before heavier quantum calculations.

### RDKit

RDKit should be treated as a required chemistry capability for OCAgent:

- canonical SMILES normalization
- known-molecule novelty filtering
- ring / rotatable-bond / donor-acceptor counting
- later scaffold and descriptor-based design rules

On this cluster there is an RDKit module available:

```bash
module load rdkit/2025_03_5-gcc14.2.0
```

Important: the current default `.venv` was created with Python `3.13`, while the cluster RDKit module is built against Python `3.11`. That means the default venv is not ABI-compatible with the module.

Use the dedicated setup script to build an RDKit-compatible environment instead:

```bash
bash scripts/setup_rdkit_env.sh
module load rdkit/2025_03_5-gcc14.2.0
source .venv-rdkit/bin/activate
```

## Recommended login-node workflow

For this cluster, the most practical way to keep the discovery loop alive is:

- run the light OCAgent supervisor on the login node
- keep heavy `sample / relax / postprocess` work on Slurm
- keep the supervisor inside `tmux` so SSH or VSCode disconnects do not stop it

Initialize a porous HOF campaign once:

```bash
cd /users/ffmmgg/scratch/OCAgent
bash scripts/init_porous_hof_campaign.sh
```

Start the long-running supervisor in `tmux`:

```bash
cd /users/ffmmgg/scratch/OCAgent
bash scripts/start_campaign_tmux.sh
```

Attach to the live session:

```bash
cd /users/ffmmgg/scratch/OCAgent
bash scripts/attach_campaign_tmux.sh
```

Check status without attaching:

```bash
cd /users/ffmmgg/scratch/OCAgent
bash scripts/campaign_tmux_status.sh
```

Stop the supervisor:

```bash
cd /users/ffmmgg/scratch/OCAgent
bash scripts/stop_campaign_tmux.sh
```

The default campaign file is:

- `/users/ffmmgg/scratch/OCAgent/campaigns/porous_hof.json`

The default tmux session name is:

- `ocagent-hof`

The default log file is:

- `/users/ffmmgg/scratch/OCAgent/logs/ocagent-hof.log`

### CSP Slurm defaults

The CSP workflow currently defaults to:

- `partition = gpu-a100-cooper,gpu-h100`
- `time_limit = 3-00:00:00`
- `gpus = 1`
- `sampler expand_maxV = 3.0`
- `relax model = small`
- `relax max_steps = 1000`

This matches the current cluster constraint that `gpu-h100` jobs should stay within 3 days, while still allowing Slurm to place the job on either preferred GPU partition.

## Resume / continue behavior

Long discovery campaigns are expected to exceed a single 10-day Slurm walltime limit. The current workflow therefore uses `resume-by-stage` instead of assuming one uninterrupted job.

Current behavior:

- `CSP` jobs now run as three Slurm stages: `sampling -> relaxation -> postprocess`
- `CSP` jobs keep a `stage_state.json` in the job directory and `csp-status` advances the workflow by submitting the next stage when the previous one finishes
- `sampling` runs in the AlphaPack apptainer environment, `relaxation` runs in the BOMLIP conda environment, and `postprocess` handles dedup / landscape merge / archive cleanup
- `postprocess` defaults to the `cooper` partition with a `03:00:00` walltime so archive / dedup work does not consume GPU slots
- `postprocess` now also writes SVG landscape plots with `density` on the x-axis and relative `energy_kj` (minimum normalized to 0) on the y-axis
- `ORCA` jobs reuse the same working directory and skip profiles whose expected output files already exist
- `LAMMPS` jobs reuse the same working directory and skip relaxation if `final.xyz` and the final energy marker are already present

### Formal spike identification rule

For a relaxed crystal landscape, spike identification is carried out on the refined-deduplicated landscape CSV after the global minimum has been shifted to `relative_energy_kj = 0`.

Definitions:

- each point is a deduplicated relaxed crystal structure with `(density, relative_energy_kj)`
- the landscape is sorted by increasing `density`
- the lower envelope is the ordered subset of structures that successively decrease the best observed energy as density increases
- a density neighborhood is a consecutive block of lower-envelope points whose pairwise stepwise density gaps do not exceed `density_cluster_window`
- each density neighborhood contributes one candidate basin floor: the lowest-energy structure inside that neighborhood

Operationally, the current procedure is:

1. sort the refined-deduplicated landscape by `density`
2. build the lower envelope by keeping only points that set a new running minimum in `energy_kj`
3. partition consecutive lower-envelope points into density neighborhoods using `density_cluster_window = 0.10 g/cm^3`
4. within each neighborhood, identify the local lowest-energy structure
5. reject that candidate if its `relative_energy_kj` exceeds the campaign spike window, currently `50 kJ/mol`
6. apply the neighborhood-span rule described below

Let the lower-envelope density neighborhoods be indexed by increasing density as
`C_1, C_2, ..., C_m`.

For each neighborhood `C_i`, define:

- local floor energy:
  `E_i^min = min_{p in C_i} E_rel(p)`

- local top energy:
  `E_i^max = max_{p in C_i} E_rel(p)`

- neighborhood span:
  `Delta_i = E_i^max - E_i^min`

where `E_rel(p)` is the relative energy of point `p` in `kJ/mol`.

The spike acceptance rule is:

- energy-window condition:
  `E_i^min <= 50`

- multi-point neighborhood condition:
  if `|C_i| >= 2`, accept the neighborhood floor if
  `Delta_i >= 10`

- single-point neighborhood condition:
  if `|C_i| = 1`, accept the point only if `i > 1` and
  `E_{i-1}^min - E_i^min >= 10`

This means a singleton spike is defined only when it opens a new lower-energy basin relative to the immediately preceding lower-density neighborhood on the lower envelope.

Interpretation:

- a spike is a metastable basin floor identified on the lower envelope
- the basin floor must lie inside the campaign metastability window
- a multi-point basin must show an internal energy span of at least `10 kJ/mol`
- a singleton basin must sit at least `10 kJ/mol` below the floor of the previous lower-density envelope neighborhood
- a global minimum may be retained when it belongs to a multi-point neighborhood satisfying the same span criterion

Artifacts written by postprocess:

- `optimized_landscape.csv`: all relaxed structures merged into one landscape table
- `optimized_landscape_dedup.csv`: deduplicated landscape used for scientific review
- `spikes/spike_manifest.json`: structured spike records
- `spikes/spike_summary.csv`: tabular spike summary
- `spikes/*.cif`: exported spike representative CIFs

### Current humid-CO2 reevaluation policy

For the current refined humid-CO2 reprocessing campaign, every identified spike is treated as a real adsorption target.

That means:

- spike identification is a scientific landscape-analysis step
- adsorption submission is no longer restricted to only one low-density representative per molecule
- once retrospective reprocessing is complete, the target state is that every identified spike is either:
  - already covered by a completed humid-CO2 adsorption job, or
  - already submitted and waiting to finish

This policy is stricter than the older lightweight screening convention that excluded the highest-density spike during a first adsorption pass.

Requeue commands:

```bash
organic-crystal-agent csp-requeue --job-path /path/to/job.json
organic-crystal-agent orca-requeue --job-path /path/to/job.json
organic-crystal-agent lammps-requeue --job-path /path/to/job.json
```

## Usage

Install dependencies:

```bash
pip install -e .
pip install -e .[csp]
```

Optional OpenAI research backend:

```bash
pip install -e .[openai]
export OPENAI_API_KEY="..."
export ORGANIC_CRYSTAL_AGENT_RESEARCH_BACKEND=openai
```

The app does **not** require an API key for basic use. If the OpenAI backend is unavailable, it falls back to the built-in mock research backend instead of crashing.

Optional local UI:

```bash
pip install -e .[ui]
organic-crystal-agent ui --port 8501
```

Server deployment notes:

- [SERVER_DEPLOYMENT.md](/home/ffmmgg/projects/HOF_agent/SERVER_DEPLOYMENT.md)
- bootstrap script: [scripts/setup_server.sh](/home/ffmmgg/projects/HOF_agent/scripts/setup_server.sh)

Run one planning step:

```bash
organic-crystal-agent run "Find promising crystal forms for a rigid aromatic acceptor for charge transport"
```

Persist a campaign and the latest proposal:

```bash
organic-crystal-agent plan \
  "Find promising crystal forms for a rigid aromatic acceptor for charge transport" \
  --campaign-path campaigns/default_campaign.json \
  --proposal-path campaigns/latest_proposal.json
```

Record the outcome of the latest primary trial:

```bash
organic-crystal-agent record \
  --outcome-category solvate \
  --summary "Recovered a solvated phase instead of the target form." \
  --evidence "PXRD suggests guest-filled lattice" \
  --evidence "TGA shows early mass loss" \
  --pxrd-status mismatch \
  --thermal-status solvent_loss \
  --mass-loss-percent 8.5 \
  --single-crystal-status powder_only \
  --campaign-path campaigns/default_campaign.json \
  --proposal-path campaigns/latest_proposal.json
```

The `record` command supports structured characterization fields for:

- `PXRD`
- `thermal analysis`
- `single-crystal availability`

These fields are used to infer likely failure modes such as:

- `solvate_trap`
- `competing_polymorph`
- `poor_crystallinity`
- `inconclusive_characterization`

If `ORGANIC_CRYSTAL_AGENT_RESEARCH_BACKEND` is unset, the platform uses:

- `openai` automatically when `OPENAI_API_KEY` is available
- otherwise `mock`

Run tests:

```bash
pytest
```

## Discovery workflow usage

Initialize a long-running campaign:

```bash
organic-crystal-agent campaign-init \
  --campaign-path campaigns/porous.json \
  --campaign-id porous-screen \
  --target-description "Find porous organic crystals with favorable transport-related packing."
```

Submit a CSP workflow:

```bash
organic-crystal-agent campaign-submit-csp \
  --campaign-path campaigns/porous.json \
  --molecule-id mol-001 \
  --sample-count 1000 \
  --z-value 4
```

Preview the exact generated CSP Slurm script without submitting:

```bash
organic-crystal-agent csp-preview \
  --molecule-path /path/to/candidate.xyz \
  --molecule-label candidate_a \
  --sample-count 1000 \
  --z-value 4
```

Extract packing features from a packed or bonded CIF:

```bash
organic-crystal-agent packing-extract \
  --cif-path /path/to/sample_bonded.cif \
  --out-dir /path/to/packing_outputs \
  --max-dimers 12 \
  --cluster-size 4
```

Submit ORCA pair-interaction jobs for the top extracted dimers:

```bash
organic-crystal-agent packing-submit-pairs \
  --packing-summary-path /path/to/packing_outputs/packing_summary.json \
  --top-n 5 \
  --method B3LYP \
  --basis def2-SVP
```

Refresh ORCA pair-interaction batch results:

```bash
organic-crystal-agent packing-refresh-pairs \
  --batch-manifest-path /path/to/packing_outputs/orca_pair_batch/pair_batch_manifest.json
```

Submit LAMMPS local-cluster relaxations:

```bash
organic-crystal-agent packing-submit-clusters \
  --packing-summary-path /path/to/packing_outputs/packing_summary.json \
  --top-n 5 \
  --max-iter 2000
```

Refresh LAMMPS cluster-relaxation batch:

```bash
organic-crystal-agent packing-refresh-clusters \
  --batch-manifest-path /path/to/packing_outputs/lammps_cluster_batch/cluster_batch_manifest.json
```

Attach packing analysis to a campaign molecule:

```bash
organic-crystal-agent campaign-submit-packing \
  --campaign-path campaigns/porous.json \
  --molecule-id mol-001
```

Show campaign packing ranking:

```bash
organic-crystal-agent campaign-packing-report \
  --campaign-path campaigns/porous.json
```

Submit a standalone ORCA property job:

```bash
organic-crystal-agent orca-submit \
  --structure-path /path/to/mol.xyz \
  --molecule-label candidate_a \
  --property-profile hole_reorg
```

Submit a standalone LAMMPS proxy relaxation:

```bash
organic-crystal-agent lammps-submit \
  --structure-path /path/to/dimer.xyz \
  --system-label dimer_001 \
  --save-job-path jobs/lammps_dimer_001.json
```

Fetch LAMMPS results:

```bash
organic-crystal-agent lammps-fetch --job-path jobs/lammps_dimer_001.json
```

## Design priors and unfamiliar applications

The front-end design stack now uses a shared `application spec` layer:

- for familiar applications such as `HOF` and `charge_transport`, the system injects built-in design priors
- for unfamiliar applications, it generates an inferred spec from the target description and desired properties

Each application spec contains:

- hard constraints
- soft preferences
- red flags
- evaluation criteria
- synthesis heuristics
- literature search queries

The current planning layer uses this spec to:

- steer literature synthesis
- enrich molecule design evidence
- write the application prior into file-backed memory for later planner steps
- reject weak candidate proposals before they enter CSP
- run a second-pass design critic that looks for red-flag patterns and revision targets
- reject candidates whose canonical SMILES already appears in the known CSD-derived registry

The current known-molecule registry is:

- `/users/ffmmgg/scratch/AlphaPack_llm/AlphaPackllm/data/CSD_smiles/smiles_properties_all.csv`

This CSV already stores canonical SMILES, so OCAgent compares candidate canonical SMILES directly against the `smiles` column without re-canonicalizing the registry.

This is intended to keep the workflow useful both for:

- common high-prior domains with known crystal-engineering heuristics
- open-ended discovery tasks where the agent must first discover plausible design rules

## Design principle

This repo treats an agent platform as a software system, not as a chat wrapper.

The important abstractions are:

- `request`
- `tool call`
- `observation`
- `campaign memory`
- `next experiment proposal`
- `verification`

That is the right starting point for an organic crystal agent that will eventually need to reason over:

- polymorphs
- packing motifs
- solvates
- co-crystals
- realization risk
- crystallization conditions
