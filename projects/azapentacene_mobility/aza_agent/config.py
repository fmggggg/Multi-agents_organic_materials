"""Machine-specific locations. Every entry can be overridden with the environment variable named next to it;
the defaults are the paths on the development machine (viz1)."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def _env(var: str, default) -> str:
    return os.environ.get(var) or str(default)


# Clari-H crystal structure generator (source checkout; imported in-process by the orchestrator)
CLARI_ROOT = _env("AZA_CLARI_ROOT", "/mnt/scratch/users/ffmmgg/clari")
HF_HOME = _env("AZA_HF_HOME", Path.home() / ".cache/huggingface_clari")

# MACE-OFF worker: python of an environment with mace-torch + ase + pymatgen (BOMLIP_CSP env)
MACE_PY = _env("AZA_MACE_PY", "/mnt/fastscratch/users/ffmmgg/conda_envs/BOMLIP_CSP/bin/python")
MACE_CACHE = _env("AZA_MACE_CACHE", "/mnt/fastscratch/users/ffmmgg/.cache")

# MCRT worker: apptainer image with the MCRT environment (the MCRT source + checkpoint paths are read by
# workers/mcrt_worker.py from AZA_MCRT_ROOT / AZA_MCRT_CKPT, see there)
MCRT_SIF = _env("AZA_MCRT_SIF", "/mnt/scratch/users/ffmmgg/MCRT_container.sif")
APPTAINER = _env("AZA_APPTAINER", "/opt/apps/pkg/tools/apptainer/1.3.6/bin/apptainer")
APPTAINER_BINDS = _env("AZA_APPTAINER_BINDS", "/mnt/scratch,/mnt/fastscratch")

# reference set of known azapentacenes (duplicate gate); not shipped with the repository
KNOWN_CSV = _env("AZA_KNOWN_CSV", PROJECT / "data/mobility_xyz_bonded_from_inchi.csv")

# LLM API keys are read from these files (or OPENAI_API_KEY / OPENROUTER_API_KEY); never commit keys
OPENAI_KEY_FILE = _env("OPENAI_API_KEY_FILE", "/mnt/scratch/users/ffmmgg/OCAgent/.openai_api_key")
OPENROUTER_KEY_FILE = _env("OPENROUTER_API_KEY_FILE", Path.home() / ".config/openrouter_api_key")
