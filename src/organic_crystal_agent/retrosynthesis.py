from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel


class RetrosynthesisConfig(BaseModel):
    enabled: bool = False
    strict: bool = False
    max_steps: int = 5
    buyables_jsonl: str = ""
    buyables_smiles: str = ""
    stock_hdf5: str = ""
    aizynth_config: str = ""
    expansion_model: str = ""
    expansion_templates: str = ""
    filter_model: str = ""


class RetrosynthesisStatus(BaseModel):
    aizynthfinder_available: bool
    smiles2stock_available: bool
    buyables_jsonl_exists: bool
    buyables_smiles_exists: bool
    stock_hdf5_exists: bool
    aizynth_config_exists: bool
    config: RetrosynthesisConfig


class RetrosynthesisAssessment(BaseModel):
    smiles: str
    evaluated: bool = False
    passes: bool = False
    is_solved: bool = False
    steps: int | None = None
    precursor_count: int | None = None
    precursor_in_stock_count: int | None = None
    precursor_smiles: list[str] = []
    output_path: str = ""
    notes: list[str] = []


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_retrosyn_dir() -> Path:
    return resolve_repo_root() / "retrosyn"


def resolve_retrosynthesis_config() -> RetrosynthesisConfig:
    retrosyn_dir = default_retrosyn_dir()
    public_dir = retrosyn_dir / "public_data"
    buyables_jsonl = Path(
        os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_BUYABLES_JSONL",
            str(retrosyn_dir / "buyables.jsonl"),
        )
    )
    buyables_smiles = Path(
        os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_BUYABLES_SMILES",
            str(retrosyn_dir / "buyables.smiles"),
        )
    )
    stock_hdf5 = Path(
        os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_STOCK_HDF5",
            str(retrosyn_dir / "buyables_stock.hdf5"),
        )
    )
    aizynth_config = Path(
        os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_CONFIG",
            str(retrosyn_dir / "aizynth_config.yml"),
        )
    )
    return RetrosynthesisConfig(
        enabled=os.getenv("ORGANIC_CRYSTAL_AGENT_RETROSYN_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
        strict=os.getenv("ORGANIC_CRYSTAL_AGENT_RETROSYN_STRICT", "0").lower() in {"1", "true", "yes", "on"},
        max_steps=max(0, int(os.getenv("ORGANIC_CRYSTAL_AGENT_RETROSYN_MAX_STEPS", "5"))),
        buyables_jsonl=str(buyables_jsonl),
        buyables_smiles=str(buyables_smiles),
        stock_hdf5=str(stock_hdf5),
        aizynth_config=str(aizynth_config),
        expansion_model=os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_MODEL",
            str(public_dir / "uspto_model.onnx"),
        ),
        expansion_templates=os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_TEMPLATES",
            str(public_dir / "uspto_templates.csv.gz"),
        ),
        filter_model=os.getenv(
            "ORGANIC_CRYSTAL_AGENT_RETROSYN_FILTER_MODEL",
            str(public_dir / "uspto_filter_model.onnx"),
        ),
    )


def extract_buyable_smiles(
    jsonl_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    src = Path(jsonl_path).expanduser().resolve()
    dst = Path(output_path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    seen: set[str] = set()
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            smiles = str(obj.get("smiles", "")).strip()
            if not smiles:
                skipped += 1
                continue
            if smiles in seen:
                continue
            seen.add(smiles)
            fout.write(smiles + "\n")
            written += 1
    return {
        "jsonl_path": str(src),
        "output_path": str(dst),
        "written_smiles": written,
        "skipped_records": skipped,
    }


def write_aizynthfinder_config(
    output_path: str | Path,
    *,
    stock_hdf5: str | Path,
    max_transforms: int = 2,
    iteration_limit: int = 150,
    time_limit: int = 120,
    expansion_model: str = "${ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_MODEL}",
    expansion_templates: str = "${ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_TEMPLATES}",
    filter_model: str = "${ORGANIC_CRYSTAL_AGENT_RETROSYN_FILTER_MODEL}",
) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    expansion_model_path = str(expansion_model)
    expansion_templates_path = str(expansion_templates)
    filter_model_path = str(filter_model)
    payload = {
        "search": {
            "algorithm": "mcts",
            "max_transforms": max_transforms,
            "iteration_limit": iteration_limit,
            "time_limit": time_limit,
        },
        "expansion": {
            "buyable_route_policy": {
                "type": "template-based",
                "model": expansion_model_path,
                "template": expansion_templates_path,
            }
        },
        "filter": {
            "buyable_route_filter": {
                "type": "quick-filter",
                "model": filter_model_path,
            }
        },
        "stock": {
            "buyables": {
                "type": "inchiset",
                "path": str(Path(stock_hdf5).expanduser().resolve()),
            }
        },
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def smiles2stock_available() -> bool:
    return shutil.which("smiles2stock") is not None


def rdkit_inchi_key_available() -> bool:
    try:
        from rdkit import Chem
    except Exception:
        return False
    if hasattr(Chem, "MolToInchiKey"):
        return True
    try:
        from rdkit.Chem import inchi  # noqa: F401
    except Exception:
        return False
    return hasattr(Chem, "MolToInchiKey")


def ensure_buyable_stock_hdf5(
    *,
    smiles_path: str | Path,
    stock_hdf5: str | Path,
) -> dict[str, str | bool]:
    smiles_file = Path(smiles_path).expanduser().resolve()
    stock_file = Path(stock_hdf5).expanduser().resolve()
    stock_file.parent.mkdir(parents=True, exist_ok=True)
    force_rebuild = os.getenv("ORGANIC_CRYSTAL_AGENT_RETROSYN_REBUILD_STOCK", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if stock_file.exists() and not force_rebuild:
        return {
            "created": False,
            "stock_hdf5": str(stock_file),
            "message": "Existing stock found; keeping it. Set ORGANIC_CRYSTAL_AGENT_RETROSYN_REBUILD_STOCK=1 to force a rebuild.",
        }

    if not smiles2stock_available():
        raise RuntimeError(
            "smiles2stock is not available on PATH. Install AiZynthFinder and ensure its CLI entrypoints are available."
        )
    if not rdkit_inchi_key_available():
        if stock_file.exists():
            return {
                "created": False,
                "stock_hdf5": str(stock_file),
                "message": "RDKit lacks InChI support in this environment, so the existing stock is being reused.",
            }
        raise RuntimeError(
            "RDKit in the active environment does not expose MolToInchiKey / InChI support, so AiZynthFinder cannot build a new stock file here."
        )

    proc = subprocess.run(
        [
            "smiles2stock",
            "--files",
            str(smiles_file),
            "--output",
            str(stock_file),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        message = stderr or stdout or "smiles2stock exited with a non-zero status."
        if stock_file.exists():
            return {
                "created": False,
                "stock_hdf5": str(stock_file),
                "message": f"smiles2stock failed, but an existing stock is available and will be reused: {message}",
            }
        raise RuntimeError(f"smiles2stock failed: {message}")
    return {
        "created": True,
        "stock_hdf5": str(stock_file),
        "message": "Built AiZynthFinder stock from buyable SMILES.",
    }


def aizynthfinder_available() -> bool:
    try:
        import aizynthfinder  # noqa: F401
    except Exception:
        return False
    return True


def retrosynthesis_status() -> RetrosynthesisStatus:
    config = resolve_retrosynthesis_config()
    return RetrosynthesisStatus(
        aizynthfinder_available=aizynthfinder_available(),
        smiles2stock_available=smiles2stock_available(),
        buyables_jsonl_exists=Path(config.buyables_jsonl).exists(),
        buyables_smiles_exists=Path(config.buyables_smiles).exists(),
        stock_hdf5_exists=Path(config.stock_hdf5).exists(),
        aizynth_config_exists=Path(config.aizynth_config).exists(),
        config=config,
    )


def retrosynthesis_filter_enabled() -> bool:
    return resolve_retrosynthesis_config().enabled


def evaluate_retrosynthesis(smiles: str) -> RetrosynthesisAssessment:
    config = resolve_retrosynthesis_config()
    assessment = RetrosynthesisAssessment(smiles=smiles)
    if not config.enabled:
        assessment.notes.append("Retrosynthesis filter is disabled.")
        return assessment
    if not aizynthfinder_available():
        assessment.notes.append("AiZynthFinder is not installed in the active environment.")
        return assessment

    config_path = Path(config.aizynth_config).expanduser().resolve()
    if not config_path.exists():
        assessment.notes.append(f"AiZynthFinder config is missing: {config_path}")
        return assessment

    with tempfile.TemporaryDirectory(prefix="ocagent-retrosyn-") as tmpdir:
        output_path = Path(tmpdir) / "routes.json"
        command = [
            "aizynthcli",
            "--smiles",
            smiles,
            "--config",
            str(config_path),
            "--policy",
            "buyable_route_policy",
            "--filter",
            "buyable_route_filter",
            "--stocks",
            "buyables",
            "--output",
            str(output_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_MODEL": config.expansion_model,
                    "ORGANIC_CRYSTAL_AGENT_RETROSYN_EXPANSION_TEMPLATES": config.expansion_templates,
                    "ORGANIC_CRYSTAL_AGENT_RETROSYN_FILTER_MODEL": config.filter_model,
                },
            )
        except subprocess.CalledProcessError as exc:
            assessment.notes.append(
                f"AiZynthFinder failed with return code {exc.returncode}: {(exc.stderr or exc.stdout or '').strip()}"
            )
            return assessment

        if not output_path.exists():
            assessment.notes.append("AiZynthFinder completed but did not write a route JSON output.")
            return assessment

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        routes = payload if isinstance(payload, list) else []
        best = _select_best_route(routes)
        assessment.evaluated = True
        assessment.output_path = str(output_path)
        if best is None:
            assessment.notes.append("AiZynthFinder returned no routes.")
            return assessment

        scores = best.get("scores", {})
        metadata = best.get("metadata", {})
        assessment.is_solved = bool(metadata.get("is_solved", False))
        assessment.steps = _safe_int(scores.get("number of reactions"))
        assessment.precursor_count = _safe_int(scores.get("number of pre-cursors"))
        assessment.precursor_in_stock_count = _safe_int(scores.get("number of pre-cursors in stock"))
        assessment.precursor_smiles = sorted(_leaf_precursors(best))
        assessment.passes = (
            assessment.is_solved
            and assessment.steps is not None
            and assessment.steps <= config.max_steps
            and assessment.precursor_count is not None
            and assessment.precursor_in_stock_count is not None
            and assessment.precursor_count == assessment.precursor_in_stock_count
        )
        if assessment.passes:
            assessment.notes.append(
                f"AiZynthFinder found a solved route in {assessment.steps} step(s) with all precursors in buyable stock."
            )
        else:
            assessment.notes.append(
                "AiZynthFinder did not find a solved route that keeps all precursors in stock within the configured step limit."
            )
        return assessment


def retrosynthesis_prompt_constraints() -> list[str]:
    config = resolve_retrosynthesis_config()
    return [
        "Experimental synthesis screen: prefer molecules that can plausibly be reached from buyable ligands rather than bespoke inaccessible precursors.",
        f"Experimental synthesis screen: favor candidates whose retrosynthesis to buyable ligands is likely to take no more than {config.max_steps} steps.",
        "When uncertain, choose more modular symmetric derivatives with commercially familiar aryl nitrile, fluoroaryl, aza-aryl, halide, boronic-acid, or carboxyaryl precursor families.",
    ]


def build_retrosyn_setup_commands(python_executable: str | None = None) -> list[list[str]]:
    python_bin = python_executable or sys.executable
    config = resolve_retrosynthesis_config()
    return [
        [
            python_bin,
            "-m",
            "organic_crystal_agent.cli",
            "retrosyn-prepare-stock",
            "--buyables-jsonl",
            config.buyables_jsonl,
            "--output-smiles",
            config.buyables_smiles,
            "--output-stock-hdf5",
            config.stock_hdf5,
            "--output-config",
            config.aizynth_config,
            "--max-steps",
            str(config.max_steps),
        ]
    ]


def _select_best_route(routes: list[dict]) -> dict | None:
    if not routes:
        return None
    ranked = sorted(
        routes,
        key=lambda route: (
            not bool(route.get("metadata", {}).get("is_solved", False)),
            _safe_int(route.get("scores", {}).get("number of reactions"), default=999),
            -float(route.get("scores", {}).get("state score", 0.0) or 0.0),
        ),
    )
    return ranked[0]


def _leaf_precursors(node: dict) -> set[str]:
    if node.get("type") == "mol" and not node.get("children"):
        smiles = str(node.get("smiles", "")).strip()
        return {smiles} if smiles else set()
    precursors: set[str] = set()
    for child in node.get("children", []) or []:
        precursors.update(_leaf_precursors(child))
    return precursors


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default
