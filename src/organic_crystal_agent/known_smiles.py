from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_KNOWN_SMILES_CSV = (
    "/users/ffmmgg/scratch/AlphaPack_llm/AlphaPackllm/data/CSD_smiles/smiles_properties_all.csv"
)


def normalize_canonical_smiles(smiles: str) -> str:
    return smiles.strip()


@lru_cache(maxsize=4)
def load_known_smiles_set(
    csv_path: str = DEFAULT_KNOWN_SMILES_CSV,
    smiles_column: str = "smiles",
) -> set[str]:
    path = Path(os.path.expanduser(csv_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Known SMILES CSV not found: {path}")

    known: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if smiles_column not in (reader.fieldnames or []):
            raise KeyError(f"Column {smiles_column!r} not found in {path}")
        for row in reader:
            value = normalize_canonical_smiles(row.get(smiles_column, ""))
            if value:
                known.add(value)
    return known


def is_known_canonical_smiles(smiles: str, csv_path: str = DEFAULT_KNOWN_SMILES_CSV) -> bool:
    normalized = normalize_canonical_smiles(smiles)
    if not normalized:
        return False
    return normalized in load_known_smiles_set(csv_path)


def ensure_novel_canonical_smiles(smiles: str, csv_path: str = DEFAULT_KNOWN_SMILES_CSV) -> str:
    normalized = normalize_canonical_smiles(smiles)
    if not normalized:
        return normalized
    if is_known_canonical_smiles(normalized, csv_path=csv_path):
        raise ValueError(
            "Candidate canonical SMILES already exists in the known molecule registry: "
            f"{normalized}"
        )
    return normalized
