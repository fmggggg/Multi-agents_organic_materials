from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

PrefixMode = Literal["none", "first_n", "delim_k"]


def resolve_cif(name: str, cif_dir: str | Path) -> Path | None:
    source = Path(cif_dir)
    exact = [path for path in source.glob("*.cif") if path.stem == name]
    if exact:
        return sorted(exact)[0]
    candidates = sorted(source.glob(f"{name}*.cif"))
    return candidates[0] if candidates else None


def get_prefix(name: str, mode: PrefixMode, n: int, delim: str) -> str:
    if mode == "none":
        return "all"
    if mode == "first_n":
        return name[:n] if n > 0 else name
    parts = name.split(delim)
    return delim.join(parts[: max(1, n)]) if parts else name


def ensure_amd_available() -> object:
    try:
        module = importlib.import_module("amd")
        if all(hasattr(module, attr) for attr in ("CifReader", "AMD", "AMD_pdist")):
            return module
        raise ImportError("Imported 'amd' module does not expose crystal-dedup APIs.")
    except ImportError:
        if os.environ.get("ORGANIC_CRYSTAL_AGENT_AUTO_INSTALL_AMD", "1") == "0":
            raise RuntimeError(
                "The 'amd' package is required for CSP deduplication and auto-install is disabled."
            ) from None
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "average-minimum-distance", "pandas<3"],
            check=True,
            capture_output=True,
            text=True,
        )
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("amd")
            if all(hasattr(module, attr) for attr in ("CifReader", "AMD", "AMD_pdist")):
                return module
            raise ImportError("Imported 'amd' module does not expose crystal-dedup APIs.")
        except ImportError as exc:  # pragma: no cover - install failure edge case
            raise RuntimeError(
                "Failed to import the crystal-dedup 'amd' module after automatic installation."
            ) from exc


def _load_structure_vector(path: Path, k: int, amd_module: object) -> object:
    crystals = list(amd_module.CifReader(str(path)))
    if not crystals:
        raise RuntimeError(f"No crystal found in {path}")
    return amd_module.AMD(crystals[0], k)


def _pairwise_distance(left: object, right: object, metric: str, amd_module: object) -> float:
    values = amd_module.AMD_pdist([left, right], metric=metric, low_memory=False)
    return float(values[0])


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def deduplicate_landscape(
    csv_path: str | Path,
    cif_dir: str | Path,
    out_dir: str | Path,
    *,
    name_col: str = "name",
    energy_col: str = "energy_kj",
    step_col: str = "step_used_2",
    fail_step_value: int | None = None,
    prefix_mode: PrefixMode = "delim_k",
    prefix_n: int = 1,
    prefix_delim: str = "_",
    k: int = 150,
    metric: str = "chebyshev",
    threshold: float = 0.05,
    energy_abs_max: float = 1.0e12,
) -> dict:
    csv_source = Path(csv_path)
    cif_source = Path(cif_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    amd_module = ensure_amd_available()

    dedup_csv = output_root / "optimized_landscape_dedup.csv"
    cluster_csv = output_root / "dedup_clusters.csv"
    merged_cif = output_root / "merged_optimized_dedup.cif"
    summary_path = output_root / "dedup_summary.json"

    if not csv_source.exists() or not cif_source.exists():
        summary = {
            "dedup_performed": False,
            "dedup_method": "unavailable",
            "kept_rows": 0,
            "cluster_count": 0,
            "missing_cif_count": 0,
            "dedup_csv": "",
            "cluster_csv": "",
            "merged_cif": "",
            "notes": ["Dedup skipped because optimized landscape CSV or CIF directory was missing."],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    with csv_source.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    if not rows or name_col not in fieldnames or energy_col not in fieldnames:
        summary = {
            "dedup_performed": False,
            "dedup_method": "unavailable",
            "kept_rows": 0,
            "cluster_count": 0,
            "missing_cif_count": 0,
            "dedup_csv": "",
            "cluster_csv": "",
            "merged_cif": "",
            "notes": ["Dedup skipped because required columns were missing from optimized landscape CSV."],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    usable_rows: list[dict[str, str]] = []
    excluded_rows = 0
    for row in rows:
        try:
            energy = float(row.get(energy_col, ""))
        except (TypeError, ValueError):
            excluded_rows += 1
            continue
        if not (energy == energy and abs(energy) <= energy_abs_max):
            excluded_rows += 1
            continue
        if fail_step_value is not None and step_col in row and row.get(step_col) not in (None, ""):
            try:
                step_value = int(float(str(row.get(step_col))))
            except (TypeError, ValueError):
                step_value = None
            if step_value is not None and step_value >= fail_step_value:
                excluded_rows += 1
                continue
        usable_rows.append(row)
    for row in usable_rows:
        row["__prefix__"] = get_prefix(str(row[name_col]), prefix_mode, prefix_n, prefix_delim)

    kept_rows: list[dict[str, str]] = []
    cluster_rows: list[dict[str, str | int | float]] = []
    kept_cifs: list[Path] = []
    missing_cifs: list[str] = []
    cluster_index = 1
    dedup_method = "amd"

    for prefix in sorted({row["__prefix__"] for row in usable_rows}):
        prefix_rows = [row for row in usable_rows if row["__prefix__"] == prefix]
        vectors: list[object] = []
        enriched_rows: list[dict[str, str]] = []
        cif_paths: list[Path] = []
        for row in prefix_rows:
            cif_path = resolve_cif(str(row[name_col]), cif_source)
            if cif_path is None or not cif_path.exists():
                missing_cifs.append(str(row[name_col]))
                continue
            try:
                vectors.append(_load_structure_vector(cif_path, k, amd_module))
            except Exception:
                missing_cifs.append(str(row[name_col]))
                continue
            enriched_rows.append(row)
            cif_paths.append(cif_path)

        if not enriched_rows:
            continue

        uf = UnionFind(len(enriched_rows))
        for i in range(len(vectors) - 1):
            for j in range(i + 1, len(vectors)):
                if _pairwise_distance(vectors[i], vectors[j], metric, amd_module) <= threshold:
                    uf.union(i, j)

        groups: dict[int, list[int]] = {}
        for index in range(len(enriched_rows)):
            groups.setdefault(uf.find(index), []).append(index)

        for members in groups.values():
            group_rows = [enriched_rows[idx] for idx in members]
            best_row = min(
                group_rows,
                key=lambda row: (
                    float(row.get(energy_col, "inf") or "inf"),
                    str(row.get(name_col, "")),
                ),
            )
            kept_rows.append(best_row)
            kept_path = resolve_cif(str(best_row[name_col]), cif_source)
            if kept_path is not None:
                kept_cifs.append(kept_path)
            cluster_rows.append(
                {
                    "cluster_id": f"cluster-{cluster_index:04d}",
                    "prefix": prefix,
                    "cluster_size": len(group_rows),
                    "kept_name": str(best_row[name_col]),
                    "kept_energy_kj": float(best_row.get(energy_col, "nan") or "nan"),
                    "members": ";".join(str(row[name_col]) for row in sorted(group_rows, key=lambda item: item[name_col])),
                }
            )
            cluster_index += 1

    kept_rows = sorted(
        kept_rows,
        key=lambda row: (
            row.get("__prefix__", ""),
            float(row.get(energy_col, "inf") or "inf"),
            str(row.get(name_col, "")),
        ),
    )
    clean_fieldnames = [field for field in fieldnames if field != "__prefix__"]
    with dedup_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=clean_fieldnames)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({field: row.get(field, "") for field in clean_fieldnames})

    with cluster_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["cluster_id", "prefix", "cluster_size", "kept_name", "kept_energy_kj", "members"],
        )
        writer.writeheader()
        writer.writerows(cluster_rows)

    with merged_cif.open("w", encoding="utf-8") as fh:
        for path in kept_cifs:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            fh.write(text)
            fh.write("\n\n")

    summary = {
        "dedup_performed": True,
        "dedup_method": dedup_method,
        "kept_rows": len(kept_rows),
        "cluster_count": len(cluster_rows),
        "missing_cif_count": len(missing_cifs),
        "excluded_rows": excluded_rows,
        "dedup_csv": str(dedup_csv),
        "cluster_csv": str(cluster_csv),
        "merged_cif": str(merged_cif),
        "notes": [
            "AMD-based deduplication applied with AlphaPack-style thresholding."
        ],
    }
    if missing_cifs:
        summary["notes"].append(
            f"{len(missing_cifs)} optimized structures were skipped during dedup because CIF loading failed."
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
