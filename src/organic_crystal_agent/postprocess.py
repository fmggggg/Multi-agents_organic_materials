from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tarfile
import zipfile
from html import escape
from pathlib import Path
from typing import Any

from organic_crystal_agent.dedup import deduplicate_landscape


def concatenate_cif_files(input_dir: str | Path, output_path: str | Path) -> int:
    source = Path(input_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cif_paths = sorted(source.glob("*.cif")) if source.exists() else []
    count = 0
    with target.open("w", encoding="utf-8") as out:
        for path in cif_paths:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            count += 1
    return count


def collect_json_rows(json_dir: str | Path, csv_path: str | Path) -> int:
    source = Path(json_dir)
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in sorted(source.glob("*.json")) if source.exists() else []:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def _parse_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_landscape_row_failure(
    row: dict[str, Any],
    *,
    fail_step_value: int | None,
    step_col: str = "step_used_2",
    energy_col: str = "energy_kj",
    energy_abs_max: float = 1.0e12,
) -> str | None:
    energy = _parse_float(row.get(energy_col, row.get("energy_kj_mol")))
    if energy is None or not math.isfinite(energy):
        return "invalid_energy"
    if abs(energy) > energy_abs_max:
        return "energy_outlier"
    if fail_step_value is not None:
        steps = _parse_int(row.get(step_col))
        if steps is not None and steps >= fail_step_value:
            return "step_limit_reached"
    return None


def _find_window_gap_cut(
    rows: list[dict[str, Any]],
    *,
    value_getter: callable,
    window_size: int,
    gap_threshold: float,
    side: str,
) -> int | None:
    if len(rows) < 2:
        return None
    if side not in {"low", "high"}:
        raise ValueError(f"Unsupported side: {side}")

    if side == "low":
        start = 0
        end = min(len(rows), window_size)
        window = rows[start:end]
        offset = 0
    else:
        start = max(0, len(rows) - window_size)
        end = len(rows)
        window = rows[start:end]
        offset = start

    best_cut: int | None = None
    best_gap = gap_threshold
    for idx in range(len(window) - 1):
        left = value_getter(window[idx])
        right = value_getter(window[idx + 1])
        if left is None or right is None:
            continue
        gap = right - left
        if gap > best_gap:
            best_gap = gap
            best_cut = offset + idx
    return best_cut


def trim_landscape_band(
    rows: list[dict[str, Any]],
    *,
    value_getter: callable,
    sort_key: callable,
    window_size: int = 10,
    gap_threshold: float = 50.0,
    low_failure_reason: str,
    high_failure_reason: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sortable: list[dict[str, Any]] = []
    for row in rows:
        value = value_getter(row)
        if value is None or not math.isfinite(value):
            continue
        sortable.append(row)
    sortable.sort(key=sort_key)

    if len(sortable) < 2:
        return sortable, []

    excluded_rows: list[dict[str, Any]] = []
    kept = sortable

    while True:
        changed = False
        low_cut = _find_window_gap_cut(
            kept,
            value_getter=value_getter,
            window_size=window_size,
            gap_threshold=gap_threshold,
            side="low",
        )
        if low_cut is not None:
            excluded_rows.extend(
                dict(row, failure_reason=low_failure_reason) for row in kept[: low_cut + 1]
            )
            kept = kept[low_cut + 1 :]
            changed = True

        high_cut = _find_window_gap_cut(
            kept,
            value_getter=value_getter,
            window_size=window_size,
            gap_threshold=gap_threshold,
            side="high",
        )
        if high_cut is not None:
            excluded_rows.extend(
                dict(row, failure_reason=high_failure_reason) for row in kept[high_cut + 1 :]
            )
            kept = kept[: high_cut + 1]
            changed = True

        if not changed:
            break

    return kept, excluded_rows


def trim_landscape_energy_band(
    rows: list[dict[str, Any]],
    *,
    energy_col: str = "energy_kj",
    window_size: int = 10,
    gap_threshold_kj: float = 50.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return trim_landscape_band(
        rows,
        value_getter=lambda row: _parse_float(row.get(energy_col, row.get("energy_kj_mol"))),
        sort_key=lambda row: float(row.get(energy_col, row.get("energy_kj_mol"))),
        window_size=window_size,
        gap_threshold=gap_threshold_kj,
        low_failure_reason="energy_gap_low_tail",
        high_failure_reason="energy_gap_high_tail",
    )


def trim_landscape_density_band(
    rows: list[dict[str, Any]],
    *,
    density_col: str = "density",
    window_size: int = 10,
    gap_threshold_density: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return trim_landscape_band(
        rows,
        value_getter=lambda row: _parse_float(row.get(density_col)),
        sort_key=lambda row: float(row.get(density_col)),
        window_size=window_size,
        gap_threshold=gap_threshold_density,
        low_failure_reason="density_gap_low_tail",
        high_failure_reason="density_gap_high_tail",
    )


def filter_landscape_csv(
    csv_path: str | Path,
    excluded_csv_path: str | Path,
    *,
    fail_step_value: int | None,
    step_col: str = "step_used_2",
    energy_col: str = "energy_kj",
    energy_abs_max: float = 1.0e12,
    band_window_size: int = 10,
    band_gap_threshold_kj: float = 50.0,
    density_col: str = "density",
    density_band_window_size: int = 10,
    density_gap_threshold: float = 0.2,
) -> dict[str, int]:
    source = Path(csv_path)
    excluded = Path(excluded_csv_path)
    if not source.exists():
        return {"kept_rows": 0, "excluded_rows": 0}

    with source.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    kept_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        failure_reason = classify_landscape_row_failure(
            row,
            fail_step_value=fail_step_value,
            step_col=step_col,
            energy_col=energy_col,
            energy_abs_max=energy_abs_max,
        )
        if failure_reason is None:
            kept_rows.append(row)
        else:
            excluded_row = dict(row)
            excluded_row["failure_reason"] = failure_reason
            excluded_rows.append(excluded_row)

    kept_rows, band_excluded_rows = trim_landscape_energy_band(
        kept_rows,
        energy_col=energy_col,
        window_size=band_window_size,
        gap_threshold_kj=band_gap_threshold_kj,
    )
    excluded_rows.extend(band_excluded_rows)

    kept_rows, density_excluded_rows = trim_landscape_density_band(
        kept_rows,
        density_col=density_col,
        window_size=density_band_window_size,
        gap_threshold_density=density_gap_threshold,
    )
    excluded_rows.extend(density_excluded_rows)

    with source.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(kept_rows)

    excluded.parent.mkdir(parents=True, exist_ok=True)
    excluded_fieldnames = fieldnames + ["failure_reason"]
    with excluded.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=excluded_fieldnames)
        if excluded_fieldnames:
            writer.writeheader()
            writer.writerows(excluded_rows)

    return {
        "kept_rows": len(kept_rows),
        "excluded_rows": len(excluded_rows),
        "band_window_size": band_window_size,
        "band_gap_threshold_kj": band_gap_threshold_kj,
        "density_band_window_size": density_band_window_size,
        "density_gap_threshold": density_gap_threshold,
    }


def archive_directory_contents(source_dir: str | Path, archive_path: str | Path) -> int:
    source = Path(source_dir)
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(path for path in source.rglob("*") if path.is_file()) if source.exists() else []
    with tarfile.open(target, "w:gz") as tar:
        for path in file_paths:
            tar.add(path, arcname=path.relative_to(source))
    return len(file_paths)


def archive_directory_contents_zip(source_dir: str | Path, archive_path: str | Path) -> int:
    source = Path(source_dir)
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(path for path in source.rglob("*") if path.is_file()) if source.exists() else []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            zf.write(path, arcname=path.relative_to(source))
    return len(file_paths)


def prune_directory_files(source_dir: str | Path, keep_names: set[str] | None = None) -> int:
    source = Path(source_dir)
    keep_names = keep_names or set()
    removed = 0
    if not source.exists():
        return removed
    for path in sorted(source.rglob("*"), reverse=True):
        if not path.exists():
            continue
        if path.is_file() and path.name not in keep_names:
            path.unlink()
            removed += 1
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def render_svg_to_png(svg_path: str | Path, png_path: str | Path) -> bool:
    source = Path(svg_path)
    target = Path(png_path)
    if not source.exists():
        return False
    converter = shutil.which("rsvg-convert")
    if not converter:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [converter, "-o", str(target), str(source)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False
    return target.exists()


def _merge_selected_cifs_from_csv(
    dedup_csv_path: str | Path,
    cif_dir: str | Path,
    output_path: str | Path,
) -> int:
    csv_source = Path(dedup_csv_path)
    cif_source = Path(cif_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not csv_source.exists() or not cif_source.exists():
        return 0

    with csv_source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    count = 0
    with target.open("w", encoding="utf-8") as out:
        for row in rows:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            candidates = [
                cif_source / f"{name}_opt.cif",
                cif_source / f"{name}.cif",
            ]
            cif_path = next((path for path in candidates if path.exists()), None)
            if cif_path is None:
                matches = sorted(cif_source.glob(f"{name}*.cif"))
                cif_path = matches[0] if matches else None
            if cif_path is None or not cif_path.exists():
                continue
            text = cif_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            count += 1
    return count


def run_refined_landscape_dedup(
    landscape_csv: str | Path,
    cif_dir: str | Path,
    out_dir: str | Path,
    *,
    dynamic_thresholds: str = "0.5,0.3,0.2,0.1,0.05",
    max_coarse_group_size: int = 20,
    min_dynamic_threshold: float = 0.001,
    metric: str = "chebyshev",
    k: int = 150,
    packing_shell_size: int = 15,
    min_matched: int = 15,
    rmsd_cutoff: float = 0.5,
) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "refined_landscape_dedup.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Refined dedup script not found: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--landscape-csv",
        str(Path(landscape_csv)),
        "--cif-dir",
        str(Path(cif_dir)),
        "--outdir",
        str(root),
        "--dynamic-thresholds",
        dynamic_thresholds,
        "--max-coarse-group-size",
        str(max_coarse_group_size),
        "--min-dynamic-threshold",
        str(min_dynamic_threshold),
        "--metric",
        metric,
        "--k",
        str(k),
        "--engine",
        "ccdc",
        "--packing-shell-size",
        str(packing_shell_size),
        "--min-matched",
        str(min_matched),
        "--rmsd-cutoff",
        str(rmsd_cutoff),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    refined_csv = root / "optimized_landscape_refined_dedup.csv"
    refined_cluster_csv = root / "refined_dedup_clusters.csv"
    refined_summary_json = root / "refined_dedup_summary.json"
    coarse_summary_json = root / "coarse_dedup_summary.json"
    standard_csv = root / "optimized_landscape_dedup.csv"
    standard_cluster_csv = root / "dedup_clusters.csv"
    standard_summary_json = root / "dedup_summary.json"
    merged_cif = root / "merged_optimized_dedup.cif"

    if refined_csv.exists():
        shutil.copy2(refined_csv, standard_csv)
    if refined_cluster_csv.exists():
        shutil.copy2(refined_cluster_csv, standard_cluster_csv)
    if refined_summary_json.exists():
        shutil.copy2(refined_summary_json, standard_summary_json)

    merged_cif_count = _merge_selected_cifs_from_csv(standard_csv, cif_dir, merged_cif)

    summary: dict[str, Any] = {}
    if standard_summary_json.exists():
        summary = json.loads(standard_summary_json.read_text(encoding="utf-8"))
    summary["dedup_method"] = "refined_dynamic_amd_ccdc"
    summary["standard_dedup_csv"] = str(standard_csv) if standard_csv.exists() else ""
    summary["standard_cluster_csv"] = str(standard_cluster_csv) if standard_cluster_csv.exists() else ""
    summary["standard_summary_json"] = str(standard_summary_json) if standard_summary_json.exists() else ""
    summary["coarse_summary_json"] = str(coarse_summary_json) if coarse_summary_json.exists() else ""
    summary["merged_cif"] = str(merged_cif) if merged_cif.exists() else ""
    summary["merged_cif_count"] = merged_cif_count
    summary["stdout_log"] = completed.stdout[-4000:] if completed.stdout else ""
    summary["stderr_log"] = completed.stderr[-4000:] if completed.stderr else ""
    standard_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_landscape_points(csv_path: str | Path) -> list[dict[str, float | str]]:
    source = Path(csv_path)
    if not source.exists():
        return []
    points: list[dict[str, float | str]] = []
    with source.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                density = float(row.get("density", ""))
                energy = float(row.get("energy_kj", row.get("energy_kj_mol", "")))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(density) and math.isfinite(energy)):
                continue
            points.append(
                {
                    "name": str(row.get("name", "")),
                    "density": density,
                    "energy_kj": energy,
                }
            )
    return points


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _identify_landscape_spikes(
    points: list[dict[str, float | str]],
    *,
    max_relative_energy_kj: float = 50.0,
    density_cluster_window: float = 0.10,
) -> set[int]:
    if not points:
        return set()

    energies = [float(point["energy_kj"]) for point in points]
    min_energy = min(energies)
    relative = [energy - min_energy for energy in energies]
    points_sorted = sorted(range(len(points)), key=lambda idx: float(points[idx]["density"]))
    front: list[int] = []
    best_energy = math.inf
    for idx in points_sorted:
        energy = float(points[idx]["energy_kj"])
        if energy < best_energy - 1e-9:
            front.append(idx)
            best_energy = energy

    if not front:
        return set()

    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    for idx in front:
        density = float(points[idx]["density"])
        if not current_cluster:
            current_cluster = [idx]
            continue
        previous_density = float(points[current_cluster[-1]]["density"])
        if density - previous_density <= density_cluster_window:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    if current_cluster:
        clusters.append(current_cluster)

    min_spike_span_kj = 10.0
    representatives: set[int] = set()
    for cluster_index, cluster in enumerate(clusters):
        best_idx = min(cluster, key=lambda idx: float(points[idx]["energy_kj"]))
        if relative[best_idx] > max_relative_energy_kj:
            continue
        cluster_relative = [relative[idx] for idx in cluster]
        if len(cluster_relative) >= 2:
            spike_span_kj = max(cluster_relative) - min(cluster_relative)
            if spike_span_kj < min_spike_span_kj:
                continue
            representatives.add(best_idx)
            continue
        if cluster_index == 0:
            continue
        previous_floor = min(relative[idx] for idx in clusters[cluster_index - 1])
        if previous_floor - relative[best_idx] < min_spike_span_kj:
            continue
        representatives.add(best_idx)
    return representatives


def load_spike_representative_points(
    csv_path: str | Path,
    *,
    max_relative_energy_kj: float = 50.0,
    density_cluster_window: float = 0.10,
) -> list[dict[str, float | str]]:
    points = _load_landscape_points(csv_path)
    if not points:
        return []
    min_energy = min(float(point["energy_kj"]) for point in points)
    indices = sorted(
        _identify_landscape_spikes(
            points,
            max_relative_energy_kj=max_relative_energy_kj,
            density_cluster_window=density_cluster_window,
        ),
        key=lambda idx: float(points[idx]["density"]),
    )
    representatives: list[dict[str, float | str]] = []
    for idx in indices:
        point = dict(points[idx])
        point["relative_energy_kj"] = float(point["energy_kj"]) - min_energy
        representatives.append(point)
    return representatives


def export_spike_representative_structures(
    job_dir: str | Path,
    *,
    max_relative_energy_kj: float = 50.0,
    density_cluster_window: float = 0.10,
    output_dir_name: str = "spikes",
    landscape_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy representative spike CIFs into a stable per-molecule folder.

    The adsorption workflow can recover spike structures from the archived relax
    output, but users need a stable, human-readable location in the CSP job
    directory. This exporter writes all representative spike structures, marks
    the highest-density representative that first-pass adsorption skips, and
    records the density/relative-energy metadata used for selection.
    """

    root = Path(job_dir).expanduser().resolve()
    if landscape_csv_path is not None:
        landscape_csv = Path(landscape_csv_path).expanduser().resolve()
    else:
        landscape_csv = root / "optimized_landscape_dedup.csv"
        if not landscape_csv.exists():
            landscape_csv = root / "optimized_landscape.csv"
    if not landscape_csv.exists():
        raise FileNotFoundError(f"No optimized landscape CSV found in {root}")

    representatives = load_spike_representative_points(
        landscape_csv,
        max_relative_energy_kj=max_relative_energy_kj,
        density_cluster_window=density_cluster_window,
    )
    representatives = sorted(
        representatives,
        key=lambda row: (float(row["density"]), float(row["relative_energy_kj"])),
    )

    spikes_dir = root / output_dir_name
    if spikes_dir.exists():
        for path in spikes_dir.iterdir():
            if path.is_file():
                path.unlink()
    spikes_dir.mkdir(parents=True, exist_ok=True)

    skipped_name = ""
    if representatives:
        skipped = max(
            representatives,
            key=lambda row: (float(row["density"]), -float(row["relative_energy_kj"])),
        )
        skipped_name = str(skipped["name"])

    records: list[dict[str, Any]] = []
    for index, row in enumerate(representatives, start=1):
        name = str(row["name"])
        cif_name = f"{name}_opt.cif"
        is_skipped = name == skipped_name
        prefix = "excluded_highest_density" if is_skipped else "selected"
        output_name = f"spike_{index:02d}_{prefix}_{cif_name}"
        output_path = spikes_dir / output_name
        source_path = _copy_cif_from_job_outputs(root, cif_name, output_path)
        records.append(
            {
                "rank_by_density": index,
                "name": name,
                "density": float(row["density"]),
                "relative_energy_kj": float(row["relative_energy_kj"]),
                "source_cif_name": cif_name,
                "source_path": str(source_path),
                "spike_cif_path": str(output_path),
                "selected_for_first_pass_adsorption": not is_skipped,
                "excluded_reason": (
                    "highest-density spike representative; skipped by first-pass adsorption"
                    if is_skipped
                    else ""
                ),
            }
        )

    manifest = {
        "job_dir": str(root),
        "landscape_csv": str(landscape_csv),
        "spikes_dir": str(spikes_dir),
        "max_relative_energy_kj": max_relative_energy_kj,
        "density_cluster_window": density_cluster_window,
        "spike_count": len(records),
        "records": records,
    }
    (spikes_dir / "spike_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (spikes_dir / "spike_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "rank_by_density",
            "name",
            "density",
            "relative_energy_kj",
            "source_cif_name",
            "spike_cif_path",
            "selected_for_first_pass_adsorption",
            "excluded_reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return manifest


def _copy_cif_from_job_outputs(root: Path, cif_name: str, output_path: Path) -> Path:
    candidates = [
        root / "relax" / "cif_result_final" / cif_name,
        root / "_restore_relax_artifacts" / "cif_result_final" / cif_name,
        root / "_adsorption_selected_cif" / cif_name,
    ]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, output_path)
            return source

    for archive_path in (root / "relax_artifacts.zip", root / "relax_artifacts.tar.gz"):
        if _extract_cif_from_archive(archive_path, cif_name, output_path):
            return archive_path

    raise FileNotFoundError(f"Could not locate spike CIF {cif_name} in {root}")


def _extract_cif_from_archive(archive_path: Path, cif_name: str, output_path: Path) -> bool:
    if not archive_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if member.endswith(cif_name):
                    with zf.open(member) as src, output_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
        return False
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith(cif_name):
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src, output_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
    return False


def write_landscape_svg(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    title: str = "Crystal Landscape",
) -> int:
    points = _load_landscape_points(csv_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        return 0

    min_energy = min(float(point["energy_kj"]) for point in points)
    densities = [float(point["density"]) for point in points]
    relative_energies = [float(point["energy_kj"]) - min_energy for point in points]

    representative_spike_indices = _identify_landscape_spikes(points)

    legend_entries_estimate = max(1, len(representative_spike_indices))
    legend_width = min(420, max(250, 16 * max(len(str(point["name"])) for point in points if point)))
    width = 1120 + legend_width
    height = 620
    left = 90
    right = 40 + legend_width
    top = 40
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    min_density = min(densities)
    max_density = max(densities)
    min_rel_energy = 0.0
    max_rel_energy = max(relative_energies)

    density_span = max(max_density - min_density, 1e-9)
    energy_span = max(max_rel_energy - min_rel_energy, 1.0)
    density_pad = max(density_span * 0.05, 0.02)
    energy_pad = max(energy_span * 0.08, 1.0)
    x_min = min_density - density_pad
    x_max = max_density + density_pad
    y_min = 0.0
    y_max = max_rel_energy + energy_pad

    def map_x(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_width

    def map_y(value: float) -> float:
        return top + plot_height - (value - y_min) / max(y_max - y_min, 1e-9) * plot_height

    x_ticks = 5
    y_ticks = 5
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="20" font-family="Arial">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="2"/>',
    ]

    for i in range(x_ticks + 1):
        frac = i / x_ticks
        value = x_min + frac * (x_max - x_min)
        x = left + frac * plot_width
        svg_lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial">{value:.2f}</text>'
        )

    for i in range(y_ticks + 1):
        frac = i / y_ticks
        value = y_min + frac * (y_max - y_min)
        y = top + plot_height - frac * plot_height
        svg_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{value:.1f}</text>'
        )

    svg_lines.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="14" font-family="Arial">Density</text>',
            (
                f'<text x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="14" '
                f'font-family="Arial" transform="rotate(-90 24 {top + plot_height / 2:.1f})">'
                "Relative energy (kJ/mol)"
                "</text>"
            ),
        ]
    )

    min_index = relative_energies.index(min(relative_energies))
    spike_palette = [
        "#d62839",
        "#f77f00",
        "#2a9d8f",
        "#7b2cbf",
        "#ff006e",
        "#6a994e",
        "#c1121f",
        "#8338ec",
    ]
    spike_order = sorted(
        representative_spike_indices,
        key=lambda idx: (
            float(points[idx]["density"]),
            float(points[idx]["energy_kj"]) - min_energy,
        ),
    )
    spike_color_map = {
        idx: spike_palette[i % len(spike_palette)]
        for i, idx in enumerate(spike_order)
    }
    if len(spike_order) >= 2:
        poly_points = " ".join(
            f"{map_x(float(points[idx]['density'])):.2f},{map_y(float(points[idx]['energy_kj']) - min_energy):.2f}"
            for idx in spike_order
        )
        svg_lines.append(
            f'<polyline points="{poly_points}" fill="none" stroke="#444444" '
            f'stroke-width="1.8" stroke-dasharray="6 4" opacity="0.9"/>'
        )
    legend_entries: list[tuple[str, str]] = []
    for idx, point in enumerate(points):
        x = map_x(float(point["density"]))
        y = map_y(float(point["energy_kj"]) - min_energy)
        is_spike = idx in representative_spike_indices
        radius = 6 if idx == min_index else (5 if is_spike else 4)
        fill = spike_color_map[idx] if is_spike else "#9aa0a6"
        label = escape(str(point["name"]))
        svg_lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}">'
            f'<title>{label} | density={float(point["density"]):.4f} | ΔE={float(point["energy_kj"]) - min_energy:.3f} kJ/mol</title>'
            "</circle>"
        )
        if is_spike:
            color = spike_color_map[idx]
            legend_entries.append((color, label))

    if legend_entries:
        box_x = left + plot_width + 22
        box_y = top + 8
        box_w = legend_width
        box_h = 18 + 18 * len(legend_entries)
        svg_lines.append(
            f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            f'fill="white" fill-opacity="0.88" stroke="#cccccc" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{box_x + 10:.1f}" y="{box_y + 16:.1f}" font-size="12" font-family="Arial" '
            f'font-weight="bold" fill="#333">Pareto-front spike representatives (ΔE ≤ 50 kJ/mol)</text>'
        )
        for i, (color, label) in enumerate(legend_entries):
            y = box_y + 31 + i * 18
            svg_lines.append(
                f'<line x1="{box_x + 10:.1f}" y1="{y - 4:.1f}" x2="{box_x + 24:.1f}" y2="{y - 4:.1f}" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            svg_lines.append(
                f'<circle cx="{box_x + 17:.1f}" cy="{y - 4:.1f}" r="4" fill="{color}"/>'
            )
            svg_lines.append(
                f'<text x="{box_x + 30:.1f}" y="{y:.1f}" text-anchor="start" '
                f'font-size="12" font-family="Arial" font-weight="bold" fill="{color}">{label}</text>'
            )

    svg_lines.append("</svg>")
    target.write_text("\n".join(svg_lines), encoding="utf-8")
    return len(points)


def postprocess_job_dir(job_dir: str | Path) -> dict:
    root = Path(job_dir)
    manifest_path = root / "job_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing job manifest in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sample_dir = Path(manifest["local_sample_dir"])
    relax_dir = Path(manifest["relax_dir"])
    pre_relax_dir = Path(manifest.get("pre_relax_dir", root / "pre_relax"))
    json_dir = relax_dir / "json_result"
    final_dir = relax_dir / "cif_result_final"
    press_dir = relax_dir / "cif_result_press"

    merged_samples_cif = root / "merged_samples.cif"
    merged_optimized_cif = root / "merged_optimized.cif"
    merged_optimized_dedup_cif = root / "merged_optimized_dedup.cif"
    optimized_landscape_csv = root / "optimized_landscape.csv"
    optimized_landscape_dedup_csv = root / "optimized_landscape_dedup.csv"
    optimized_landscape_excluded_csv = root / "optimized_landscape_excluded.csv"
    landscape_svg = root / "landscape_energy_density.svg"
    landscape_dedup_svg = root / "landscape_energy_density_dedup.svg"
    landscape_png = root / "landscape_energy_density.png"
    landscape_dedup_png = root / "landscape_energy_density_dedup.png"
    sample_archive = root / "sample_artifacts.tar.gz"
    relax_archive = root / "relax_artifacts.tar.gz"
    pre_relax_archive = root / "pre_relax_artifacts.tar.gz"
    sample_zip_archive = root / "sample_artifacts.zip"
    relax_zip_archive = root / "relax_artifacts.zip"
    pre_relax_zip_archive = root / "pre_relax_artifacts.zip"

    sample_cif_count = concatenate_cif_files(sample_dir, merged_samples_cif)
    optimized_cif_count = concatenate_cif_files(final_dir, merged_optimized_cif)
    json_row_count = collect_json_rows(json_dir, optimized_landscape_csv)
    max_steps = _parse_int(manifest.get("settings", {}).get("max_steps"))
    landscape_filter_summary = filter_landscape_csv(
        optimized_landscape_csv,
        optimized_landscape_excluded_csv,
        fail_step_value=max_steps,
    )
    try:
        dedup_summary = run_refined_landscape_dedup(
            optimized_landscape_csv,
            final_dir,
            root,
        )
    except Exception as exc:
        dedup_summary = deduplicate_landscape(
            optimized_landscape_csv,
            final_dir,
            root,
            fail_step_value=max_steps,
        )
        dedup_summary["dedup_method"] = "amd_fallback"
        dedup_summary["fallback_reason"] = str(exc)
    landscape_plot_points = write_landscape_svg(
        optimized_landscape_csv,
        landscape_svg,
        title="Crystal Landscape (all optimized structures)",
    )
    dedup_landscape_plot_points = write_landscape_svg(
        optimized_landscape_dedup_csv,
        landscape_dedup_svg,
        title="Crystal Landscape (AMD deduplicated)",
    )
    render_svg_to_png(landscape_svg, landscape_png)
    render_svg_to_png(landscape_dedup_svg, landscape_dedup_png)
    spike_export_summary = export_spike_representative_structures(root)
    archived_sample_files = archive_directory_contents(sample_dir, sample_archive)
    archived_relax_files = archive_directory_contents(relax_dir, relax_archive)
    archived_pre_relax_files = archive_directory_contents(pre_relax_dir, pre_relax_archive)
    archive_directory_contents_zip(sample_dir, sample_zip_archive)
    archive_directory_contents_zip(relax_dir, relax_zip_archive)
    archive_directory_contents_zip(pre_relax_dir, pre_relax_zip_archive)

    prune_directory_files(sample_dir, keep_names={"all_samples.csv"})
    prune_directory_files(pre_relax_dir)
    prune_directory_files(final_dir)
    prune_directory_files(press_dir)
    prune_directory_files(json_dir)

    summary = {
        "merged_samples_cif": str(merged_samples_cif) if merged_samples_cif.exists() else "",
        "merged_optimized_cif": str(merged_optimized_cif) if merged_optimized_cif.exists() else "",
        "merged_optimized_dedup_cif": str(merged_optimized_dedup_cif) if merged_optimized_dedup_cif.exists() else "",
        "optimized_landscape_csv": str(optimized_landscape_csv) if optimized_landscape_csv.exists() else "",
        "optimized_landscape_excluded_csv": (
            str(optimized_landscape_excluded_csv) if optimized_landscape_excluded_csv.exists() else ""
        ),
        "optimized_landscape_dedup_csv": (
            str(optimized_landscape_dedup_csv) if optimized_landscape_dedup_csv.exists() else ""
        ),
        "landscape_svg": str(landscape_svg) if landscape_svg.exists() else "",
        "landscape_dedup_svg": str(landscape_dedup_svg) if landscape_dedup_svg.exists() else "",
        "landscape_png": str(landscape_png) if landscape_png.exists() else "",
        "landscape_dedup_png": str(landscape_dedup_png) if landscape_dedup_png.exists() else "",
        "sample_archive": str(sample_archive) if sample_archive.exists() else "",
        "relax_archive": str(relax_archive) if relax_archive.exists() else "",
        "pre_relax_archive": str(pre_relax_archive) if pre_relax_archive.exists() else "",
        "sample_zip_archive": str(sample_zip_archive) if sample_zip_archive.exists() else "",
        "relax_zip_archive": str(relax_zip_archive) if relax_zip_archive.exists() else "",
        "pre_relax_zip_archive": str(pre_relax_zip_archive) if pre_relax_zip_archive.exists() else "",
        "sample_cif_count": sample_cif_count,
        "optimized_cif_count": optimized_cif_count,
        "json_row_count": json_row_count,
        "landscape_filter_summary": landscape_filter_summary,
        "landscape_plot_points": landscape_plot_points,
        "landscape_dedup_plot_points": dedup_landscape_plot_points,
        "archived_sample_files": archived_sample_files,
        "archived_relax_files": archived_relax_files,
        "archived_pre_relax_files": archived_pre_relax_files,
        "dedup_summary": dedup_summary,
        "spike_export_summary": spike_export_summary,
    }
    (root / "postprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
