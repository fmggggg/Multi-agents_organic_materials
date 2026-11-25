from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from organic_crystal_agent.postprocess import (
    _identify_landscape_spikes,
    _load_landscape_points,
    render_svg_to_png,
)


class LandscapeVisionPoint(BaseModel):
    name: str
    density: float
    relative_energy_kj: float


class LandscapeStructureSnippet(BaseModel):
    name: str
    cif_path: str
    cif_excerpt: str


class SpikeFamilyFinding(BaseModel):
    family_id: str
    representative_point: str
    point_names: list[str] = Field(default_factory=list)
    worth_attention: bool = True
    attention_reason: str = ""
    hydrogen_bond_mode: str = ""
    visual_rationale: str = ""
    relative_energy_kj_min: float | None = None
    relative_energy_kj_max: float | None = None


class LandscapeVisionReport(BaseModel):
    model: str
    png_path: str
    csv_path: str
    candidate_points: list[LandscapeVisionPoint] = Field(default_factory=list)
    structure_snippets: list[LandscapeStructureSnippet] = Field(default_factory=list)
    summary: str = ""
    spikes_worth_attention: list[str] = Field(default_factory=list)
    spike_families: list[SpikeFamilyFinding] = Field(default_factory=list)
    raw_response_text: str = ""


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _encode_png_data_url(png_path: str | Path) -> str:
    payload = Path(png_path).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resolve_png_path(png_path: str | Path) -> Path:
    target = Path(png_path).expanduser().resolve()
    if target.exists():
        return target

    if target.suffix.lower() == ".png":
        sibling_svg = target.with_suffix(".svg")
        if sibling_svg.exists() and render_svg_to_png(sibling_svg, target):
            return target

    raise FileNotFoundError(f"Missing PNG path: {target}")


def build_candidate_points(
    csv_path: str | Path,
    *,
    max_relative_energy_kj: float = 100.0,
) -> list[LandscapeVisionPoint]:
    points = _load_landscape_points(csv_path)
    if not points:
        return []
    min_energy = min(float(point["energy_kj"]) for point in points)
    spike_indices = _identify_landscape_spikes(
        points,
        max_relative_energy_kj=max_relative_energy_kj,
    )
    candidates: list[LandscapeVisionPoint] = []
    for idx in sorted(
        spike_indices,
        key=lambda item: (
            float(points[item]["density"]),
            float(points[item]["energy_kj"]) - min_energy,
        ),
    ):
        point = points[idx]
        candidates.append(
            LandscapeVisionPoint(
                name=str(point["name"]),
                density=float(point["density"]),
                relative_energy_kj=float(point["energy_kj"]) - min_energy,
            )
        )
    return candidates


def collect_structure_snippets(
    cif_dir: str | Path | None,
    point_names: list[str],
    *,
    max_chars_per_cif: int = 4000,
) -> list[LandscapeStructureSnippet]:
    if not cif_dir:
        return []
    root = Path(cif_dir)
    if not root.exists():
        return []
    snippets: list[LandscapeStructureSnippet] = []
    for name in point_names:
        candidates = [
            root / f"{name}_opt.cif",
            root / f"{name}.cif",
        ]
        cif_path = next((path for path in candidates if path.exists()), None)
        if cif_path is None:
            continue
        excerpt = cif_path.read_text(encoding="utf-8", errors="ignore")[:max_chars_per_cif]
        snippets.append(
            LandscapeStructureSnippet(
                name=name,
                cif_path=str(cif_path.resolve()),
                cif_excerpt=excerpt,
            )
        )
    return snippets


def _build_prompt(
    *,
    molecule_label: str | None,
    candidate_points: list[LandscapeVisionPoint],
    structure_snippets: list[LandscapeStructureSnippet],
    max_relative_energy_kj: float,
) -> str:
    point_payload = [point.model_dump() for point in candidate_points]
    snippet_payload = [snippet.model_dump() for snippet in structure_snippets]
    label_line = f"Molecule/job label: {molecule_label}\n" if molecule_label else ""
    return (
        "You are analyzing a crystal-energy landscape image for hydrogen-bonded organic framework discovery.\n"
        "The image already shows a density-vs-relative-energy landscape with named candidate spike points highlighted from a Pareto-front density-clustering detector.\n"
        "Your job is to use the image first, then the structured point list and CIF excerpts, to decide which labeled points belong to the same spike family.\n\n"
        "Decision focus:\n"
        "- A spike family can include several nearby highlighted points if they appear to belong to one visually coherent downward spike region.\n"
        "- Worth-attention spikes are the ones that look experimentally plausible and sit relatively close to the low-energy region, but the window is broad for large molecules.\n"
        "- For each family, infer the likely hydrogen-bond mode from the CIF excerpts if possible. If uncertain, say so clearly.\n"
        "- Do not just restate the Pareto front. Group points into families based on the visual landscape shape.\n\n"
        "Return strict JSON with keys:\n"
        "summary: string\n"
        "spikes_worth_attention: list of representative point names\n"
        "spike_families: list of objects with keys "
        "[family_id, representative_point, point_names, worth_attention, attention_reason, hydrogen_bond_mode, visual_rationale, relative_energy_kj_min, relative_energy_kj_max]\n\n"
        f"{label_line}"
        f"Candidate points were pre-filtered from the landscape using a Pareto-front screen within ΔE <= {max_relative_energy_kj:.1f} kJ/mol.\n"
        f"Candidate points JSON:\n{json.dumps(point_payload, indent=2)}\n\n"
        f"CIF excerpts JSON:\n{json.dumps(snippet_payload, indent=2)}\n"
    )


def analyze_landscape_with_openai(
    *,
    png_path: str | Path,
    csv_path: str | Path,
    cif_dir: str | Path | None = None,
    molecule_label: str | None = None,
    model: str | None = None,
    max_relative_energy_kj: float = 100.0,
    max_chars_per_cif: int = 4000,
) -> LandscapeVisionReport:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Landscape vision analysis requires `openai` to be installed.") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Landscape vision analysis requires OPENAI_API_KEY.")

    png = _resolve_png_path(png_path)
    csv_file = Path(csv_path).expanduser().resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing CSV path: {csv_file}")

    chosen_model = model or os.getenv("ORGANIC_CRYSTAL_AGENT_OPENAI_VISION_MODEL", "gpt-5.4")
    candidate_points = build_candidate_points(csv_file, max_relative_energy_kj=max_relative_energy_kj)
    structure_snippets = collect_structure_snippets(
        cif_dir,
        [point.name for point in candidate_points],
        max_chars_per_cif=max_chars_per_cif,
    )
    prompt = _build_prompt(
        molecule_label=molecule_label,
        candidate_points=candidate_points,
        structure_snippets=structure_snippets,
        max_relative_energy_kj=max_relative_energy_kj,
    )

    client = OpenAI()
    response = client.responses.create(
        model=chosen_model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_png_data_url(png)},
                ],
            }
        ],
    )
    raw_text = getattr(response, "output_text", "") or ""
    payload = json.loads(_strip_json_fence(raw_text))
    return LandscapeVisionReport(
        model=chosen_model,
        png_path=str(png),
        csv_path=str(csv_file),
        candidate_points=candidate_points,
        structure_snippets=structure_snippets,
        summary=str(payload.get("summary", "")),
        spikes_worth_attention=[str(item) for item in payload.get("spikes_worth_attention", [])],
        spike_families=[
            SpikeFamilyFinding.model_validate(item)
            for item in payload.get("spike_families", [])
        ],
        raw_response_text=raw_text,
    )
