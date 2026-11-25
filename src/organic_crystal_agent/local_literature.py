from __future__ import annotations

import os
import re
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

from organic_crystal_agent.application_specs import build_application_spec
from organic_crystal_agent.schemas import DiscoveryCampaign, LiteratureHypothesis


@dataclass(frozen=True)
class LocalFragmentRecord:
    smiles: str
    co2_binding_energy: float
    h2o_binding_energy: float
    water_penalty: float
    selection_score: float
    families: tuple[str, ...]


@dataclass(frozen=True)
class LocalEvidenceCard:
    title: str
    summary: str
    evidence: tuple[str, ...]
    design_implications: tuple[str, ...]
    risk_flags: tuple[str, ...]
    confidence: float
    source_path: str


def get_literature_root() -> Path:
    configured = os.getenv("ORGANIC_CRYSTAL_AGENT_LITERATURE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "literature"


def resolve_local_literature_hypotheses(
    campaign: DiscoveryCampaign,
    *,
    max_hypotheses: int = 5,
) -> list[LiteratureHypothesis]:
    cards = local_evidence_cards(campaign)[:max_hypotheses]
    return [
        LiteratureHypothesis(
            hypothesis_id=f"lit-local-{index:03d}",
            title=card.title,
            summary=card.summary,
            evidence=list(card.evidence),
            design_implications=list(card.design_implications),
            risk_flags=list(card.risk_flags),
            confidence=card.confidence,
        )
        for index, card in enumerate(cards, start=1)
    ]


def local_evidence_cards(campaign: DiscoveryCampaign) -> list[LocalEvidenceCard]:
    spec = build_application_spec(campaign)
    if spec.application_mode != "humid_co2_capture":
        return []

    literature_root = get_literature_root()
    rules_path = literature_root / "humid_co2_fragment_rules.md"
    paper_path = literature_root / "Bottom-up computational design of shape-selective organic macrocycles for humid CO2 capture.pdf"
    supp_path = literature_root / "Article Supplementary information Bottom-up computational design of shapeselective organic macrocycles for humid CO2 capture.pdf"
    xlsx_path = literature_root / "SMILES and binding energy results of CHNCHOCHFCHNF datasets.xlsx"

    cards: list[LocalEvidenceCard] = []
    if rules_path.exists():
        cards.append(_build_rules_card(rules_path))
    if paper_path.exists():
        cards.append(_build_paper_card(paper_path))
    if supp_path.exists():
        cards.append(_build_supplement_card(supp_path))
    if xlsx_path.exists():
        cards.append(_build_fragment_dataset_card(xlsx_path))
    cards.extend(_build_hof_review_cards(literature_root))
    return cards


def render_open_access_literature(campaign: DiscoveryCampaign) -> str:
    lines = ["# Open Access Literature", "", f"Campaign: {campaign.campaign_id}", ""]
    cards = local_evidence_cards(campaign)
    if not cards:
        lines.append("No campaign-specific local literature bundle was detected.")
        return "\n".join(lines)

    for card in cards:
        lines.extend(
            [
                f"## {card.title}",
                card.summary,
                f"Source: {card.source_path}",
                f"Confidence: {card.confidence:.2f}",
                "",
                "Evidence:",
            ]
        )
        lines.extend(f"- {item}" for item in card.evidence)
        lines.extend(["", "Design implications:"])
        lines.extend(f"- {item}" for item in card.design_implications)
        if card.risk_flags:
            lines.extend(["", "Risk flags:"])
            lines.extend(f"- {item}" for item in card.risk_flags)
        lines.append("")

    fragments = ranked_local_fragments(campaign)
    if fragments:
        lines.extend(["## Ranked Fragment Families", ""])
        for family, rows in fragments.items():
            examples = ", ".join(row.smiles for row in rows[:3])
            best_score = rows[0].selection_score
            lines.append(f"- {family}: top score={best_score:.3f}; examples={examples}")
        lines.append("")
    return "\n".join(lines)


def ranked_local_fragments(
    campaign: DiscoveryCampaign,
    *,
    per_family: int = 5,
) -> dict[str, list[LocalFragmentRecord]]:
    spec = build_application_spec(campaign)
    if spec.application_mode != "humid_co2_capture":
        return {}

    xlsx_path = get_literature_root() / "SMILES and binding energy results of CHNCHOCHFCHNF datasets.xlsx"
    if not xlsx_path.exists():
        return {}
    return ranked_local_fragments_for_path(xlsx_path, per_family=per_family)


def build_local_proposal_blueprints(
    campaign: DiscoveryCampaign,
    *,
    max_molecules: int = 5,
) -> list[dict[str, object]]:
    """Legacy hook retained for compatibility.

    Humid-CO2 discovery now forbids deterministic whole-molecule template
    fallback. Molecules must come from the active AI proposal pathway.
    """

    _ = (campaign, max_molecules)
    return []


def local_fixed_core_scaffolds(campaign: DiscoveryCampaign) -> list[dict[str, object]]:
    spec = build_application_spec(campaign)
    if spec.application_mode != "humid_co2_capture":
        return []
    return [
        {
            "family": "tritopic-triazine-core",
            "label": "Triazine HOF core example",
            "parent_scaffold_smiles": "OC(=O)c1ccc(cc1)c2nc(nc(n2)c3ccc(cc3)C(O)=O)c4ccc(cc4)C(O)=O",
            "notes": [
                "One literature-aligned HOF tecton example; not an exclusive allowed scaffold.",
                "AI should decide whether to use this family or propose a different symmetric HOF-credible core.",
            ],
        },
        {
            "family": "tetratopic-tetraphenylmethane-core",
            "label": "Tetraphenylmethane HOF core example",
            "parent_scaffold_smiles": "OC(=O)c1ccc(cc1)C(c2ccc(cc2)C(O)=O)(c3ccc(cc3)C(O)=O)c4ccc(cc4)C(O)=O",
            "notes": [
                "One literature-aligned HOF tecton example; not an exclusive allowed scaffold.",
                "AI should decide whether to use this family or propose a different symmetric HOF-credible core.",
            ],
        },
        {
            "family": "triptycene-core",
            "label": "Triptycene HOF core example",
            "parent_scaffold_smiles": "C1=CC2=C(C=C1)C3C4=CC=CC=C4C2C5=CC=CC=C35",
            "notes": [
                "Triptycene is locally available in the lab and should be treated as an allowed rigid aromatic cavity-forming scaffold reference.",
                "AI should decide whether to derivatize this triptycene core into a HOF-credible tecton or choose a different scaffold family based on the literature evidence.",
            ],
        },
    ]


def ranked_local_fragments_for_path(path: Path, *, per_family: int = 5) -> dict[str, list[LocalFragmentRecord]]:
    grouped: dict[str, list[LocalFragmentRecord]] = defaultdict(list)
    for row in _load_fragment_rows(path):
        for family in row.families:
            grouped[family].append(row)
    for family, rows in grouped.items():
        rows.sort(key=lambda item: (-item.selection_score, item.water_penalty, item.smiles))
        grouped[family] = rows[:per_family]
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (-item[1][0].selection_score if item[1] else float("inf"), item[0]),
        )
    )


def _build_rules_card(path: Path) -> LocalEvidenceCard:
    return LocalEvidenceCard(
        title="Local humid CO2 fragment rules",
        summary=(
            "The campaign-local rules file already distills the humid CO2 paper, supplementary notes, and XLSX into an executable policy: "
            "prefer rigid aromatic pi-surfaces, nitriles, fluorinated aryls, and aza-aromatics; keep amidines as a control branch; "
            "deprioritize O-rich pore-lining groups; and rank candidates by humid CO2 performance instead of dense minima. "
            "Five-membered or pentagonal cavity motifs remain a literature-supported branch, but they should compete with fused aza-aromatics, broader polyaryl panels, and other rigid HOF-credible scaffold families rather than dominate the search."
        ),
        evidence=(
            "Prioritize rigid, contorted aromatic fragments with broad pi-surfaces rather than simply increasing polarity.",
            "Retain nitrile-bearing, fluorinated, and aza-aromatic fragment families as first-round exploration axes.",
            "Keep amidine and guanidine motifs as minority control experiments because they may overbind water or bind CO2 too specifically.",
            "Ignore the densest collapsed spike by default during first-pass humid CO2 ranking.",
            "Five-membered-ring or pentagonal cavity motifs are worth testing when they are compatible with HOF assembly and do not over-hydrophilize the pore lining, but they are not the only preferred branch.",
        ),
        design_implications=(
            "Reward broad aromatic pi-surfaces with distributed weak contacts, but do not keep recycling one narrow bent-clip topology.",
            "Use only enough hydrogen-bonding functionality to assemble the framework; do not fill the pore lining with O-rich groups.",
            "Keep five-membered heteroaromatic or pentagonal cavity-forming motifs as one branch among several HOF-credible tecton families.",
            "Treat humid CO2 property score as the primary winner criterion once adsorption results exist.",
        ),
        risk_flags=("collapse_on_activation",),
        confidence=0.93,
        source_path=str(path),
    )


def _build_paper_card(path: Path) -> LocalEvidenceCard:
    text = _pdf_to_text(path)
    extracted = _matching_snippets(
        text,
        [
            r"27,446 diverse molecular fragments",
            r"most molecules favour water binding",
            r"bent, clip-like aromatic molecules exhibit potential",
            r"multiple weak pi interactions",
        ],
    )
    evidence = tuple(extracted) if extracted else (
        "The local humid-CO2 paper screens tens of thousands of fragments against both CO2 and water.",
        "Most isolated fragments still favor water, so the best motifs are unusual and should be handled explicitly.",
        "Bent, clip-like rigid aromatics are highlighted as reverse-selectivity candidates under humid conditions.",
    )
    return LocalEvidenceCard(
        title="Humid CO2 macrocycle screening paper",
        summary=(
            "The local humid-CO2 paper argues that reverse selectivity does not come from maximally polar binding sites. "
            "Instead, rigid aromatic enclosures can give CO2 enough weak dispersion-rich contacts to compete with water. "
            "The paper supports shape-selective aromatic cavities, but it does not require every proposal to collapse onto one five-membered or pentagonal clip family."
        ),
        evidence=evidence,
        design_implications=(
            "Prefer aromatic pore environments that create multiple weak CO2 contacts over strongly hydrophilic pore-lining motifs.",
            "Look for multiple weak CO2 contacts rather than one localized sticky binding motif.",
            "Treat five-membered ring and pentagonal cavity motifs as one plausible route when they still support HOF assembly and scaffold diversity.",
            "Keep humid-CO2 and H2O competition explicit in the ranking loop.",
        ),
        risk_flags=("solvate_trap",),
        confidence=0.88,
        source_path=str(path),
    )


def _build_supplement_card(path: Path) -> LocalEvidenceCard:
    text = _pdf_to_text(path)
    extracted = _matching_snippets(
        text,
        [
            r"humidity-stable",
            r"reverse selectivity",
            r"CO2",
            r"water",
        ],
        limit=4,
    )
    evidence = tuple(extracted) if extracted else (
        "The supplementary material expands the fragment-screening logic and the humidity-stable experimental validation.",
        "The target condition is explicitly humid CO2 capture rather than dry-gas uptake.",
    )
    return LocalEvidenceCard(
        title="Humid CO2 supplementary design notes",
        summary=(
            "The supplementary file reinforces that humid CO2 capture must be treated as a mixed-adsorbate problem and that the paper's successful shapes are rigid aromatic cavity formers, not generic polar binders."
        ),
        evidence=evidence,
        design_implications=(
            "Keep water competition visible in every fragment and structure ranking step.",
            "Use supplementary conditions as the default reference for pressure and humidity assumptions.",
        ),
        risk_flags=(),
        confidence=0.79,
        source_path=str(path),
    )


def _build_fragment_dataset_card(path: Path) -> LocalEvidenceCard:
    grouped = ranked_local_fragments_for_path(path)
    evidence_lines: list[str] = []
    for family, members in list(grouped.items())[:4]:
        top = members[0]
        evidence_lines.append(
            f"{family}: top fragment {top.smiles} with CO2={top.co2_binding_energy:.4f} and H2O={top.h2o_binding_energy:.4f}."
        )
    if not evidence_lines:
        evidence_lines = ["The local XLSX exposes fragment-level CO2 and H2O binding energies that can be ranked directly."]
    return LocalEvidenceCard(
        title="Local fragment binding-energy spreadsheet",
        summary=(
            "The local XLSX is a directly usable structured evidence source for fragment selection. "
            "It supports nitrile-bearing aromatics, fluorinated aromatics, fused aromatics, and aza-aromatics as better humid-CO2 compromises than O-rich pore-lining motifs."
        ),
        evidence=tuple(evidence_lines),
        design_implications=(
            "Use the fragment table as the first filter before proposing whole molecules.",
            "Favor fragments with strong CO2 interaction but limited extra water penalty.",
            "Convert fragment families into whole molecules using HOF-credible symmetric tectons; triazine and tetraphenylmethane are examples, not exclusive allowed cores.",
        ),
        risk_flags=(),
        confidence=0.9,
        source_path=str(path),
    )


def _build_hof_review_cards(literature_root: Path) -> list[LocalEvidenceCard]:
    review_specs = [
        (
            "Hydrogen-bonded organic frameworks Chemistry and functions.pdf",
            "HOF chemistry and functions review",
            (
                "This review emphasizes that successful HOFs often rely on carefully balanced directional hydrogen bonds "
                "plus broad aromatic surfaces, rather than maximizing polar functionality everywhere. "
                "For humid CO2 design, that supports rigid aromatic tectons with enough carboxylic-acid or complementary "
                "donor/acceptor chemistry to assemble the framework while preserving pi-rich cavity walls."
            ),
            (
                "Bias proposals toward tectons that combine robust H-bonding motifs with extended aromatic pi-faces.",
                "Prefer framework-forming groups at the periphery while keeping pore-facing surfaces comparatively aromatic.",
                "Treat whole-tecton shape and assembly topology as first-class design variables, not only fragment scores.",
            ),
            [
                r"hydrogen-bonded organic frameworks",
                r"chemistry and functions",
                r"porous",
                r"gas",
                r"aromatic",
            ],
            0.8,
        ),
        (
            "Hydrogen-bonded organic framework Construction and gas separation application.pdf",
            "HOF construction and gas separation review",
            (
                "This review adds a framework-engineering perspective: gas separation in HOFs depends on how tecton geometry, "
                "packing frustration, and pore aperture are expressed in the assembled crystal, not just on isolated binding fragments. "
                "That pushes the generator to consider scaffold-level cavity shape and anti-interpenetration features."
            ),
            (
                "Favor tectons whose geometry can create accessible apertures and discourage dense interdigitation.",
                "Use gas-separation precedent to justify bent, contorted, or shape-persistent aromatic cores.",
                "Prefer candidates where H-bond assembly motifs and pore-shape control reinforce each other.",
            ),
            [
                r"gas separation",
                r"construction",
                r"porous",
                r"packing",
                r"hydrogen bond",
            ],
            0.78,
        ),
        (
            "Flexible hydrogen-bonded organic frameworks (HOFs) opportunities and challenges.pdf",
            "Flexible HOF opportunities and challenges review",
            (
                "This review is useful as a warning signal: flexibility can create interesting response behavior, but too much softness "
                "usually hurts pore retention and activation stability. For humid CO2 discovery, the main lesson is to pursue "
                "shape-persistent aromatic faces and controllable flexibility rather than floppy multi-rotor scaffolds."
            ),
            (
                "Penalize overly flexible, many-rotor tectons unless there is a strong cavity-shaping justification.",
                "Use rigid or contorted aromatic motifs to preserve porosity after activation.",
                "When introducing flexibility, keep it localized and mechanistically motivated.",
            ),
            [
                r"opportunities and challenges",
                r"flexible",
                r"activation",
                r"stability",
                r"porous",
            ],
            0.76,
        ),
        (
            "Multifunctional porous hydrogen-bonded organic framework materials.pdf",
            "Multifunctional porous HOF materials review",
            (
                "This review broadens the search space beyond one or two canonical cores: multifunctional porous HOFs can be built "
                "from diverse aromatic tectons as long as directional H-bonding and shape-persistent packing cooperate. "
                "That argues against collapsing the search into only fluorinated or nitrile-decorated variants of one scaffold family."
            ),
            (
                "Maintain scaffold diversity across triacid, diacid, aza-aromatic, and fused-aromatic HOF tectons.",
                "Use review precedent to justify aromatic-surface-rich frameworks even when fragment decoration is modest.",
                "Balance functionality with pore preservation instead of maximizing substituent count.",
            ),
            [
                r"multifunctional",
                r"porous",
                r"hydrogen-bonded organic framework",
                r"aromatic",
                r"function",
            ],
            0.77,
        ),
    ]
    cards: list[LocalEvidenceCard] = []
    for filename, title, summary, implications, patterns, confidence in review_specs:
        path = literature_root / filename
        if not path.exists():
            continue
        text = _pdf_to_text(path)
        extracted = _matching_snippets(text, patterns, limit=4)
        evidence = tuple(extracted) if extracted else (
            "Review article exists in the local literature folder and was selected as a scaffold-level HOF design reference.",
        )
        cards.append(
            LocalEvidenceCard(
                title=title,
                summary=summary,
                evidence=evidence,
                design_implications=implications,
                risk_flags=("collapse_on_activation",) if "Flexible HOF" in title else (),
                confidence=confidence,
                source_path=str(path),
            )
        )
    return cards


@lru_cache(maxsize=8)
def _load_fragment_rows(path: Path) -> tuple[LocalFragmentRecord, ...]:
    raw_rows = _read_xlsx_rows(path)
    fragments: list[LocalFragmentRecord] = []
    for row in raw_rows:
        smiles = str(row.get("SMILES", "")).strip()
        if not smiles:
            continue
        try:
            co2 = float(row.get("CO2_binding_energy", ""))
            h2o = float(row.get("H2O_binding_energy", ""))
        except ValueError:
            continue
        water_penalty = max(0.0, co2 - h2o)
        selection_score = round((-co2) - 1.5 * water_penalty, 6)
        families = tuple(_classify_fragment_smiles(smiles))
        if not families:
            continue
        fragments.append(
            LocalFragmentRecord(
                smiles=smiles,
                co2_binding_energy=co2,
                h2o_binding_energy=h2o,
                water_penalty=water_penalty,
                selection_score=selection_score,
                families=families,
            )
        )
    return tuple(fragments)


def _classify_fragment_smiles(smiles: str) -> list[str]:
    families: list[str] = []
    aromatic_n = smiles.count("n") + smiles.count("[nH]")
    ring_digits = len(set(re.findall(r"\d", smiles)))
    oxygen_count = smiles.count("O")
    if "C#N" in smiles or "N#C" in smiles:
        families.append("nitrile-bearing aromatics")
    if "F" in smiles:
        families.append("fluorinated aromatics")
    if aromatic_n:
        families.append("aza-aromatic clips")
    if ring_digits >= 2 and smiles.count("c") >= 6:
        families.append("rigid fused aromatics")
    if "C(=N)N" in smiles or "NC(=N)" in smiles or "N=C(N)" in smiles:
        families.append("amidine-guanidine controls")
    if oxygen_count >= 2:
        families.append("oxygen-rich pore-lining risks")
    return sorted(set(families))


@lru_cache(maxsize=8)
def _pdf_to_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout


def _matching_snippets(text: str, patterns: list[str], *, limit: int = 4) -> list[str]:
    snippets: list[str] = []
    compact = " ".join(text.split())
    for pattern in patterns:
        match = re.search(rf"(.{{0,120}}{pattern}.{{0,140}})", compact, flags=re.IGNORECASE)
        if not match:
            continue
        snippet = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _load_shared_strings(archive)
        rows_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = rows_root.findall(".//x:sheetData/x:row", namespace)
    parsed_rows: list[dict[str, str]] = []
    headers: list[str] = []
    for row_index, row in enumerate(rows):
        values: dict[int, str] = {}
        for cell in row.findall("x:c", namespace):
            ref = cell.attrib.get("r", "")
            idx = _column_index_from_ref(ref)
            value = _xlsx_cell_value(cell, namespace, shared_strings)
            values[idx] = value
        if not values:
            continue
        max_index = max(values)
        ordered = [values.get(idx, "") for idx in range(max_index + 1)]
        if row_index == 0:
            headers = ordered
            continue
        parsed_rows.append({headers[idx]: ordered[idx] if idx < len(ordered) else "" for idx in range(len(headers))})
    return parsed_rows


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall("x:si", namespace):
        text_parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        strings.append("".join(text_parts))
    return strings


def _xlsx_cell_value(cell, namespace: dict[str, str], shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value = cell.findtext("x:v", default="", namespaces=namespace)
    if cell_type == "s" and value:
        index = int(value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return value or ""


def _column_index_from_ref(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter.upper()) - ord("A") + 1)
    return max(0, index - 1)
