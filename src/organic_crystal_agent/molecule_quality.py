from __future__ import annotations

import re
from pydantic import BaseModel, Field

from organic_crystal_agent.application_specs import ApplicationSpec
from organic_crystal_agent.rdkit_utils import compute_smiles_descriptors, smiles_element_symbols
from organic_crystal_agent.schemas import MoleculeRecord


class MoleculeQualityReport(BaseModel):
    molecule_label: str
    quality_score: float = 0.0
    passes_hard_filters: bool = True
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    derived_tags: list[str] = Field(default_factory=list)


def assess_molecule_quality(
    molecule: MoleculeRecord,
    spec: ApplicationSpec,
) -> MoleculeQualityReport:
    proposal_notes = _proposal_note_text(molecule.notes)
    proposal_text = " ".join(
        [
            molecule.molecule_label,
            molecule.rationale,
            molecule.design_family,
            proposal_notes,
        ]
    ).lower()
    text = " ".join(
        [
            molecule.molecule_label,
            molecule.rationale,
            molecule.design_family,
            " ".join(molecule.evidence),
            proposal_notes,
        ]
    ).lower()
    strengths: list[str] = []
    concerns: list[str] = []
    tags: list[str] = []
    score = 0.0

    rigidity_hits = _count_matches(text, ["rigid", "fused", "shape-persistent", "contorted", "buttressed", "planar"])
    hb_hits = _count_matches(text, ["hydrogen-bond", "hydrogen bond", "amide", "urea", "triazine", "carbox", "pyrid"])
    synth_hits = _count_matches(text, ["benchmark", "precedent", "modular", "tractable", "control", "commercial", "single-step", "iterative"])
    symmetry_hits = _count_word_matches(text, ["symmetric", "symmetrical", "high symmetry", "c3 symmetry", "c4 symmetry"])
    asymmetry_hits = _count_word_matches(text, ["asymmetric", "unsymmetrical", "nonuniform", "mixed-arm", "mixed arm", "one arm", "single arm"])
    flexibility_hits = _count_matches(text, ["flexible", "rotatable", "aliphatic", "floppy"])
    dense_hits = _count_matches(text, ["dense", "interdigitation", "collapse", "solvate"])
    large_hits = _count_matches(text, ["large", "cage", "tetratopic", "multidentate", "extended", "star"])
    five_member_hits = _count_positive_five_member_hits(proposal_text)

    if rigidity_hits:
        strengths.append("Rationale emphasizes rigidity or shape persistence.")
        score += 2.0
        tags.append("rigid")
    if hb_hits:
        strengths.append("Design includes directional donor/acceptor or hydrogen-bond motifs.")
        score += 2.0
        tags.append("hbond_rich")
    if large_hits:
        strengths.append("Design suggests sufficient size or multiconnectivity for network formation.")
        score += 1.5
        tags.append("large_multivalent")
    if synth_hits:
        strengths.append("Proposal references tractable or precedent-backed design choices.")
        score += 1.0
        tags.append("tractable")
    if symmetry_hits:
        strengths.append("Proposal explicitly favors high symmetry, which is helpful for HOF synthesis and packing screening.")
        score += 1.0
        tags.append("symmetric")
    if five_member_hits:
        strengths.append("Proposal explicitly explores five-membered or pentagonal motifs, which are one literature-supported humid-CO2 branch.")
        score += 0.5
        tags.append("five_member_priority")

    if flexibility_hits:
        concerns.append("Text suggests flexibility or rotatable segments that may hurt structural persistence.")
        score -= 2.0
        tags.append("flexible_risk")
    if asymmetry_hits:
        concerns.append("Text suggests asymmetric or mixed-arm substitution; symmetry would usually be easier, but exploratory asymmetry can still be useful.")
        score -= 0.5
        tags.append("asymmetry_risk")
    if dense_hits:
        concerns.append("Text indicates collapse, solvate trapping, or dense-packing risk.")
        score -= 1.5
        tags.append("packing_risk")
    elif synth_hits == 0:
        concerns.append("Synthetic accessibility is not explicitly justified in the text, so this candidate should rank lower until the rationale is documented.")
        score -= 0.5
        tags.append("synthesis_rationale_missing")

    passes = True
    descriptors = compute_smiles_descriptors(molecule.smiles or "")
    if molecule.smiles and spec.application_mode in {"hof", "humid_co2_capture"}:
        try:
            element_symbols = set(smiles_element_symbols(molecule.smiles))
        except Exception:
            element_symbols = set()
        if "Si" in element_symbols:
            concerns.append("HOF mode excludes silicon-containing scaffolds from formal campaign registration.")
            passes = False
            score -= 3.0
            tags.append("forbidden_element_si")
    if descriptors:
        if descriptors["rotatable_bonds"] <= 1:
            strengths.append("RDKit descriptors suggest low conformational flexibility.")
            score += 1.5
            tags.append("low_rotor_count")
        elif descriptors["rotatable_bonds"] >= 4:
            concerns.append("RDKit descriptors show several rotatable bonds.")
            score -= 1.5
            tags.append("rotor_heavy")

        if descriptors["ring_count"] >= 2:
            strengths.append("RDKit descriptors show a multi-ring scaffold.")
            score += 1.0
            tags.append("polycyclic")
        elif spec.application_mode == "hof":
            concerns.append("RDKit descriptors suggest too few rings for a rigid HOF tecton.")
            score -= 1.0
            tags.append("few_rings")

        if descriptors["hbd"] + descriptors["hba"] >= 4:
            strengths.append("RDKit descriptors show rich donor/acceptor functionality.")
            score += 1.0
            tags.append("rich_hbond_inventory")

        if spec.application_mode == "hof":
            if descriptors["rotatable_bonds"] > 2:
                concerns.append("HOF mode prefers fewer rotatable bonds than this candidate currently has.")
                passes = False
            if descriptors["hbd"] + descriptors["hba"] < 3:
                concerns.append("HOF mode expects stronger donor/acceptor inventory than this candidate shows.")
                passes = False
        elif spec.application_mode == "charge_transport":
            if descriptors["aromatic_rings"] < 2:
                concerns.append("Charge-transport mode prefers multi-ring aromatic cores.")
                passes = False
            if descriptors["fraction_csp3"] > 0.4:
                concerns.append("Charge-transport mode disfavors high sp3 character.")
                passes = False

    if spec.application_mode == "hof":
        if hb_hits == 0:
            concerns.append("HOF mode expects explicit hydrogen-bond donor/acceptor motifs.")
            passes = False
        if rigidity_hits == 0:
            concerns.append("HOF mode expects rigid or shape-persistent scaffolds.")
            passes = False
        if large_hits == 0:
            concerns.append("HOF mode expects larger or multivalent tectons for framework formation.")
            passes = False
        if asymmetry_hits and symmetry_hits == 0:
            concerns.append("HOF mode still prefers symmetric candidates, but asymmetry is now treated as an exploration cost rather than an automatic failure.")
    elif spec.application_mode == "humid_co2_capture":
        if asymmetry_hits and symmetry_hits == 0:
            concerns.append("Humid-CO2 HOF workflow prefers symmetric candidates, but asymmetry is now treated as a soft penalty rather than an automatic failure.")
    elif spec.application_mode == "charge_transport":
        if rigidity_hits == 0:
            concerns.append("Charge-transport mode expects rigid conjugated motifs.")
            passes = False
        if _count_matches(text, ["pi", "conjug", "stack", "semiconductor", "imide", "fused"]) == 0:
            concerns.append("Charge-transport mode expects explicit conjugation/packing rationale.")
            passes = False

    return MoleculeQualityReport(
        molecule_label=molecule.molecule_label,
        quality_score=round(score, 6),
        passes_hard_filters=passes,
        strengths=strengths,
        concerns=concerns,
        derived_tags=sorted(set(tags)),
    )


def _count_matches(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if re.search(re.escape(term), text))


def _count_word_matches(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text))


def _count_positive_five_member_hits(text: str) -> int:
    motif_terms = [
        "five-member",
        "five member",
        "five-membered",
        "pentagon",
        "pentagonal",
        "imidazole",
        "pyrazole",
        "triazole",
        "pyrrole",
        "thiophene",
        "oxazole",
        "isoxazole",
        "thiazole",
        "indole",
        "benzothiadiazole",
        "cyclopent",
        "furan",
    ]
    negative_terms = [
        "move away from",
        "moves away from",
        "designed as a scaffold-diversification move away from",
        "diversify away from",
        "away from",
        "avoid",
        "avoids",
        "avoiding",
        "not default",
        "do not",
        "don't",
        "rather than",
        "instead of",
        "switch to a different",
    ]
    hits = 0
    for sentence in re.split(r"[.;\n]+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not any(term in sentence for term in motif_terms):
            continue
        if any(term in sentence for term in negative_terms):
            continue
        hits += 1
    return hits


def _proposal_note_text(notes: list[str]) -> str:
    excluded_prefixes = (
        "quality ",
        "quality:",
        "quality score:",
        "quality pass:",
        "quality strength:",
        "quality concern:",
        "quality tag:",
        "critic ",
        "critic:",
        "core symmetry preference:",
        "3d coords generated",
        "dft-refined monomer geometry ready",
    )
    kept = []
    for note in notes:
        normalized = note.strip().lower()
        if normalized.startswith(excluded_prefixes):
            continue
        kept.append(note)
    return " ".join(kept)
