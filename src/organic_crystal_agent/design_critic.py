from __future__ import annotations

import re

from pydantic import BaseModel, Field

from organic_crystal_agent.application_specs import ApplicationSpec
from organic_crystal_agent.molecule_quality import MoleculeQualityReport
from organic_crystal_agent.rdkit_utils import compute_smiles_descriptors
from organic_crystal_agent.schemas import MoleculeRecord


class DesignCritique(BaseModel):
    molecule_label: str
    should_reject: bool = False
    positive_signals: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    red_flag_hits: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    synthesis_complexity_score: float = 0.0
    synthesis_complexity_label: str = "unknown"
    synthesis_complexity_reasons: list[str] = Field(default_factory=list)


def critique_molecule_design(
    molecule: MoleculeRecord,
    spec: ApplicationSpec,
    quality: MoleculeQualityReport | None = None,
) -> DesignCritique:
    text = " ".join(
        [
            molecule.molecule_label,
            molecule.rationale,
            molecule.design_family,
        ]
    ).lower()

    positive_signals: list[str] = []
    concerns: list[str] = []
    red_flag_hits: list[str] = []
    suggested_revisions: list[str] = []
    synthesis_report = _assess_synthetic_complexity(molecule, text)

    if quality and quality.strengths:
        positive_signals.extend(quality.strengths[:3])

    flexible_hits = _count_matches(text, ["flexible", "rotatable", "aliphatic", "floppy"])
    collapse_hits = _count_matches(text, ["collapse", "dense", "interdigitation", "self-complementary"])
    synthesis_hits = _count_matches(text, ["unstable", "impractical", "multi-step", "exotic", "unprecedented"])
    asymmetry_hits = _count_matches(text, ["asymmetric", "mixed-arm", "mixed arm", "one arm", "single arm", "nonuniform", "unsymmetrical"])
    localized_hits = _count_matches(text, ["single-point", "localized sticky", "one localized sticky patch"])

    if flexible_hits:
        red_flag_hits.append("flexibility_risk")
        concerns.append("The proposal still reads as too flexible for robust crystal-level persistence.")
        suggested_revisions.append("Increase scaffold rigidity or remove floppy linkers.")
    if collapse_hits:
        red_flag_hits.append("dense_or_collapse_risk")
        concerns.append("The design rationale suggests collapse, interdigitation, or dense-packing risk.")
        suggested_revisions.append("Introduce steric buttressing or distribute interaction sites to frustrate dense collapse.")
    if synthesis_hits:
        red_flag_hits.append("synthetic_implausibility")
        concerns.append("The proposed scaffold looks synthetically fragile or difficult to access at screening scale.")
        suggested_revisions.append("Prefer modular derivatives with stronger literature precedent.")
    if asymmetry_hits:
        concerns.append("The proposal reads as asymmetric or mixed-arm; symmetry would be easier to compare and synthesize, but exploratory asymmetry can still be worth testing.")
        suggested_revisions.append("Prefer symmetric all-arm substitution patterns when easy to realize, but keep asymmetric variants if they open new scaffold space.")
    if localized_hits:
        red_flag_hits.append("localized_binding_risk")
        concerns.append("Binding functionality appears too localized, which can bias chain growth or dense self-association.")
        suggested_revisions.append("Distribute donor/acceptor motifs across the scaffold instead of concentrating one sticky patch.")
    if synthesis_report["label"] == "high":
        red_flag_hits.append("synthetic_complexity_risk")
        concerns.append(
            "The candidate looks synthetically overcomplicated for a fast screening campaign."
        )
        suggested_revisions.append(
            "Simplify the scaffold or reduce repeated heavily decorated arms while preserving the core humid-CO2 idea."
        )
    elif synthesis_report["label"] == "moderate":
        concerns.append(
            "The candidate looks moderately complex to synthesize; keep it only if the scaffold or cavity idea is clearly distinctive."
        )

    if spec.application_mode == "hof":
        if _count_matches(text, ["hydrogen-bond", "hydrogen bond", "amide", "urea", "triazine", "carbox", "pyrid"]) >= 2:
            positive_signals.append("The proposal keeps explicit multivalent H-bonding motifs in view.")
        if _count_matches(text, ["rigid", "shape-persistent", "contorted", "cage", "tetratopic", "large"]) >= 2:
            positive_signals.append("The scaffold description is consistent with size and shape persistence expected for HOF tectons.")
        if flexible_hits or localized_hits:
            concerns.append("For HOF discovery this candidate is likely to underperform because it lacks robust distributed framework-forming cues.")
    elif spec.application_mode == "charge_transport":
        if _count_matches(text, ["fused", "planar", "conjug", "pi", "imide", "quinonoid"]) >= 2:
            positive_signals.append("The proposal preserves rigid conjugation that is relevant to charge transport.")
        if flexible_hits or _count_matches(text, ["bulky", "twisted"]) > 0:
            red_flag_hits.append("transport_contact_risk")
            concerns.append("The current description risks weak intermolecular electronic contact.")
            suggested_revisions.append("Bias toward more planar, less sterically blocking substitution patterns.")

    if quality and not quality.passes_hard_filters:
        concerns.append("The molecule already failed the application-specific hard quality gate.")
        red_flag_hits.append("failed_quality_gate")
    if quality:
        for concern in quality.concerns[:3]:
            if concern not in concerns:
                concerns.append(concern)

    should_reject = False
    if quality and not quality.passes_hard_filters:
        should_reject = True
    elif len(red_flag_hits) >= 2:
        should_reject = True
    elif spec.application_mode == "hof" and (
        "flexibility_risk" in red_flag_hits
        or "dense_or_collapse_risk" in red_flag_hits
    ):
        should_reject = True
    elif "synthetic_complexity_risk" in red_flag_hits and "tractable" not in (quality.derived_tags if quality else []):
        should_reject = True

    return DesignCritique(
        molecule_label=molecule.molecule_label,
        should_reject=should_reject,
        positive_signals=positive_signals,
        concerns=concerns,
        red_flag_hits=sorted(set(red_flag_hits)),
        suggested_revisions=sorted(set(suggested_revisions)),
        synthesis_complexity_score=float(synthesis_report["score"]),
        synthesis_complexity_label=str(synthesis_report["label"]),
        synthesis_complexity_reasons=list(synthesis_report["reasons"]),
    )


def _count_matches(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if re.search(re.escape(term), text))


def _assess_synthetic_complexity(molecule: MoleculeRecord, text: str) -> dict[str, object]:
    descriptors = compute_smiles_descriptors(molecule.smiles or "")
    score = 0.0
    reasons: list[str] = []

    if descriptors:
        heavy_atoms = int(descriptors.get("heavy_atom_count", 0))
        aromatic_rings = int(descriptors.get("aromatic_rings", 0))
        ring_count = int(descriptors.get("ring_count", 0))
        rotors = int(descriptors.get("rotatable_bonds", 0))
        hba = int(descriptors.get("hba", 0))
        hbd = int(descriptors.get("hbd", 0))

        if heavy_atoms >= 36:
            score += 1.0
            reasons.append(f"heavy_atom_count={heavy_atoms}")
        if heavy_atoms >= 46:
            score += 1.0
        if aromatic_rings >= 4:
            score += 1.0
            reasons.append(f"aromatic_rings={aromatic_rings}")
        if ring_count >= 5:
            score += 1.0
            reasons.append(f"ring_count={ring_count}")
        if rotors >= 4:
            score += 1.0
            reasons.append(f"rotatable_bonds={rotors}")
        if hba + hbd >= 8:
            score += 1.0
            reasons.append(f"hetero_inventory={hba + hbd}")

    smiles = molecule.smiles or ""
    carboxyl_count = smiles.count("C(=O)O") + smiles.count("C(O)=O")
    nitrile_count = smiles.count("C#N")
    fluoro_count = smiles.count("F")
    chloro_count = smiles.count("Cl")
    bromo_count = smiles.count("Br")
    iodo_count = smiles.count("I")
    halogen_count = fluoro_count + chloro_count + bromo_count + iodo_count

    if carboxyl_count >= 3:
        score += 1.0
        reasons.append(f"carboxylates={carboxyl_count}")
    if carboxyl_count >= 4:
        score += 1.0
    if nitrile_count >= 3:
        score += 1.0
        reasons.append(f"nitriles={nitrile_count}")
    if nitrile_count >= 4:
        score += 1.0
    if halogen_count >= 4:
        score += 1.0
        reasons.append(f"halogens={halogen_count}")
    if carboxyl_count + nitrile_count + halogen_count >= 8:
        score += 1.0
        reasons.append("many repeated arm decorations")

    if _count_matches(text, ["modular", "tractable", "commercial", "precedent", "benchmark", "control", "simple", "synthesis-first"]) >= 2:
        score -= 1.0
        reasons.append("explicit modular/precedent rationale")
    if _count_matches(text, ["overfunctionalized", "bespoke", "macrocycle", "pentakis", "hexakis", "spiro", "triptycene", "exotic", "unprecedented"]) >= 1:
        score += 1.5
        reasons.append("text flags bespoke or highly specialized chemistry")

    if score <= 1.0:
        label = "low"
    elif score <= 3.0:
        label = "moderate"
    else:
        label = "high"

    return {
        "score": round(score, 3),
        "label": label,
        "reasons": reasons,
    }
