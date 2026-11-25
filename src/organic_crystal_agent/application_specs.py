from __future__ import annotations

from pydantic import BaseModel, Field

from organic_crystal_agent.schemas import DiscoveryCampaign


class ApplicationSpec(BaseModel):
    application_mode: str
    evidence_source: str = "heuristic"
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    synthesis_heuristics: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


KNOWN_APPLICATION_SPECS: dict[str, ApplicationSpec] = {
    "humid_co2_capture": ApplicationSpec(
        application_mode="humid_co2_capture",
        evidence_source="workflow_default_constraints",
        hard_constraints=[
            "Ground fragment and tecton choices in the active campaign literature, supplementary data, and local structured evidence rather than a preset scaffold library.",
            "Require porous HOF candidates that remain structurally plausible after activation, CSP, and relaxation before any humid-CO2 ranking is trusted.",
            "Evaluate candidates against explicit CO2/H2O adsorption evidence instead of relying on dry-gas intuition or a single dense minimum.",
        ],
        soft_preferences=[
            "Prefer modular designs whose fragment-level rationale can be traced back to campaign literature cards or supplemental datasets.",
            "Prefer candidates with multiple interpretable design levers so the next round can compare fragment substitutions instead of restarting from scratch.",
            "Prefer porous spike families that stay physically plausible after relaxation and can be checked with mixed-gas adsorption rather than only one-off low-energy structures.",
        ],
        red_flags=[
            "Design proposals that are not supported by the active campaign literature, local spreadsheets, or in-campaign evidence.",
            "Soft scaffolds or activation-sensitive motifs that collapse before meaningful mixed-gas evaluation.",
            "Designs that only make sense if one very specific dense polymorph appears despite no landscape support.",
        ],
        evaluation_criteria=[
            "CO2 uptake under humid conditions or mixed-gas CO2/H2O conditions",
            "Suppressed H2O uptake or improved CO2/H2O balance under the chosen simulation protocol",
            "Porous-spike plausibility after CSP and relaxation",
            "Synthetic tractability",
        ],
        synthesis_heuristics=[
            "Prefer modular derivatives so the workflow can tune polarity, contortion, and pore lining incrementally across rounds.",
            "Document the literature or supplemental-data rationale for each proposed fragment family before submitting CSP.",
        ],
        search_queries=[
            "humid CO2 capture hydrogen-bonded organic framework design",
            "CO2 H2O competitive adsorption porous organic crystal",
            "humid CO2 capture fragment design organic framework",
        ],
    ),
    "hof": ApplicationSpec(
        application_mode="hof",
        evidence_source="workflow_default_constraints",
        hard_constraints=[
            "Require multiple directional hydrogen-bond donor/acceptor sites.",
            "Prefer rigid or shape-persistent cores over highly flexible scaffolds.",
            "Require sufficient molecular size and connectivity to support open frameworks.",
        ],
        soft_preferences=[
            "High symmetry and multivalent binding motifs.",
            "Large aromatic or fused cores that discourage dense collapse.",
            "Distributed donor/acceptor sites rather than one localized sticky patch.",
        ],
        red_flags=[
            "Too many rotatable bonds or floppy aliphatic segments.",
            "Single-point hydrogen-bonding motifs that favor chain growth without porosity.",
            "Likely dense interdigitation or strong self-complementary collapse.",
        ],
        evaluation_criteria=[
            "Hydrogen-bond richness",
            "Rigidity / shape persistence",
            "Synthetic accessibility",
            "Open-packing plausibility",
        ],
        synthesis_heuristics=[
            "Favor scaffold families with literature precedents from polycarboxylates, pyridyl-rich tectons, amides, ureas, and triazines.",
            "Prefer modular substitutions and avoid routes that require unusually unstable intermediates.",
            "Prefer symmetric derivatives obtainable by repeating the same head substitution pattern across all arms instead of mixed-arm bespoke synthesis.",
            "Prefer candidates that look accessible from commercially common or precedent-adjacent aryl halide, nitrile, fluoroaryl, aza-aryl, or carboxylate building blocks.",
        ],
        search_queries=[
            "hydrogen-bonded organic framework design principles",
            "porous HOF rigid multidentate tectons",
            "HOF collapse and solvent-trapping failure modes",
        ],
    ),
    "charge_transport": ApplicationSpec(
        application_mode="charge_transport",
        evidence_source="workflow_default_constraints",
        hard_constraints=[
            "Require conjugated rigid backbones with limited torsional freedom.",
            "Favor low reorganization-energy motifs and stable frontier energy alignment.",
        ],
        soft_preferences=[
            "Planar or controllably slipped pi-surfaces.",
            "Packing motifs compatible with strong dimer coupling.",
            "Substitution patterns that avoid excessive dipolar disorder.",
        ],
        red_flags=[
            "Highly twisted cores with many rotatable bonds.",
            "Bulky substitutions that destroy productive intermolecular contact.",
            "Unstable radicals or obviously impractical synthetic targets.",
        ],
        evaluation_criteria=[
            "Rigidity",
            "Conjugation length",
            "Packing-aware transport proxy",
            "Synthetic tractability",
        ],
        synthesis_heuristics=[
            "Favor fused aromatic, imide, quinonoid, and heteroaromatic semiconductor motifs with precedent.",
        ],
        search_queries=[
            "organic semiconductor molecular design low reorganization energy",
            "packing motifs charge transport organic crystals",
            "dimer coupling crystal engineering organic semiconductors",
        ],
    ),
}


def infer_application_mode(campaign: DiscoveryCampaign) -> str:
    text = f"{campaign.target_description} {' '.join(campaign.desired_properties)}".lower()
    if "co2" in text and ("humid" in text or "h2o" in text or "water" in text):
        return "humid_co2_capture"
    if "hof" in text or "hydrogen-bonded organic framework" in text:
        return "hof"
    if "transport" in text or "mobility" in text or "semiconductor" in text:
        return "charge_transport"
    if "porous" in text:
        return "porous_crystal"
    return "general"


def build_application_spec(campaign: DiscoveryCampaign) -> ApplicationSpec:
    mode = infer_application_mode(campaign)
    if mode in KNOWN_APPLICATION_SPECS:
        return KNOWN_APPLICATION_SPECS[mode].model_copy(deep=True)
    objective = campaign.target_description
    desired = ", ".join(campaign.desired_properties) or "target property"
    return ApplicationSpec(
        application_mode=mode,
        evidence_source="inferred_from_goal",
        hard_constraints=[
            f"Preserve structural motifs plausibly linked to {desired}.",
            "Avoid excessively flexible or synthetically implausible candidates.",
        ],
        soft_preferences=[
            f"Favor scaffolds with literature precedents adjacent to {objective}.",
            "Prefer molecules that expose multiple interpretable design levers for follow-up.",
        ],
        red_flags=[
            "Unconstrained molecular growth without a mechanistic rationale.",
            "Designs that cannot be evaluated with current CSP and property workflow.",
        ],
        evaluation_criteria=[
            "Mechanistic relevance to target application",
            "Synthetic tractability",
            "Packing plausibility",
        ],
        synthesis_heuristics=[
            "Prefer modular derivatives or motifs with nearby precedent in crystal engineering or molecular materials.",
        ],
        search_queries=[
            objective,
            f"{objective} design rules organic crystal",
            f"{objective} failure modes crystal engineering",
        ],
    )
