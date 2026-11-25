"""Seed knowledge used by the initial deterministic platform.

This is not intended to be the long-term knowledge layer. It exists to keep the
first platform version testable before local literature ingestion is wired in.
"""

SEED_KNOWLEDGE = [
    {
        "topic": "crystal-form-aware screening",
        "insight": (
            "Organic crystal discovery should optimize over molecular identity, crystal "
            "form, and crystallization route rather than molecule-only properties."
        ),
    },
    {
        "topic": "polymorphism",
        "insight": (
            "Low-energy polymorphs are not automatically experimentally accessible; "
            "kinetics, solvents, and seeding strongly affect which form appears."
        ),
    },
    {
        "topic": "realizability",
        "insight": (
            "Promising candidates should be ranked by formability, polymorph risk, "
            "stability window, and characterization distinguishability."
        ),
    },
    {
        "topic": "organic semiconductors",
        "insight": (
            "Packing-aware discovery is critical because transport-relevant behavior is "
            "strongly controlled by crystal structure rather than isolated molecules."
        ),
    },
    {
        "topic": "porous molecular crystals",
        "insight": (
            "Low-density porous forms are especially vulnerable to solvent trapping and "
            "post-activation collapse, so verification must be part of ranking."
        ),
    },
]
