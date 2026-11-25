from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def rdkit_status() -> tuple[bool, str]:
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, "available"


@contextmanager
def suppress_rdkit_errors():
    ok, _ = rdkit_status()
    if not ok:
        yield
        return
    blocker = None
    try:
        from rdkit import rdBase

        blocker = rdBase.BlockLogs()
    except Exception:
        blocker = None
    try:
        yield
    finally:
        if blocker is not None:
            del blocker


def canonicalize_smiles(smiles: str) -> str:
    stripped = smiles.strip()
    ok, _ = rdkit_status()
    if not stripped or not ok:
        return stripped
    from rdkit import Chem

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_and_validate_registration_smiles(smiles: str) -> str:
    stripped = smiles.strip()
    ok, detail = rdkit_status()
    if not ok:
        raise RuntimeError(f"RDKit unavailable: {detail}")
    if not stripped:
        raise ValueError("SMILES is empty.")

    from rdkit import Chem

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(stripped, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    try:
        mol.UpdatePropertyCache(strict=True)
        with suppress_rdkit_errors():
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception as exc:
        raise ValueError(f"RDKit sanitize/kekulize failed: {exc}") from exc

    total_formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if total_formal_charge != 0:
        raise ValueError(
            f"Candidate must be neutral overall for campaign registration (formal charge {total_formal_charge:+d})."
        )

    radical_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetNumRadicalElectrons() != 0
    ]
    if radical_atoms:
        raise ValueError(
            "Candidate must be closed-shell for campaign registration; radical electrons were found on atoms "
            f"{radical_atoms}."
        )

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {9, 17, 35, 53}:
            continue
        bonds = list(atom.GetBonds())
        if len(bonds) != 1:
            raise ValueError(
                f"Halogen valence sanity failed for atom {atom.GetIdx()} ({atom.GetSymbol()}): expected one terminal bond."
            )
        bond = bonds[0]
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError(
                f"Halogen valence sanity failed for atom {atom.GetIdx()} ({atom.GetSymbol()}): expected a terminal single bond."
            )
        if atom.GetDegree() != 1 or atom.GetTotalValence() != 1 or atom.GetFormalCharge() != 0:
            raise ValueError(
                f"Halogen valence sanity failed for atom {atom.GetIdx()} ({atom.GetSymbol()}): "
                "halogens must be neutral terminal single-bond substituents."
            )

    return Chem.MolToSmiles(mol, canonical=True)


def smiles_element_symbols(smiles: str) -> list[str]:
    canonical = canonicalize_smiles(smiles)
    ok, detail = rdkit_status()
    if not ok:
        raise RuntimeError(f"RDKit unavailable: {detail}")
    if not canonical:
        return []

    from rdkit import Chem

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return []
    return sorted({atom.GetSymbol() for atom in mol.GetAtoms()})


def compute_smiles_descriptors(smiles: str) -> dict[str, Any] | None:
    canonical = canonicalize_smiles(smiles)
    ok, _ = rdkit_status()
    if not canonical or not ok:
        return None

    from rdkit import Chem
    from rdkit.Chem import Crippen
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Lipinski
    from rdkit.Chem import rdMolDescriptors

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    return {
        "canonical_smiles": canonical,
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(ring_info.NumRings()) if ring_info is not None else 0,
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def allowed_hof_core_symmetry_report(smiles: str) -> dict[str, Any] | None:
    canonical = canonicalize_smiles(smiles)
    ok, _ = rdkit_status()
    if not canonical or not ok:
        return None

    from rdkit import Chem

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None

    tpm_signatures = _tetraphenylmethane_arm_signatures(mol)
    if tpm_signatures:
        unique = sorted(set(tpm_signatures))
        return {
            "core_family": "tetraphenylmethane",
            "arm_count": len(tpm_signatures),
            "arm_signatures": tpm_signatures,
            "unique_arm_signatures": unique,
            "all_arms_identical": len(unique) == 1,
        }

    triazine_signatures = _triazine_arm_signatures(mol)
    if triazine_signatures:
        unique = sorted(set(triazine_signatures))
        return {
            "core_family": "triazine",
            "arm_count": len(triazine_signatures),
            "arm_signatures": triazine_signatures,
            "unique_arm_signatures": unique,
            "all_arms_identical": len(unique) == 1,
        }

    generic = _generic_bridge_arm_symmetry_report(mol)
    if generic is not None:
        return generic

    return {
        "core_family": None,
        "arm_count": 0,
        "arm_signatures": [],
        "unique_arm_signatures": [],
        "all_arms_identical": False,
    }


def smiles_to_xyz(
    smiles: str,
    output_path: str | Path,
    *,
    comment: str = "",
) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
    ok, detail = rdkit_status()
    if not ok:
        raise RuntimeError(f"RDKit unavailable: {detail}")
    if not canonical:
        raise ValueError("SMILES is empty.")

    from rdkit import Chem
    from rdkit.Chem import AllChem

    with suppress_rdkit_errors():
        mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.useRandomCoords = False
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        retry = AllChem.ETKDGv3()
        retry.randomSeed = 0xF00D
        retry.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol, retry)
    if embed_status != 0:
        raise RuntimeError(f"RDKit embedding failed for {canonical}")

    force_field = "MMFF"
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if mmff_props is not None:
        optimize_status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
    else:
        force_field = "UFF"
        optimize_status = AllChem.UFFOptimizeMolecule(mol)
    if optimize_status not in (0, 1):
        raise RuntimeError(f"RDKit {force_field} optimization failed for {canonical}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment or canonical]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2} {pos.x: .10f} {pos.y: .10f} {pos.z: .10f}"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "canonical_smiles": canonical,
        "output_path": str(target),
        "force_field": force_field,
        "atom_count": int(mol.GetNumAtoms()),
    }


def _tetraphenylmethane_arm_signatures(mol) -> list[str]:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 4:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 4:
            continue
        if not all(neighbor.GetAtomicNum() == 6 and neighbor.GetIsAromatic() for neighbor in neighbors):
            continue
        core_set = {atom.GetIdx()}
        return sorted(_arm_signature(mol, neighbor.GetIdx(), core_set) for neighbor in neighbors)
    return []


def _triazine_arm_signatures(mol) -> list[str]:
    from rdkit import Chem

    pattern = Chem.MolFromSmarts("n1c([*])nc([*])nc1[*]")
    if pattern is None:
        return []
    for match in mol.GetSubstructMatches(pattern):
        if len(match) != 9:
            continue
        core_indices = {match[idx] for idx in (0, 1, 3, 4, 6, 7)}
        arm_starts = [match[idx] for idx in (2, 5, 8)]
        if any(start in core_indices for start in arm_starts):
            continue
        return sorted(_arm_signature(mol, start, core_indices) for start in arm_starts)
    return []


def _arm_signature(mol, start_idx: int, core_indices: set[int]) -> str:
    from rdkit import Chem

    atoms_to_use = _collect_component_atoms(mol, start_idx, core_indices)

    # RDKit forbids rootedAtAtom when the parent molecule has multiple fragments.
    # Candidates with multi-fragment SMILES should be rejected upstream, but this
    # guard keeps the symmetry helper from crashing hard.
    rooted = start_idx
    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            rooted = -1
    except Exception:
        rooted = -1

    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(atoms_to_use),
        rootedAtAtom=rooted,
        canonical=True,
    )


def _collect_component_atoms(mol, start_idx: int, core_indices: set[int]) -> set[int]:
    stack = [start_idx]
    visited: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited or idx in core_indices:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited and neighbor_idx not in core_indices:
                stack.append(neighbor_idx)
    return visited


def _generic_bridge_arm_symmetry_report(mol) -> dict[str, Any] | None:
    from collections import defaultdict

    from rdkit import Chem

    total_atoms = mol.GetNumAtoms()
    if total_atoms < 8:
        return None

    symmetry_classes = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    bridge_records: list[dict[str, Any]] = []

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        side_begin = _collect_component_atoms_without_bond(mol, begin, frozenset({begin, end}))
        side_end = _collect_component_atoms_without_bond(mol, end, frozenset({begin, end}))
        if len(side_begin) == total_atoms or len(side_end) == total_atoms:
            continue
        if len(side_begin) == len(side_end):
            continue

        if len(side_begin) < len(side_end):
            arm_atoms = side_begin
            arm_root = begin
            anchor_idx = end
        else:
            arm_atoms = side_end
            arm_root = end
            anchor_idx = begin

        if len(arm_atoms) < 3:
            continue

        arm_signature = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=sorted(arm_atoms),
            rootedAtAtom=arm_root,
            canonical=True,
        )
        bridge_records.append(
            {
                "anchor_idx": anchor_idx,
                "anchor_symmetry_class": symmetry_classes[anchor_idx],
                "anchor_symbol": mol.GetAtomWithIdx(anchor_idx).GetSymbol(),
                "arm_signature": arm_signature,
            }
        )

    if not bridge_records:
        return None

    by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_symmetry_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in bridge_records:
        by_anchor[record["anchor_idx"]].append(record)
        by_symmetry_class[record["anchor_symmetry_class"]].append(record)

    for anchor_idx, records in by_anchor.items():
        if len(records) < 3:
            continue
        signatures = sorted(record["arm_signature"] for record in records)
        unique = sorted(set(signatures))
        if len(unique) == 1:
            atom = mol.GetAtomWithIdx(anchor_idx)
            return {
                "core_family": f"generic-shared-anchor-{atom.GetSymbol().lower()}",
                "arm_count": len(signatures),
                "arm_signatures": signatures,
                "unique_arm_signatures": unique,
                "all_arms_identical": True,
            }

    for symmetry_class, records in by_symmetry_class.items():
        anchor_indices = {record["anchor_idx"] for record in records}
        if len(anchor_indices) < 3:
            continue
        signatures_by_anchor: dict[int, list[str]] = defaultdict(list)
        for record in records:
            signatures_by_anchor[record["anchor_idx"]].append(record["arm_signature"])
        if any(len(signatures) != 1 for signatures in signatures_by_anchor.values()):
            continue
        signatures = sorted(signatures[0] for signatures in signatures_by_anchor.values())
        unique = sorted(set(signatures))
        if len(unique) == 1:
            return {
                "core_family": f"generic-equivalent-sites-class-{symmetry_class}",
                "arm_count": len(signatures),
                "arm_signatures": signatures,
                "unique_arm_signatures": unique,
                "all_arms_identical": True,
            }

    return None


def _collect_component_atoms_without_bond(mol, start_idx: int, blocked_edge: frozenset[int]) -> set[int]:
    stack = [start_idx]
    visited: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            edge = frozenset({idx, neighbor.GetIdx()})
            if edge == blocked_edge:
                continue
            if neighbor.GetIdx() not in visited:
                stack.append(neighbor.GetIdx())
    return visited
