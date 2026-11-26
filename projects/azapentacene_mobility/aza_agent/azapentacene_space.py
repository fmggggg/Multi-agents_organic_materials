"""RDKit-only functions extracted verbatim from the bundled AlphaPack sampler."""
import os

import pandas as pd
from rdkit import Chem

def force_aromatic_rings_simple(mol):
    """
    粗暴版芳香性重构：
    - 找所有环；
    - 若环长为 5 或 6 且只含 C/N/O/S，则把环上 bond 全部设为 AROMATIC，同时标记 atom 为 aromatic。
    不依赖 RDKit 的 hybridization 判定。
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return mol

    for ring in atom_rings:
        ring = list(ring)
        if len(ring) not in (5, 6):
            continue

        ok = True
        for idx in ring:
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            if sym not in ["C", "N", "O", "S"]:
                ok = False
                break
        if not ok:
            continue

        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            b = mol.GetBondBetweenAtoms(a1, a2)
            if b is None:
                continue
            b.SetBondType(Chem.BondType.AROMATIC)
            b.SetIsAromatic(True)

        for idx in ring:
            a = mol.GetAtomWithIdx(idx)
            a.SetIsAromatic(True)

    return mol


def to_mol_safe(smiles: str):
    """SMILES -> mol (sanitized). Return None if fails."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def aromatize_for_key(mol):
    """
    Aromatization heuristic aligned with build_rotated_molecule_from_selfies:
    - RemoveHs
    - Sanitize
    - force_aromatic_rings_simple
    - Sanitize: set aromaticity + adjustHs + properties + symmrings
    Return mol (H removed) or None.
    """
    if mol is None:
        return None
    m = Chem.Mol(Chem.RemoveHs(mol))
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None

    m = force_aromatic_rings_simple(m)
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS
            | Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
    except Exception:
        return None
    return m


def make_graph_key(mol):
    """
    Graph-key: ignore bond orders/aromaticity; keep connectivity + elements.
    - RemoveHs
    - Set all bonds SINGLE and non-aromatic; atoms non-aromatic
    - Canonical SMILES (isomericSmiles=False)
    """
    if mol is None:
        return None
    m = Chem.Mol(Chem.RemoveHs(mol))
    try:
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        for b in m.GetBonds():
            b.SetIsAromatic(False)
            b.SetBondType(Chem.BondType.SINGLE)
        # minimal sanitize to keep rings consistent
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_CLEANUP,
        )
    except Exception:
        # fallback: still try to serialize
        pass

    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def make_aroma_key(mol):
    """
    Aroma-key: canonical smiles after aromatize_for_key().
    """
    m = aromatize_for_key(mol)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def load_known_keys(known_csv, smiles_column="smiles"):
    """
    从 CSV 读取已知分子的 SMILES，构建两套 key：
      - graph_key_set
      - aroma_key_set
    如果没有提供 known_csv，返回 (None, None)。
    """
    if known_csv is None:
        return None, None

    if not os.path.isfile(known_csv):
        raise FileNotFoundError(f"known_csv not found: {known_csv}")

    df = pd.read_csv(known_csv)
    cols = list(df.columns)
    if smiles_column not in df.columns:
        print(f"[WARN] Column '{smiles_column}' not found in {known_csv}, use first column '{cols[0]}' instead.")
        smiles_column = cols[0]

    graph_set = set()
    aroma_set = set()
    n_total = 0
    n_bad = 0
    n_no_gk = 0
    n_no_ak = 0

    for s in df[smiles_column].dropna().astype(str):
        n_total += 1
        mol = to_mol_safe(s)
        if mol is None:
            n_bad += 1
            continue

        gk = make_graph_key(mol)
        ak = make_aroma_key(mol)
        if gk is None:
            n_no_gk += 1
        else:
            graph_set.add(gk)
        if ak is None:
            n_no_ak += 1
        else:
            aroma_set.add(ak)

    print(
        f"[INFO] Loaded known keys from {known_csv} (rows={n_total}, parse_fail={n_bad})\n"
        f"       graph_keys={len(graph_set)} (missing={n_no_gk}), aroma_keys={len(aroma_set)} (missing={n_no_ak})"
    )
    return graph_set, aroma_set


def _ring_adjacency_is_linear_fused(atom_rings):
    """
    ring adjacency graph must be a simple path of 5 rings:
      - edges == 4
      - degree multiset == [1,1,2,2,2]
    Two rings are adjacent if they share exactly 2 atoms (share one bond).
    Reject spiro (share 1 atom) or bridged (>2 atoms) patterns.
    """
    n = len(atom_rings)
    if n != 5:
        return False, {"why": "n_rings_not_5", "n": n}

    ring_sets = [set(r) for r in atom_rings]
    adj = [[] for _ in range(n)]
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            inter = ring_sets[i].intersection(ring_sets[j])
            if len(inter) == 2:
                adj[i].append(j)
                adj[j].append(i)
                edge_count += 1
            elif len(inter) != 0:
                return False, {"why": "spiro_or_bridged", "i": i, "j": j, "shared_atoms": len(inter)}

    degs = sorted(len(x) for x in adj)

    if edge_count != 4:
        return False, {"why": "edge_count_not_4", "edge_count": edge_count, "degrees": degs}

    if degs != [1, 1, 2, 2, 2]:
        return False, {"why": "degree_multiset_not_path", "degrees": degs}

    # connectivity
    seen = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                stack.append(v)

    if len(seen) != 5:
        return False, {"why": "ring_graph_disconnected", "seen": len(seen), "degrees": degs}

    return True, {"why": "ok", "edge_count": edge_count, "degrees": degs}


def is_azapentacene_space(mol, require_all_aromatic=True, return_reason=False):
    """
    Nat. Commun. s41467-025-65564-8 azapentacene space:
      - exactly 5 rings (SSSR), each ring is 6-membered
      - linear fused topology (pentacene-like): ring graph is a path
      - only C/N heavy atoms (aza substitution)
      - heavy atoms == 22
      - all heavy atoms are in rings
      - optional: all heavy atoms aromatic
      - each atom belongs to <= 2 rings

    IMPORTANT: call this on an aromatized mol (see aromatize_for_key),
               otherwise aromatic checks may reject everything.
    """
    def _ret(ok, reason, detail=None):
        if return_reason:
            return ok, reason, detail
        return ok

    if mol is None:
        return _ret(False, "mol_none", None)

    mh = Chem.RemoveHs(mol)

    # only C/N
    for a in mh.GetAtoms():
        if a.GetSymbol() not in ("C", "N"):
            return _ret(False, "not_only_C_or_N", {"bad_atom": (a.GetIdx(), a.GetSymbol())})

    # pentacene skeleton has 22 heavy atoms
    if mh.GetNumAtoms() != 22:
        return _ret(False, "heavy_atom_count_not_22", {"heavy_atoms": mh.GetNumAtoms()})

    # ring perception
    try:
        Chem.GetSymmSSSR(mh)
    except Exception as e:
        return _ret(False, "sssr_failed", {"err": str(e)})

    ri = mh.GetRingInfo()
    if ri is None:
        return _ret(False, "ringinfo_none", None)

    atom_rings = list(ri.AtomRings())
    if len(atom_rings) != 5:
        return _ret(False, "ring_count_not_5", {"ring_count": len(atom_rings)})

    lens = [len(r) for r in atom_rings]
    if any(L != 6 for L in lens):
        return _ret(False, "not_all_rings_6_membered", {"ring_lengths": lens})

    # all heavy atoms must be in rings (+ optional aromatic)
    for a in mh.GetAtoms():
        if not a.IsInRing():
            return _ret(False, "has_non_ring_heavy_atom", {"atom": (a.GetIdx(), a.GetSymbol())})
        if require_all_aromatic and (not a.GetIsAromatic()):
            return _ret(False, "not_all_heavy_atoms_aromatic", {"atom": (a.GetIdx(), a.GetSymbol())})

    # each atom in <= 2 rings
    membership = {}
    for r in atom_rings:
        for idx in r:
            membership[idx] = membership.get(idx, 0) + 1
    if membership and max(membership.values()) > 2:
        return _ret(False, "atom_in_more_than_2_rings", {"max_membership": max(membership.values())})

    # linear fused topology
    ok_path, info = _ring_adjacency_is_linear_fused(atom_rings)
    if not ok_path:
        return _ret(False, "ring_adjacency_not_linear_path", info)

    return _ret(True, "ok", info)
