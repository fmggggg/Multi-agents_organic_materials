"""Azapentacene chemical space: skeleton enumeration, site-based aza substitution,
space gate (the original AlphaPack RDKit rules) and duplicate keys.

The space is the original rule set from azapentacene_space.py (22 C/N ring atoms,
five 6-membered rings fused as an unbranched path, all aromatic, <=2 rings/atom)
plus at least one N. All 10 unbranched catacondensed five-ring benzenoid
skeletons pass that rule; N may replace any of the 14 C-H positions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from .azapentacene_space import (
    aromatize_for_key,
    is_azapentacene_space,
    make_aroma_key,
    make_graph_key,
    to_mol_safe,
)

RDLogger.DisableLog("rdApp.*")

from .config import KNOWN_CSV as _KNOWN_CSV

KNOWN_CSV = Path(_KNOWN_CSV)
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]  # axial hex directions, cyclic
NAMES = {  # names only where the annelation code is unambiguous
    "LLL": "pentacene",
    "ALL": "benzo[a]tetracene",
    "LAL": "pentaphene",
    "ALA": "dibenz[a,j]anthracene",
    "ALB": "dibenz[a,h]anthracene",
    "ABA": "picene",
    "AAA": "[5]helicene",
}


@dataclass(frozen=True)
class Skeleton:
    code: str             # annelation code of the 3 inner rings: L linear, A/B angular (same/opposite turn)
    name: str
    smiles: str           # all-carbon parent, canonical
    mapped_smiles: str    # C-H sites carry atom-map numbers 1..14 (perimeter order)
    mol: Chem.Mol         # parent mol; atom prop 'site' on C-H atoms


def _canon_code(turns: list[int]) -> str:
    def enc(seq):
        return "".join("L" if t == 0 else ("A" if t == 1 else "B") for t in seq)

    variants = []
    for seq in (turns, turns[::-1]):
        for sign in (1, -1):
            s = [sign * t for t in seq]
            # relabel so that the first angular turn is 'A'
            first = next((t for t in s if t != 0), 1)
            variants.append(enc([t * first for t in s]))
    return min(variants)


def _hex_vertices(q: int, r: int):
    cx, cy = math.sqrt(3) * (q + r / 2), 1.5 * r
    return [(round((cx + math.cos(math.radians(30 + 60 * k))) * 1000),
             round((cy + math.sin(math.radians(30 + 60 * k))) * 1000)) for k in range(6)]


def _build_skeleton(path: list[tuple[int, int]], turns: list[int]) -> Skeleton:
    hexes = [_hex_vertices(*h) for h in path]
    verts = sorted({v for h in hexes for v in h})
    index = {v: i for i, v in enumerate(verts)}
    edge_count: dict[tuple[int, int], int] = {}
    for h in hexes:
        for k in range(6):
            a, b = index[h[k]], index[h[(k + 1) % 6]]
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1
    rw = Chem.RWMol()
    for _ in verts:
        a = Chem.Atom(6)
        a.SetIsAromatic(True)
        rw.AddAtom(a)
    for a, b in edge_count:
        rw.AddBond(a, b, Chem.BondType.AROMATIC)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    # perimeter = cycle of boundary edges (all 22 atoms for a catacondensed system)
    nbr: dict[int, list[int]] = {i: [] for i in range(len(verts))}
    for (a, b), c in edge_count.items():
        if c == 1:
            nbr[a].append(b)
            nbr[b].append(a)
    cycle = [0, nbr[0][0]]
    while len(cycle) < len(verts):
        nxt = [x for x in nbr[cycle[-1]] if x != cycle[-2]][0]
        cycle.append(nxt)
    deg = [mol.GetAtomWithIdx(i).GetDegree() for i in range(len(verts))]
    # canonical start/direction: lexicographically smallest perimeter degree string starting on a C-H
    best = None
    for direction in (cycle, cycle[::-1]):
        for s in range(len(direction)):
            rot = direction[s:] + direction[:s]
            if deg[rot[0]] != 2 or deg[rot[-1]] != 3:
                continue
            key = tuple(deg[i] for i in rot)
            if best is None or key < best[0]:
                best = (key, rot)
    order = best[1]
    site = 0
    for i in order:
        if deg[i] == 2:
            site += 1
            mol.GetAtomWithIdx(i).SetIntProp("site", site)
    mapped = Chem.Mol(mol)
    for a in mapped.GetAtoms():
        if a.HasProp("site"):
            a.SetAtomMapNum(a.GetIntProp("site"))
    code = _canon_code(turns)
    return Skeleton(code=code, name=NAMES.get(code, ""), smiles=Chem.MolToSmiles(mol),
                    mapped_smiles=Chem.MolToSmiles(mapped), mol=mol)


@lru_cache(maxsize=1)
def skeletons() -> dict[str, Skeleton]:
    found: dict[str, Skeleton] = {}

    def grow(path, dirs):
        if len(path) == 5:
            turns = [(dirs[i] - dirs[i - 1] + 3) % 6 - 3 for i in range(1, 4)]
            if any(abs(t) > 1 for t in turns):
                return
            sk = _build_skeleton(path, turns)
            found.setdefault(sk.code, sk)
            return
        q, r = path[-1]
        for d, (dq, dr) in enumerate(DIRS):
            h = (q + dq, r + dr)
            if h in path:
                continue
            # catacondensed: new ring may only touch its predecessor
            if any(abs(h[0] - p[0]) + abs(h[1] - p[1]) + abs((h[0] + h[1]) - (p[0] + p[1])) == 2
                   for p in path[:-1]):
                continue
            grow(path + [h], dirs + [d])

    grow([(0, 0)], [])
    return dict(sorted(found.items()))


def build_from_sites(code: str, sites: list[int]) -> str:
    sks = skeletons()
    if code not in sks:
        raise ValueError(f"unknown skeleton code {code!r}; choose from {sorted(sks)}")
    sites = sorted(set(int(s) for s in sites))
    if not sites:
        raise ValueError("give at least one aza site (1..14)")
    if any(s < 1 or s > 14 for s in sites):
        raise ValueError("aza sites must be integers 1..14")
    rw = Chem.RWMol(sks[code].mol)
    for a in rw.GetAtoms():
        if a.HasProp("site") and a.GetIntProp("site") in sites:
            a.SetAtomicNum(7)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


def identify(mol) -> tuple[str, list[int]] | tuple[None, None]:
    """Skeleton code and canonical (smallest) aza-site list for an in-space molecule."""
    m = Chem.RemoveHs(mol)
    for code, sk in skeletons().items():
        q = Chem.RWMol(sk.mol)
        for a in q.GetAtoms():
            a.SetAtomicNum(0)  # dummy; compare topology only
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeDummiesQueries = True
        query = Chem.AdjustQueryProperties(q.GetMol(), params)
        matches = m.GetSubstructMatches(query, uniquify=False, useChirality=False)
        if not matches:
            continue
        if m.GetNumAtoms() != sk.mol.GetNumAtoms():
            continue
        best = None
        for match in matches:
            s = sorted(sk.mol.GetAtomWithIdx(qi).GetIntProp("site")
                       for qi, mi in enumerate(match)
                       if sk.mol.GetAtomWithIdx(qi).HasProp("site") and m.GetAtomWithIdx(mi).GetSymbol() == "N")
            n_fused_n = sum(1 for qi, mi in enumerate(match)
                            if not sk.mol.GetAtomWithIdx(qi).HasProp("site") and m.GetAtomWithIdx(mi).GetSymbol() == "N")
            if n_fused_n:
                continue
            if best is None or s < best:
                best = s
        if best is not None:
            return code, best
    return None, None


@lru_cache(maxsize=1)
def known_keys() -> tuple[frozenset, frozenset, int]:
    df = pd.read_csv(KNOWN_CSV)
    g, a = set(), set()
    for s in df["Bonded_SMILES"].dropna().astype(str):
        m = to_mol_safe(s)
        if m is None:
            continue
        gk, ak = make_graph_key(m), make_aroma_key(m)
        if gk:
            g.add(gk)
        if ak:
            a.add(ak)
    return frozenset(g), frozenset(a), len(df)


def check(smiles: str, campaign_graph_keys=(), campaign_aroma_keys=(), min_n: int = 1) -> dict:
    """Gate a proposal. Returns dict with ok, reason, canonical smiles, keys, skeleton/sites."""
    out = dict(input_smiles=smiles, ok=False)
    mol = to_mol_safe(smiles)
    if mol is None:
        return {**out, "reason": "smiles_parse_or_sanitize_failed"}
    amol = aromatize_for_key(mol)
    ok, reason, detail = is_azapentacene_space(amol, require_all_aromatic=True, return_reason=True)
    if not ok:
        return {**out, "reason": f"outside_space:{reason}", "detail": str(detail)}
    n_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    if n_n < min_n:
        return {**out, "reason": f"fewer_than_{min_n}_N"}
    if Chem.GetFormalCharge(mol) != 0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return {**out, "reason": "charged_or_radical"}
    gk, ak = make_graph_key(mol), make_aroma_key(mol)
    code, sites = identify(mol)
    canon = Chem.MolToSmiles(Chem.RemoveHs(mol))
    out.update(smiles=canon, graph_key=gk, aroma_key=ak, n_N=n_n,
               formula=rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol)),
               skeleton=code, sites=sites, label=f"{code}:{'-'.join(map(str, sites))}" if code else "")
    kg, ka, _ = known_keys()
    if gk in kg or ak in ka:
        return {**out, "reason": "duplicate_known_dataset"}
    if gk in set(campaign_graph_keys) or ak in set(campaign_aroma_keys):
        return {**out, "reason": "duplicate_already_tested"}
    return {**out, "ok": True, "reason": "ok"}


def describe_space() -> str:
    lines = []
    for code, sk in skeletons().items():
        nm = f" ({sk.name})" if sk.name else ""
        lines.append(f"- {code}{nm}: {sk.mapped_smiles}")
    return "\n".join(lines)
