"""Pre-energy structure checks: periodic severe-overlap gate (crystal_geometry_qc.py, an unmodified copy of
clari/scripts/crystal_geometry_qc.py; AZA_QC_SCRIPT points to another copy) and molecular-integrity check (each periodic
molecule must have the same element-labelled bond graph as the input SMILES)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import networkx as nx
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from rdkit import Chem

_QC_PATH = os.environ.get("AZA_QC_SCRIPT") or str(Path(__file__).with_name("crystal_geometry_qc.py"))
_spec = importlib.util.spec_from_file_location("crystal_geometry_qc", _QC_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
severe_overlap_check = _mod.severe_overlap_check
OVERLAP_FACTOR = 0.6
BOND_FACTOR = 1.15  # bonded if d < 1.15 * (r_i + r_j), ASE covalent radii


def reference_graph(smiles: str) -> nx.Graph:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    g = nx.Graph()
    for a in mol.GetAtoms():
        g.add_node(a.GetIdx(), el=a.GetSymbol())
    for b in mol.GetBonds():
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g


def integrity_check(atoms, ref: nx.Graph, z: int) -> dict:
    radii = covalent_radii[atoms.numbers]
    i, j, S, d = neighbor_list("ijSd", atoms, BOND_FACTOR * radii)
    bonded = d < BOND_FACTOR * (radii[i] + radii[j])
    keep = (i < j) & bonded
    g = nx.Graph()
    for k, a in enumerate(atoms):
        g.add_node(k, el=a.symbol)
    g.add_edges_from(zip(i[keep].tolist(), j[keep].tolist()))
    shifts: dict[int, list] = {}
    for a, b, s in zip(i[bonded].tolist(), j[bonded].tolist(), S[bonded]):
        shifts.setdefault(a, []).append((b, s))
    comps = [g.subgraph(c) for c in nx.connected_components(g)]
    if len(comps) != z:
        return dict(intact=False, reason=f"n_molecules_{len(comps)}_expected_{z}")
    match = nx.algorithms.isomorphism.categorical_node_match("el", None)
    for c in comps:
        if c.number_of_nodes() != ref.number_of_nodes() or c.number_of_edges() != ref.number_of_edges() \
                or not nx.is_isomorphic(c, ref, node_match=match):
            return dict(intact=False, reason="bond_graph_mismatch")
    # unwrap each molecule across periodic boundaries (for frozen-geometry intramolecular energies)
    cell, pos, mols = atoms.cell.array, atoms.positions, []
    for c in comps:
        root = min(c.nodes)
        P = {root: pos[root]}
        stack = [root]
        while stack:
            u = stack.pop()
            for v, s in shifts.get(u, []):
                if v not in P:
                    P[v] = P[u] + (pos[v] + s @ cell - pos[u])
                    stack.append(v)
        idx = sorted(P)
        mols.append(dict(symbols=[atoms[k].symbol for k in idx], positions=[P[k].tolist() for k in idx]))
    return dict(intact=True, reason="ok", molecules=mols)
