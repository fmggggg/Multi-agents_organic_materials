from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import gemmi
from pydantic import BaseModel, Field


class PackedMolecule(BaseModel):
    molecule_id: str
    atom_symbols: list[str] = Field(default_factory=list)
    coordinates: list[list[float]] = Field(default_factory=list)
    centroid: list[float]
    atom_count: int


class DimerRecord(BaseModel):
    dimer_id: str
    molecule_a: str
    molecule_b: str
    centroid_distance: float
    min_heavy_atom_distance: float | None = None
    transport_proxy_score: float | None = None
    dimer_xyz_path: str = ""
    monomer_a_xyz_path: str = ""
    monomer_b_xyz_path: str = ""


class ClusterRecord(BaseModel):
    cluster_id: str
    central_molecule: str
    member_molecules: list[str] = Field(default_factory=list)
    cluster_xyz_path: str = ""


class PackingSummary(BaseModel):
    source_cif: str
    molecule_count: int
    dimers: list[DimerRecord] = Field(default_factory=list)
    clusters: list[ClusterRecord] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def extract_packing_features(
    cif_path: str | Path,
    out_dir: str | Path,
    *,
    max_dimers: int = 12,
    cluster_size: int = 4,
) -> PackingSummary:
    source = Path(cif_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    dimers_dir = target / "dimers"
    clusters_dir = target / "clusters"
    monomers_dir = target / "monomers"
    dimers_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    monomers_dir.mkdir(parents=True, exist_ok=True)

    molecules = _extract_molecules(source)
    artifact_paths: list[str] = []

    for molecule in molecules:
        mono_path = monomers_dir / f"{molecule.molecule_id}.xyz"
        _write_xyz(mono_path, molecule.atom_symbols, molecule.coordinates, comment=molecule.molecule_id)
        artifact_paths.append(str(mono_path))

    dimers = _build_dimers(molecules, max_dimers=max_dimers, dimers_dir=dimers_dir, monomers_dir=monomers_dir)
    clusters = _build_clusters(molecules, cluster_size=cluster_size, clusters_dir=clusters_dir)

    dimers_csv = target / "dimers.csv"
    with dimers_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dimer_id",
                "molecule_a",
                "molecule_b",
                "centroid_distance",
                "min_heavy_atom_distance",
                "transport_proxy_score",
                "dimer_xyz_path",
                "monomer_a_xyz_path",
                "monomer_b_xyz_path",
            ],
        )
        writer.writeheader()
        writer.writerows(item.model_dump() for item in dimers)
    artifact_paths.append(str(dimers_csv))

    clusters_json = target / "clusters.json"
    clusters_json.write_text(json.dumps([item.model_dump() for item in clusters], indent=2), encoding="utf-8")
    artifact_paths.append(str(clusters_json))

    summary = PackingSummary(
        source_cif=str(source),
        molecule_count=len(molecules),
        dimers=dimers,
        clusters=clusters,
        artifact_paths=artifact_paths
        + [item.dimer_xyz_path for item in dimers]
        + [item.cluster_xyz_path for item in clusters],
        notes=[
            "Packing features were extracted from per-molecule coordinates in the CIF/MMCIF file.",
            "Transport proxy currently uses centroid distance and minimum heavy-atom contact as a geometric heuristic.",
        ],
    )
    (target / "packing_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def _extract_molecules(cif_path: Path) -> list[PackedMolecule]:
    """Extract per-molecule Cartesian coordinates from a CIF/mmCIF file.

    The packing workflow sees both:
      1) mmCIF/PDB-like coordinate models (handled by :func:`gemmi.read_structure`), and
      2) small-molecule CIF exports with multiple data blocks and CIF-style tags
         (handled best-effort via :mod:`gemmi.cif` + :func:`gemmi.make_small_structure_from_block`).

    This function must be *best-effort* and never crash the planner.
    """

    # Path 1: macromolecular-style structure
    try:
        structure = gemmi.read_structure(str(cif_path))
    except Exception:
        structure = gemmi.Structure()

    # gemmi can return an empty Structure (len==0) for CIFs that don't contain
    # a parsable coordinate model.
    if len(structure) != 0:
        try:
            model = structure[0]
        except Exception:
            model = None

        if model is not None:
            molecules: list[PackedMolecule] = []
            group_index = 1
            for chain in model:
                residues = list(chain)
                if residues:
                    for residue in residues:
                        symbols: list[str] = []
                        coords: list[list[float]] = []
                        for atom in residue:
                            symbols.append(atom.element.name)
                            coords.append([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])
                        if coords:
                            molecules.append(
                                PackedMolecule(
                                    molecule_id=f"{chain.name or 'chain'}_{residue.seqid.num or group_index}",
                                    atom_symbols=symbols,
                                    coordinates=coords,
                                    centroid=_centroid(coords),
                                    atom_count=len(coords),
                                )
                            )
                            group_index += 1
                else:
                    atoms = list(chain.get_polymer())  # pragma: no cover - fallback rarely hit
                    if atoms:
                        coords = [[float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)] for atom in atoms]
                        molecules.append(
                            PackedMolecule(
                                molecule_id=f"{chain.name or 'chain'}_{group_index}",
                                atom_symbols=[atom.element.name for atom in atoms],
                                coordinates=coords,
                                centroid=_centroid(coords),
                                atom_count=len(coords),
                            )
                        )
                        group_index += 1

            if molecules:
                return molecules

            atoms = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom)
            if atoms:
                coords = [[float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)] for atom in atoms]
                return [
                    PackedMolecule(
                        molecule_id="mol_001",
                        atom_symbols=[atom.element.name for atom in atoms],
                        coordinates=coords,
                        centroid=_centroid(coords),
                        atom_count=len(coords),
                    )
                ]

    # Path 2: small-molecule CIF (possibly multi-block). Use the first block.
    return _extract_molecules_from_small_cif(cif_path)


def _extract_molecules_from_small_cif(cif_path: Path) -> list[PackedMolecule]:
    try:
        doc = gemmi.cif.read_file(str(cif_path))
    except Exception:
        return []
    if len(doc) == 0:
        return []

    try:
        block = doc[0]
        small = gemmi.make_small_structure_from_block(block)
    except Exception:
        return []

    symbols: list[str] = []
    coords: list[list[float]] = []
    for site in small.sites:
        sym = site.element.name if site.element.name and site.element.name != "X" else site.type_symbol
        sym = (sym or site.type_symbol).strip()
        if not sym:
            continue
        pos = small.cell.orthogonalize(site.fract)
        symbols.append(sym)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])

    if not coords:
        return []

    components = _connected_components(symbols, coords)
    molecules: list[PackedMolecule] = []
    for idx, comp in enumerate(components, start=1):
        if not comp:
            continue
        comp_symbols = [symbols[i] for i in comp]
        comp_coords = [coords[i] for i in comp]
        molecules.append(
            PackedMolecule(
                molecule_id=f"mol_{idx:03d}",
                atom_symbols=comp_symbols,
                coordinates=comp_coords,
                centroid=_centroid(comp_coords),
                atom_count=len(comp_coords),
            )
        )

    return molecules


def _connected_components(symbols: list[str], coords: list[list[float]]) -> list[list[int]]:
    """Connected components using a covalent-radius distance heuristic."""

    radii = [_covalent_radius(sym) for sym in symbols]
    n = len(coords)
    adj: list[list[int]] = [[] for _ in range(n)]
    tol = 0.45

    for i in range(n):
        xi, yi, zi = coords[i]
        ri = radii[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            cutoff = ri + radii[j] + tol
            if d2 <= cutoff * cutoff:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(comp)

    # Sort: larger molecules first for determinism
    components.sort(key=len, reverse=True)
    return components


def _covalent_radius(symbol: str) -> float:
    table = {
        "H": 0.31,
        "B": 0.85,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
        "Si": 1.11,
    }
    # Normalize element symbols like 'CL' -> 'Cl'
    sym = symbol.strip()
    if len(sym) > 1:
        sym = sym[0].upper() + sym[1:].lower()
    else:
        sym = sym.upper()
    return table.get(sym, 0.77)


def _build_dimers(
    molecules: list[PackedMolecule],
    *,
    max_dimers: int,
    dimers_dir: Path,
    monomers_dir: Path,
) -> list[DimerRecord]:
    pairs: list[DimerRecord] = []
    counter = 1
    for i, left in enumerate(molecules):
        for right in molecules[i + 1 :]:
            centroid_distance = _distance(left.centroid, right.centroid)
            min_heavy = _min_heavy_atom_distance(left, right)
            transport_proxy = _transport_proxy_score(centroid_distance, min_heavy)
            dimer_id = f"dimer-{counter:04d}"
            dimer_xyz = dimers_dir / f"{dimer_id}.xyz"
            _write_xyz(
                dimer_xyz,
                left.atom_symbols + right.atom_symbols,
                left.coordinates + right.coordinates,
                comment=f"{left.molecule_id}+{right.molecule_id}",
            )
            pairs.append(
                DimerRecord(
                    dimer_id=dimer_id,
                    molecule_a=left.molecule_id,
                    molecule_b=right.molecule_id,
                    centroid_distance=round(centroid_distance, 6),
                    min_heavy_atom_distance=round(min_heavy, 6) if min_heavy is not None else None,
                    transport_proxy_score=round(transport_proxy, 6) if transport_proxy is not None else None,
                    dimer_xyz_path=str(dimer_xyz),
                    monomer_a_xyz_path=str(monomers_dir / f"{left.molecule_id}.xyz"),
                    monomer_b_xyz_path=str(monomers_dir / f"{right.molecule_id}.xyz"),
                )
            )
            counter += 1
    pairs.sort(
        key=lambda item: (
            item.centroid_distance,
            -(item.transport_proxy_score or -1e9),
            item.dimer_id,
        )
    )
    return pairs[:max_dimers]


def _build_clusters(
    molecules: list[PackedMolecule],
    *,
    cluster_size: int,
    clusters_dir: Path,
) -> list[ClusterRecord]:
    clusters: list[ClusterRecord] = []
    for index, central in enumerate(molecules, start=1):
        neighbors = sorted(
            [item for item in molecules if item.molecule_id != central.molecule_id],
            key=lambda item: _distance(central.centroid, item.centroid),
        )[: max(0, cluster_size - 1)]
        members = [central, *neighbors]
        xyz_path = clusters_dir / f"cluster-{index:04d}.xyz"
        symbols: list[str] = []
        coords: list[list[float]] = []
        for molecule in members:
            symbols.extend(molecule.atom_symbols)
            coords.extend(molecule.coordinates)
        _write_xyz(
            xyz_path,
            symbols,
            coords,
            comment=";".join(item.molecule_id for item in members),
        )
        clusters.append(
            ClusterRecord(
                cluster_id=f"cluster-{index:04d}",
                central_molecule=central.molecule_id,
                member_molecules=[item.molecule_id for item in members],
                cluster_xyz_path=str(xyz_path),
            )
        )
    return clusters


def _centroid(coords: list[list[float]]) -> list[float]:
    n = len(coords)
    return [sum(point[idx] for point in coords) / n for idx in range(3)]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _min_heavy_atom_distance(left: PackedMolecule, right: PackedMolecule) -> float | None:
    heavy_left = [
        coord
        for symbol, coord in zip(left.atom_symbols, left.coordinates, strict=True)
        if symbol.upper() != "H"
    ]
    heavy_right = [
        coord
        for symbol, coord in zip(right.atom_symbols, right.coordinates, strict=True)
        if symbol.upper() != "H"
    ]
    if not heavy_left or not heavy_right:
        return None
    return min(_distance(a, b) for a in heavy_left for b in heavy_right)


def _transport_proxy_score(centroid_distance: float, min_heavy_atom_distance: float | None) -> float | None:
    if min_heavy_atom_distance is None:
        return None
    centroid_term = math.exp(-max(0.0, centroid_distance - 3.5) / 2.0)
    contact_term = math.exp(-max(0.0, min_heavy_atom_distance - 3.3) / 1.5)
    return 0.5 * centroid_term + 0.5 * contact_term


def _write_xyz(path: Path, symbols: list[str], coords: list[list[float]], *, comment: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(symbols)}\n")
        fh.write(f"{comment}\n")
        for symbol, coord in zip(symbols, coords, strict=True):
            fh.write(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n")
