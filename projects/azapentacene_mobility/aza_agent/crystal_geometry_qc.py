"""Conservative, element-aware periodic severe-overlap gate before relaxation."""

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


def severe_overlap_check(atoms, factor=.6, max_examples=8):
    if not 0 < factor < 1:
        raise ValueError('Overlap factor must be between zero and one')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('Nonfinite crystal coordinates or lattice')
    if atoms.pbc.any() and abs(np.linalg.det(atoms.cell.array)) < 1e-9:
        raise ValueError('Degenerate periodic cell')
    radii = covalent_radii[atoms.numbers]
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Missing positive covalent radius')
    i, j, image, distances = neighbor_list('ijSd', atoms, factor*radii)
    positive_image = ((image[:,0]>0) | ((image[:,0]==0) & (image[:,1]>0)) |
        ((image[:,0]==0) & (image[:,1]==0) & (image[:,2]>0)))
    unique = (i < j) | ((i == j) & positive_image)
    selected = np.flatnonzero(unique & (distances < factor*(radii[i]+radii[j])))
    selected = selected[np.argsort(distances[selected])]
    examples = []
    for k in selected[:max_examples]:
        a, b = int(i[k]), int(j[k])
        examples.append(dict(index_i=a, index_j=b, element_i=atoms[a].symbol,
            element_j=atoms[b].symbol, distance_angstrom=float(distances[k]),
            threshold_angstrom=float(factor*(radii[a]+radii[b])),
            covalent_ratio=float(distances[k]/(radii[a]+radii[b])),
            image_shift=image[k].tolist()))
    return dict(accepted=len(selected)==0, severe_overlap_pairs=len(selected),
        factor=factor, criterion='periodic distance < factor * sum of ASE covalent radii',
        min_flagged_distance_angstrom=None if not len(selected) else float(distances[selected[0]]),
        examples=examples)
