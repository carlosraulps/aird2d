
#!/usr/bin/env python3
# extract_structural.py

from ase.db import connect
import numpy as np
import pandas as pd
from pathlib import Path
from ase.neighborlist import NeighborList, natural_cutoffs

# 1) connect to the ASE SQLite DB
db_path = Path(__file__).resolve().parent.parent / 'databases' / 'c2db.db'
db = connect(db_path)

records = []

# 2) iterate over each system entry
for row in db.select():
    sid   = row.id
    atoms = row.toatoms()

    # 3) lattice parameters
    cell   = atoms.get_cell()
    a, b, c = np.linalg.norm(cell, axis=1)
    alpha  = np.degrees(np.arccos(np.dot(cell[1], cell[2])/(b*c)))
    beta   = np.degrees(np.arccos(np.dot(cell[0], cell[2])/(a*c)))
    gamma  = np.degrees(np.arccos(np.dot(cell[0], cell[1])/(a*b)))

    # 4) volume & density
    V    = atoms.get_volume()
    rho  = atoms.get_masses().sum() / V

    # 5) build and update neighbor list
    cutoffs = natural_cutoffs(atoms, mult=1.1)  # radii for each atom :contentReference[oaicite:7]{index=7}
    nl      = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # 6) compute coordination numbers via get_neighbors
    cnumbers = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)       # get_neighbors returns (indices, offsets) :contentReference[oaicite:8]{index=8}
        cnumbers.append(len(indices))
    cn_mean = float(np.mean(cnumbers))

    records.append({
        'id': sid,
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': V,
        'density_amu_per_A3': rho,
        'cn_mean': cn_mean,
        'n_atoms': len(atoms),
    })

# 7) build DataFrame & save
df_struc = pd.DataFrame(records).set_index('id')
out = Path(__file__).resolve().parent.parent / 'results' / 'structural_descriptors.parquet'
df_struc.to_parquet(out)
print(f"Saved structural descriptors to {out}")

