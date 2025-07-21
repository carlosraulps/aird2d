
#!/usr/bin/env python3
# extract_structural_commented.py

# --------------------------------------
# Script to extract structural descriptors from
# the C2DB ASE SQLite database and save as Parquet.
#
# Steps:
# 1) Connect to the ASE SQLite database
# 2) Iterate over each atomic system
# 3) Compute lattice parameters (a, b, c, α, β, γ)
# 4) Compute cell volume and atomic density
# 5) Build neighbor list and compute mean coordination number
# 6) Collect descriptors into a DataFrame and save
# --------------------------------------

# --------------------------------------
# Import necessary packages
from ase.db import connect            # Connect to ASE database interface
import numpy as np                   # Numerical operations, vector norms, and statistics
import pandas as pd                  # DataFrame handling and Parquet saving
from pathlib import Path             # Filesystem path manipulations
from ase.neighborlist import (
    NeighborList,                     # For constructing neighbor lists
    natural_cutoffs                   # To compute atomic cutoff radii automatically
)

# --------------------------------------
# 1) Connect to the ASE SQLite database
# Determine the path to c2db.db relative to this script
db_path = (
    Path(__file__).resolve()        # Absolute path of this script
    .parent.parent                  # Go up two levels to project root
    / 'databases' / 'c2db.db'       # Append 'databases/c2db.db'
)
# Establish connection via ASE's connect function
db = connect(db_path)

# Prepare a container for descriptor records
records = []  # List of dicts, one per system

# --------------------------------------
# 2) Iterate over each system entry in the database
# db.select() yields rows with .id and .toatoms() for structure data
for row in db.select():
    sid = row.id                 # Unique integer ID of the system
    atoms = row.toatoms()        # ASE Atoms object for the structure

    # --------------------------------------
    # 3) Compute lattice parameters
    # Get 3×3 cell matrix: row vectors give cell vectors a⃗, b⃗, c⃗
    cell = atoms.get_cell()
    # Compute lengths of each cell vector via Euclidean norm
    a, b, c = np.linalg.norm(cell, axis=1)
    # Compute angles between cell vectors using dot products:
    # α = angle between b⃗ & c⃗, β = between a⃗ & c⃗, γ = between a⃗ & b⃗
    alpha = np.degrees(
        np.arccos(np.dot(cell[1], cell[2]) / (b * c))
    )
    beta = np.degrees(
        np.arccos(np.dot(cell[0], cell[2]) / (a * c))
    )
    gamma = np.degrees(
        np.arccos(np.dot(cell[0], cell[1]) / (a * b))
    )

    # --------------------------------------
    # 4) Compute volume and density
    # Volume of the unit cell from ASE
    V = atoms.get_volume()
    # Total mass of all atoms in atomic mass units (amu)
    total_mass = atoms.get_masses().sum()
    # Density in amu per Å³
    rho = total_mass / V

    # --------------------------------------
    # 5) Build neighbor list for coordination analysis
    # natural_cutoffs returns covalent radii scaled by mult factor
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    # Initialize NeighborList: no self-interactions, symmetric pairs
    nl = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True
    )
    # Populate neighbor list with atomic positions
    nl.update(atoms)

    # --------------------------------------
    # 6) Compute coordination numbers for each atom
    cnumbers = []  # List of neighbor counts per atom index
    for i in range(len(atoms)):
        # get_neighbors returns arrays of neighbor indices and periodic offsets
        indices, offsets = nl.get_neighbors(i)
        # Coordination number = number of neighbors
        cnumbers.append(len(indices))
    # Mean coordination across atoms
    cn_mean = float(np.mean(cnumbers))

    # --------------------------------------
    # Append computed descriptors for this system to records
    records.append({
        'id': sid,
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': V,
        'density_amu_per_A3': rho,
        'cn_mean': cn_mean,
        'n_atoms': len(atoms),  # Number of atoms in the structure
    })

# --------------------------------------
# 7) Build DataFrame and save to Parquet
# Create DataFrame indexed by system ID
df_struc = pd.DataFrame(records).set_index('id')
# Determine output path relative to this script
out_path = (
    Path(__file__).resolve()
    .parent.parent
    / 'results'
    / 'structural_descriptors.parquet'
)
# Ensure results directory exists
out_path.parent.mkdir(exist_ok=True, parents=True)
# Save structural descriptors as Parquet for efficient storage
df_struc.to_parquet(out_path)

# Inform the user of the save location
print(f"Saved structural descriptors to {out_path}")

