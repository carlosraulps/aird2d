
#!/usr/bin/env python3
# test-db.py
"""
Connect to the SQLite database at databases/c2db.db, extract PBE, HSE06, and GW band gaps,
and generate a scatter plot (gaps.png) in the results/ directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ase.db import connect

def main():
    # Resolve the project root (two levels up from this script)
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / 'databases' / 'c2db.db'

    # Check that the database file exists
    if not db_path.is_file():
        sys.exit(f'ERROR: Database file not found at {db_path}')

    # Connect to the database and select all entries in the gap_gw table
    db = connect(str(db_path))
    rows = db.select('gap_gw')

    # Collect gap values
    pbe_gaps = []
    hse_gaps = []
    gw_gaps  = []
    for row in rows:
        # Expect columns: gap (PBE), gap_hse (HSE06), gap_gw (GW)
        pbe_gaps.append(row.gap)
        hse_gaps.append(row.gap_hse)
        gw_gaps.append(row.gap_gw)

    if not pbe_gaps:
        sys.exit('ERROR: No data retrieved from the database.')

    # Prepare the output directory
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(pbe_gaps, hse_gaps, 'o', label='HSE06 vs PBE')
    ax.plot(pbe_gaps, gw_gaps,  'x', label='GW vs PBE')
    max_pbe = max(pbe_gaps)
    ax.plot([0, max_pbe], [0, max_pbe], '-', label='Ideal (y = x)')
    ax.set_xlabel('PBE Band Gap [eV]')
    ax.set_ylabel('Band Gap [eV]')
    ax.legend()
    fig.tight_layout()

    # Save the figure
    output_file = results_dir / 'gaps.pdf'
    fig.savefig(output_file)
    print(f'Plot saved to: {output_file}')

if __name__ == '__main__':
    main()

