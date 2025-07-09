
#!/usr/bin/env python3
# inspect_c2db.py

"""
Inspect and visualize all tables in the C2DB SQLite database.

– Auto-discovers c2db.db in ../databases or ../backup
– Allows --db /path/to/other.db override
– Lists tables & schemas
– Loads into pandas, prints .info() and .head()
– Saves per-table histograms to results/*.png
– Bundles all histograms into results/tables_histograms.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_db_path(cmdline_arg: str = None) -> Path:
    if cmdline_arg:
        db = Path(cmdline_arg)
        if db.exists():
            return db
        else:
            sys.exit(f"ERROR: override path not found: {db}")
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / 'databases' / 'c2db.db',
        project_root / 'backup'    / 'c2db.db',
        project_root / 'databases' / 'ac2db.db',
    ]
    for p in candidates:
        if p.exists():
            return p
    sys.exit(
        "ERROR: could not find c2db.db in:\n" +
        "\n".join(f"  • {p}" for p in candidates)
    )

def list_tables(engine):
    return sqlalchemy.inspect(engine).get_table_names()

def print_schema(engine, table):
    cols = sqlalchemy.inspect(engine).get_columns(table)
    print(f"\nTable `{table}` schema:")
    for c in cols:
        print(f"  – {c['name']} : {c['type']}")

def load_and_describe(engine, table):
    print(f"\nLoading `{table}`…")
    df = pd.read_sql_table(table, engine)
    print(df.info(), end="\n\n")
    print(df.head(), end="\n\n")
    return df

def save_histogram(df: pd.DataFrame, table: str, out_png: Path) -> Path | None:
    num = df.select_dtypes(include='number')
    if num.empty:
        print(f"No numeric columns in `{table}` to plot.")
        return None

    # create one subplot per numeric column
    fig, axs = plt.subplots(
        nrows=len(num.columns), ncols=1,
        figsize=(8, max(2, len(num.columns))*2),
        tight_layout=True
    )

    # flatten into a Python list of Axes
    if isinstance(axs, np.ndarray):
        axes = axs.flatten().tolist()
    else:
        axes = [axs]

    for ax, col in zip(axes, num.columns):
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"{table} — {col}")
        ax.set_ylabel("count")

    fig.suptitle(f"Histograms for `{table}`", y=1.02)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"→ saved histogram PNG: {out_png}")
    return out_png

def main():
    # locate DB (allow override via CLI)
    db_path = get_db_path(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Using database: {db_path}\n")

    project_root = db_path.parent.parent
    results_dir  = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    tables = list_tables(engine)
    print("Found tables:", tables)

    png_paths = []
    for tbl in tables:
        print_schema(engine, tbl)
        df  = load_and_describe(engine, tbl)
        png = save_histogram(df, tbl, results_dir / f"{tbl}_histograms.png")
        if png:
            png_paths.append(png)

    # bundle all PNGs into a single PDF
    if png_paths:
        pdf_path = results_dir / 'tables_histograms.pdf'
        with PdfPages(pdf_path) as pdf:
            for png in png_paths:
                fig = plt.figure()
                img = plt.imread(str(png))
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        print(f"\n→ bundled all histograms into PDF: {pdf_path}")

if __name__ == '__main__':
    main()

