
#!/usr/bin/env python3
# inspect_c2db_commented.py

# --------------------------------------
# This script inspects and visualizes all
# tables in the C2DB SQLite database.
# It:
#  - Auto-discovers c2db.db in ../databases or ../backup
#  - Allows override via --db /path/to/other.db
#  - Lists tables & prints schemas
#  - Loads each table into pandas, prints info() & head()
#  - Saves per-table histograms to results/*.png
#  - Bundles all histograms into results/tables_histograms.pdf
# --------------------------------------

# --------------------------------------
# Standard library imports
import sys                      # For command-line arguments and exiting on error
from pathlib import Path       # For filesystem path manipulations

# Third-party imports
import numpy as np             # For numerical operations and array handling
import pandas as pd            # For DataFrame structures and SQL table reading
import sqlalchemy              # For database engine creation and inspection
import matplotlib.pyplot as plt  # For plotting histograms
from matplotlib.backends.backend_pdf import PdfPages  # For bundling multiple plots into a PDF


def get_db_path(cmdline_arg: str = None) -> Path:
    """
    Locate the SQLite database file to use.
    If a command-line path is given and exists, use it; otherwise search in default locations.
    """
    # If user provided override path, validate it
    if cmdline_arg:
        db = Path(cmdline_arg)  # Convert string to Path
        if db.exists():         # Check that the file exists
            return db           # Return the valid override
        else:
            sys.exit(f"ERROR: override path not found: {db}")  # Exit if invalid override

    # Determine project root based on this script's location
    project_root = Path(__file__).resolve().parent.parent

    # Candidate paths to search for the database file
    candidates = [
        project_root / 'databases' / 'c2db.db',
        project_root / 'backup'    / 'c2db.db',
        project_root / 'databases' / 'ac2db.db',
    ]
    # Iterate over candidates and return the first existing file
    for p in candidates:
        if p.exists():
            return p

    # If none found, abort with error listing attempted locations
    sys.exit(
        "ERROR: could not find c2db.db in:\n" +
        "\n".join(f"  • {p}" for p in candidates)
    )


def list_tables(engine):
    """
    List all table names in the connected database.
    """
    inspector = sqlalchemy.inspect(engine)  # Create an inspector object for the engine
    return inspector.get_table_names()      # Retrieve and return table names


def print_schema(engine, table: str) -> None:
    """
    Print the schema (column names and types) for a specific table.
    """
    cols = sqlalchemy.inspect(engine).get_columns(table)  # Get column metadata
    print(f"\nTable `{table}` schema:")
    # Iterate columns and display name and SQL type
    for c in cols:
        print(f"  – {c['name']} : {c['type']}")


def load_and_describe(engine, table: str) -> pd.DataFrame:
    """
    Load a table into a pandas DataFrame, print its info() and first rows.
    """
    print(f"\nLoading `{table}`…")
    # Read SQL table into DataFrame
    df = pd.read_sql_table(table, engine)
    # Print DataFrame summary (types, non-null counts)
    print(df.info(), end="\n\n")
    # Print first few rows for inspection
    print(df.head(), end="\n\n")
    return df  # Return DataFrame for further use


def save_histogram(df: pd.DataFrame, table: str, out_png: Path) -> Path | None:
    """
    Generate and save histograms for all numeric columns in DataFrame.
    Returns the PNG path if plots were saved, otherwise None.
    """
    # Select only numeric columns for histogramming
    num = df.select_dtypes(include='number')
    if num.empty:
        # No numeric data to plot
        print(f"No numeric columns in `{table}` to plot.")
        return None

    # Create subplots: one row per numeric column
    fig, axs = plt.subplots(
        nrows=len(num.columns), ncols=1,
        figsize=(8, max(2, len(num.columns)) * 2),  # Adjust height per column
        tight_layout=True
    )

    # Ensure axs is a flat list of Axes
    if isinstance(axs, np.ndarray):
        axes = axs.flatten().tolist()
    else:
        axes = [axs]

    # Plot histogram for each numeric column
    for ax, col in zip(axes, num.columns):
        ax.hist(df[col].dropna(), bins=20)              # Plot histogram, dropping NaNs
        ax.set_title(f"{table} — {col}")               # Title indicating table and column
        ax.set_ylabel("count")                        # Y-axis label

    # Overall figure title above subplots
    fig.suptitle(f"Histograms for `{table}`", y=1.02)
    plt.tight_layout()                                  # Adjust layout to prevent overlap
    fig.savefig(out_png)                                # Save figure to file
    plt.close(fig)                                      # Close to free memory

    print(f"→ saved histogram PNG: {out_png}")
    return out_png  # Return path to saved image


def main() -> None:
    """
    Main execution flow:
      1. Determine database path
      2. Create results directory
      3. Connect to SQLite engine
      4. Iterate tables: print schema, load/describe, save histograms
      5. Bundle PNGs into a single PDF
    """
    # Locate the database, allowing CLI override
    db_path = get_db_path(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Using database: {db_path}\n")

    # Prepare results directory next to the database folder
    project_root = db_path.parent.parent
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)  # Create directory if not already present

    # Create a SQLAlchemy engine for SQLite
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")

    # List all tables in the database
    tables = list_tables(engine)
    print("Found tables:", tables)

    png_paths = []  # To collect paths of generated PNG histograms
    for tbl in tables:
        print_schema(engine, tbl)                             # Print column schema
        df = load_and_describe(engine, tbl)                  # Load into pandas and describe
        png = save_histogram(df, tbl, results_dir / f"{tbl}_histograms.png")
        if png:
            png_paths.append(png)                            # Collect PNG if created

    # If any histograms were saved, bundle them into a PDF
    if png_paths:
        pdf_path = results_dir / 'tables_histograms.pdf'
        with PdfPages(pdf_path) as pdf:
            for png in png_paths:
                fig = plt.figure()
                img = plt.imread(str(png))                  # Read PNG image file
                plt.imshow(img)                             # Display image on a plot
                plt.axis('off')                             # Hide axes for clean look
                pdf.savefig(fig)                            # Save current figure to PDF
                plt.close(fig)
        print(f"\n→ bundled all histograms into PDF: {pdf_path}")

# Standard Python idiom to execute main() when run as a script
if __name__ == '__main__':
    main()

