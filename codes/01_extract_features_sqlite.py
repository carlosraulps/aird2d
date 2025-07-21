
#!/usr/bin/env python3
# pivot_merge_features_commented.py

# --------------------------------------
# This script loads tables from the C2DB SQLite database,
# pivots key-value and species data into feature matrices,
# merges them with system metadata, inspects missing data,
# and saves the final feature matrix to disk.
#
# Steps:
# 1) Locate or override database path via CLI
# 2) Load tables into pandas DataFrames
# 3) Build full feature DataFrame via pivots and joins
# 4) Inspect missingness and save as Parquet
# --------------------------------------

# Standard library imports
import sys  # For command-line argument handling and exiting on error
from pathlib import Path  # For filesystem path manipulation

# Third-party imports
import pandas as pd  # For DataFrame operations and SQL table reading
from sqlalchemy import create_engine  # For database engine creation


# Function to determine database path
# Allows override with first CLI argument or auto-search in project directories
# Exits with error if the file cannot be found

def get_db_path() -> Path:
    """
    Determine path to c2db.db:
      - If a CLI argument is provided, use it (must exist)
      - Otherwise search in ../databases/c2db.db and ../backup/c2db.db
    """
    # If user provided a path override via CLI
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])  # Convert argument string to Path
        if p.exists():
            return p  # Return if the file exists
        sys.exit(f"ERROR: DB not found at {p}")  # Exit on invalid override

    # No override: locate relative to this script's location
    root = Path(__file__).resolve().parent.parent
    # Potential database locations
    candidates = (
        root / 'databases' / 'c2db.db',
        root / 'backup' / 'c2db.db'
    )
    # Return the first existing candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # All attempts failed
    sys.exit("ERROR: could not locate c2db.db")


# Function to load required tables into pandas DataFrames

def load_tables(db_path: Path) -> dict:
    """
    Load the 'systems', 'number_key_values',
    'text_key_values', and 'species' tables into a dict of DataFrames.
    """
    # Create a SQLite engine pointing to the provided database
    engine = create_engine(f"sqlite:///{db_path}")
    # Table names to load
    table_names = ['systems', 'number_key_values', 'text_key_values', 'species']
    # Read each table via pandas and store in a dict
    tables = {name: pd.read_sql_table(name, engine) for name in table_names}
    return tables


# Function to build the full feature DataFrame

def build_feature_df(dfs: dict, Z2sym: dict) -> pd.DataFrame:
    """
    Construct a feature matrix by:
      a) Setting system metadata as the base index
      b) Pivoting numeric key-values into numeric features
      c) Pivoting text key-values into text features
      d) Pivoting species counts into elemental composition features
      e) Joining all into a single DataFrame
    """
    # a) System metadata with 'id' set as index
    df_sys = dfs['systems'].set_index('id')

    # b) Numeric features: pivot number_key_values rows->columns
    df_nkv = dfs['number_key_values']
    X_num = df_nkv.pivot(index='id', columns='key', values='value')

    # c) Text features: pivot text_key_values similarly
    df_tkv = dfs['text_key_values']
    X_txt = df_tkv.pivot(index='id', columns='key', values='value')

    # d) Composition features: map atomic numbers to symbols and pivot
    df_sp = dfs['species'].assign(elem=lambda d: d['Z'].map(Z2sym))
    X_comp = df_sp.pivot_table(
        index='id', columns='elem', values='n', fill_value=0
    )

    # e) Merge all feature sets with metadata on index 'id'
    df_full = df_sys.join([X_num, X_txt, X_comp], how='left')
    return df_full


# Function to inspect missing data and save the DataFrame

def inspect_and_save(df_full: pd.DataFrame, out_path: Path) -> None:
    """
    Print shape and top missing fractions,
    then save the DataFrame as Parquet.
    """
    # Print overall dimensions of the feature matrix
    print("\nFull feature matrix shape:", df_full.shape)
    # Compute fraction of missing per column
    miss_frac = df_full.isna().mean().sort_values(ascending=False)
    # Display top 10 columns with highest missing percentage
    print("\nTop 10 columns by % missing:\n", miss_frac.head(10))

    # Ensure parent directory exists (create if necessary)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    # Save DataFrame in Parquet format for efficient storage
    df_full.to_parquet(out_path.with_suffix('.parquet'))
    print(f"\nSaved feature matrix to {out_path.with_suffix('.parquet')}")


# Script entry point
if __name__ == '__main__':
    # Step 1: Get database path
    db_path = get_db_path()
    print("Using DB:", db_path)

    # Prepare mapping from atomic number to element symbol
    Z2sym = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S',
        # Add all atomic numbers 1–118 as needed
    }

    # Step 2: Load tables into DataFrames
    dfs = load_tables(db_path)
    # Step 3: Build the full feature DataFrame
    df_full = build_feature_df(dfs, Z2sym)
    # Construct output path for features
    out = Path(db_path).parent.parent / 'results' / 'features_full'
    # Step 4: Inspect missingness and save the final DataFrame
    inspect_and_save(df_full, out)

