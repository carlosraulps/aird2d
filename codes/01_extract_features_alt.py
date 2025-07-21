
#!/usr/bin/env python3
"""
connect_load_db

Project structure:
.
├── codes/
├── results/
│   ├── nkv_profile.html
│   ├── materials_features.parquet (or .csv fallback)
├── data/
├── backup/
│   └── c2db.db
├── references/
│   ├── Haastrup_2018_2D_Mater._5_042002.pdf
│   ├── Recent progress of the C2DB.pdf
│   └── The C2DB.pdf
├── README.md
└── requirements.txt 

Description:
Correct usage of SQLAlchemy and pandas to load all tables from the SQLite database, pivot features,
profile numeric key-value pairs (if available), and assemble a full feature DataFrame for materials.
Handles missing profiling or parquet dependencies gracefully.

# --------------------------------------
# Script to inspect, profile, and assemble features
# from the C2DB SQLite database.
# It:
#   - Auto-discovers c2db.db in project directories
#   - Connects via SQLAlchemy
#   - Loads tables into pandas DataFrames
#   - Optionally generates HTML profiling reports
#   - Checks for Parquet support (pyarrow/fastparquet)
#   - Pivots key-value tables and species data into feature matrix
#   - Saves features to Parquet or CSV
#   - Demonstrates ASE database access for atom retrieval
# --------------------------------------

"""
# --------------------------------------
# Standard library imports
import sys                      # For handling CLI args and exiting on error
import logging                  # For configurable logging of script progress
from pathlib import Path       # For filesystem path handling

# Third-party imports
from sqlalchemy import create_engine, inspect  # For database connection and schema inspection
import pandas as pd            # For DataFrame structures and SQL table reading
from ase.db import connect as ase_connect      # For connecting to ASE database interface

# --------------------------------------
# Logging configuration: set global log level and formatting
logging.basicConfig(
    level=logging.INFO,                   # Log messages at INFO level and above
    format="%(asctime)s %(levelname)s %(message)s",  # Include timestamp, level, and message
    datefmt="%Y-%m-%d %H:%M:%S"          # Timestamp format for readability
)
logger = logging.getLogger(__name__)     # Create a logger for this module

# --------------------------------------
# Optional profiling report setup
ProfileReport = None  # Initialize placeholder for profiling class
try:
    from ydata_profiling import ProfileReport
    logger.info("Using ydata_profiling for profiling.")  # Preferred profiling package
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        logger.info("Using pandas_profiling for profiling.")  # Fallback profiling package
    except ImportError:
        # Both profiling libraries missing: warn and skip profiling steps
        logger.warning(
            "Neither ydata_profiling nor pandas_profiling is installed; profiling report will be skipped."
        )

# --------------------------------------
# Parquet engine availability check
_parquet_engine_available = False  # Flag to indicate if we can write Parquet files
try:
    import pyarrow  # noqa: F401   # Try pyarrow first
    _parquet_engine_available = True
    logger.info("pyarrow is available for parquet support.")
except ImportError:
    try:
        import fastparquet  # noqa: F401  # Try fastparquet if pyarrow not found
        _parquet_engine_available = True
        logger.info("fastparquet is available for parquet support.")
    except ImportError:
        # Neither Parquet engine installed: warn and fallback to CSV
        logger.warning(
            "pyarrow and fastparquet are not installed; parquet support will be skipped."
        )


def get_db_path() -> Path:
    """
    Locate the SQLite database file c2db.db in standard project locations.
    Searches in:
      - <project_root>/databases/c2db.db
      - <project_root>/c2db.db
      - <project_root>/backup/c2db.db
    Exits with error if not found.
    """
    # Determine project root relative to this script
    root = Path(__file__).resolve().parent.parent
    # Candidate locations for database
    candidates = [
        root / 'databases' / 'c2db.db',
        root / 'c2db.db',
        root / 'backup' / 'c2db.db'
    ]
    # Return first existing path
    for path in candidates:
        if path.exists():
            return path
    # Log error and exit if none found
    logger.error("Database file c2db.db not found in expected locations.")
    sys.exit(1)


def load_tables(engine):
    """
    Load all tables (except sqlite_sequence) into pandas DataFrames.
    Returns:
      dict of table_name -> DataFrame
    """
    inspector = inspect(engine)  # Create SQLAlchemy inspector
    # Filter out internal SQLite sequence table
    tables = [t for t in inspector.get_table_names() if t != 'sqlite_sequence']
    logger.info(f"Found tables: {tables}")  # Log discovered tables
    # Read each table into a DataFrame using pandas
    dfs = {tbl: pd.read_sql_table(tbl, engine) for tbl in tables}
    return dfs


def profile_nkv(df_nkv: pd.DataFrame, out_path: Path):
    """
    Generate an HTML profiling report for the number_key_values table.
    Skips if ProfileReport class is unavailable.
    """
    if ProfileReport is None:
        logger.info("Skipping profiling: ProfileReport unavailable.")
        return
    try:
        # Create profiling report with explorative analysis enabled
        profile = ProfileReport(df_nkv, title="Number Key Values Profile", explorative=True)
        # Save HTML output to specified path
        profile.to_file(out_path)
        logger.info(f"Profile report saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate profiling report: {e}")


def assemble_features(dfs: dict, z_to_element: dict) -> pd.DataFrame:
    """
    Combine numeric, text key-values and species composition
    into a single feature DataFrame indexed by system ID.

    Steps:
      1. Pivot number_key_values into numeric features
      2. Pivot text_key_values into text features
      3. Map atomic numbers Z to element symbols, pivot species counts
      4. Join all features with system metadata
    Returns:
      DataFrame with combined features
    """
    # Extract individual tables from dictionary
    df_sys = dfs['systems']                  # System metadata table
    df_nkv = dfs['number_key_values']       # Numeric key-values
    df_tkv = dfs['text_key_values']         # Text key-values
    df_sp  = dfs['species']                 # Species composition table

    # Pivot numeric key-values: rows=index 'id', columns=keys, values=value
    num_feats = df_nkv.pivot(index='id', columns='key', values='value')
    # Pivot text key-values similarly
    txt_feats = df_tkv.pivot(index='id', columns='key', values='value')
    # Map atomic number Z to element symbol, then pivot species counts
    comp = (
        df_sp.assign(elem=lambda d: d['Z'].map(z_to_element))
             .pivot_table(index='id', columns='elem', values='n', fill_value=0)
    )

    # Use system metadata as base (set 'id' as index)
    df_meta = df_sys.set_index('id')
    # Join numeric, text, and composition features
    df_full = df_meta.join([num_feats, txt_feats, comp], how='left')
    return df_full


def save_features(df: pd.DataFrame, out_path: Path):
    """
    Persist the assembled feature DataFrame to disk.
    If Parquet support is available, save as .parquet;
    otherwise fallback to .csv and log a warning.
    """
    try:
        if _parquet_engine_available:
            # Save to Parquet format for efficient IO and schema preservation
            df.to_parquet(out_path.with_suffix('.parquet'))
            logger.info(f"Full feature DataFrame saved to {out_path.with_suffix('.parquet')}")
        else:
            # Force fallback via exception
            raise ImportError("No parquet engine available")
    except ImportError:
        # Write out as CSV instead
        csv_path = out_path.with_suffix('.csv')
        df.to_csv(csv_path)
        logger.warning(
            f"Parquet unsupported; saved features as CSV to {csv_path}."
        )


def main():
    """
    Main pipeline execution:
      1. Locate database file
      2. Create SQLAlchemy engine
      3. Load all tables into DataFrames
      4. Profile number_key_values table (optional)
      5. Assemble combined features DataFrame
      6. Save features to disk
      7. (Example) Connect to ASE DB and retrieve atoms
    """
    # Step 1: locate the c2db.db file
    db_path = get_db_path()
    # Step 2: establish SQLite connection via SQLAlchemy
    engine = create_engine(f"sqlite:///{db_path}")
    # Step 3: load tables into a dict of DataFrames
    dfs = load_tables(engine)

    # Step 4: profiling numeric key-values
    project_root = db_path.parent.parent
    report_path = project_root / 'results' / 'nkv_profile.html'
    profile_nkv(dfs['number_key_values'], report_path)

    # Step 5: feature assembly using atomic mapping
    Z_to_element = {
        # Populate with atomic number → element symbol mapping, e.g., 1: 'H', 6: 'C'
    }
    df_full = assemble_features(dfs, Z_to_element)

    # Step 6: save features to Parquet or CSV
    features_path = project_root / 'results' / 'materials_features'
    save_features(df_full, features_path)

    # Step 7: ASE database example (retrieve Atoms object for system_id=1)
    ase_db = ase_connect(db_path)
    # atoms = ase_db.get_atoms(system_id=1)  # Uncomment and provide valid ID

# Standard Python entry point check
if __name__ == '__main__':
    main()

