
#!/usr/bin/env python3
"""
connect_load_db.py

Project structure:
.
├── codes/
│   ├── connect_load_db.py  <- this script
│   ├── test_db.py
│   ├── train_discriminative.py
│   ├── train_generative.py
│   └── features/
├── databases/
│   └── c2db.db
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
"""
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, inspect
import pandas as pd
from ase.db import connect as ase_connect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Attempt to import ProfileReport from known packages
ProfileReport = None
try:
    from ydata_profiling import ProfileReport
    logger.info("Using ydata_profiling for profiling.")
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        logger.info("Using pandas_profiling for profiling.")
    except ImportError:
        logger.warning(
            "Neither ydata_profiling nor pandas_profiling is installed; profiling report will be skipped."
        )

# Attempt to check parquet support
_parquet_engine_available = False
try:
    # try importing pyarrow or fastparquet
    import pyarrow  # noqa: F401
    _parquet_engine_available = True
    logger.info("pyarrow is available for parquet support.")
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _parquet_engine_available = True
        logger.info("fastparquet is available for parquet support.")
    except ImportError:
        logger.warning(
            "pyarrow and fastparquet are not installed; parquet support will be skipped."
        )


def get_db_path() -> Path:
    """Locate c2db.db in project directories."""
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / 'databases' / 'c2db.db',
        root / 'c2db.db',
        root / 'backup' / 'c2db.db'
    ]
    for path in candidates:
        if path.exists():
            return path
    logger.error("Database file c2db.db not found in expected locations.")
    sys.exit(1)


def load_tables(engine):
    """Return a dict of DataFrames for each table in the DB (excluding sqlite_sequence)."""
    inspector = inspect(engine)
    tables = [t for t in inspector.get_table_names() if t != 'sqlite_sequence']
    logger.info(f"Found tables: {tables}")
    dfs = {tbl: pd.read_sql_table(tbl, engine) for tbl in tables}
    return dfs


def profile_nkv(df_nkv: pd.DataFrame, out_path: Path):
    """Generate an HTML profiling report for number_key_values if ProfileReport is available."""
    if ProfileReport is None:
        logger.info("Skipping profiling: ProfileReport unavailable.")
        return
    try:
        profile = ProfileReport(df_nkv, title="Number Key Values Profile", explorative=True)
        profile.to_file(out_path)
        logger.info(f"Profile report saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate profiling report: {e}")


def assemble_features(dfs: dict, z_to_element: dict) -> pd.DataFrame:
    """Pivot numeric and text key-values and species composition into a single feature DataFrame."""
    df_sys = dfs['systems']
    df_nkv = dfs['number_key_values']
    df_tkv = dfs['text_key_values']
    df_sp  = dfs['species']

    num_feats = df_nkv.pivot(index='id', columns='key', values='value')
    txt_feats = df_tkv.pivot(index='id', columns='key', values='value')
    comp = (
        df_sp.assign(elem=lambda d: d['Z'].map(z_to_element))
             .pivot_table(index='id', columns='elem', values='n', fill_value=0)
    )

    df_meta = df_sys.set_index('id')
    df_full = df_meta.join([num_feats, txt_feats, comp], how='left')
    return df_full


def save_features(df: pd.DataFrame, out_path: Path):
    """Save feature DataFrame to parquet or fallback to csv if parquet unsupported."""
    try:
        if _parquet_engine_available:
            df.to_parquet(out_path.with_suffix('.parquet'))
            logger.info(f"Full feature DataFrame saved to {out_path.with_suffix('.parquet')}")
        else:
            raise ImportError("No parquet engine available")
    except ImportError:
        csv_path = out_path.with_suffix('.csv')
        df.to_csv(csv_path)
        logger.warning(
            f"Parquet unsupported; saved features as CSV to {csv_path}."
        )


def main():
    db_path = get_db_path()
    engine = create_engine(f"sqlite:///{db_path}")
    dfs = load_tables(engine)

    # Profile numeric key-values
    project_root = db_path.parent.parent
    out_report = project_root / 'results' / 'nkv_profile.html'
    profile_nkv(dfs['number_key_values'], out_report)

    # Assemble features and save
    # Populate Z_to_element with atomic number → symbol mapping
    Z_to_element = {
        # e.g. 1: 'H', 6: 'C', ...
    }
    df_full = assemble_features(dfs, Z_to_element)
    out_features = project_root / 'results' / 'materials_features'
    save_features(df_full, out_features)

    # ASE example usage:
    ase_db = ase_connect(db_path)
    # atoms = ase_db.get_atoms(system_id=1)  # Uncomment and set a valid ID

if __name__ == '__main__':
    main()

