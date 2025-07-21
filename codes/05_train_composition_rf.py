
#!/usr/bin/env python3
# train_composition_model_commented.py

# --------------------------------------
# Script to train a RandomForest-based model
# predicting elemental composition fractions
# from material feature matrices.
#
# Steps:
# 1) Ensure scikit-learn is installed
# 2) Configure paths, hyperparameters, and logging
# 3) Load feature tables with engine fallbacks
# 4) Load and prepare structural descriptors
# 5) Merge feature + descriptor tables
# 6) Prepare training/testing splits
# 7) Configure and train MultiOutputRegressor
# 8) Evaluate model performance
# 9) Save trained model and metadata
# --------------------------------------

# --------------------------------------
# Standard library imports
import sys              # For CLI args and exit handling
import time             # For timing and timestamps
from pathlib import Path  # For filesystem path handling

# --------------------------------------
# Third-party imports
import pandas as pd     # For DataFrame operations and I/O
import numpy as np      # For numeric operations and array handling
import joblib           # For saving Python objects (trained model)

# Attempt to import scikit-learn, exit if not available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
except ImportError:
    sys.exit(
        "ERROR: scikit-learn is not installed. "
        "Please run `pip install scikit-learn` and try again."
    )

# --------------------------------------
# Configuration & default hyperparameters
root      = Path(__file__).resolve().parent.parent  # Project root directory
res       = root / 'results'                        # Directory for feature files, results
# Helper to build Parquet file paths by name
def parquet_p(name: str) -> Path:
    return res / f"{name}.parquet"

# Tunable hyperparameters, chosen for speed
DEFAULT_N_ESTIMATORS = 20   # Number of trees in the forest
DEFAULT_MAX_DEPTH    = 8    # Maximum depth of each tree
DEFAULT_MAX_SAMPLES  = 0.5  # Fraction of samples used for each tree

# Logging helper: print timestamped messages
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --------------------------------------
# 3) Load feature tables with fallback order
log("STARTING feature loading")
start = time.time()  # Record start time for performance logging

def load_features() -> pd.DataFrame:
    """
    Attempt to load 'features_clean.parquet' using pyarrow or fastparquet;
    if that fails, load 'features_full.parquet';
    if still not found, fall back to CSV.
    Exits if no file found.
    """
    # Try cleaned Parquet with available engines
    for engine in ['pyarrow', 'fastparquet']:
        path = parquet_p('features_clean')
        if path.exists():
            log(f"Attempting to load features_clean.parquet with engine={engine}")
            try:
                df = pd.read_parquet(path, engine=engine)
                log(f"Loaded features_clean.parquet: shape={df.shape}")
                return df
            except Exception as e:
                log(f"  ❗ Failed ({engine}): {e}")

    # Fallback to full Parquet\    
    for engine in ['pyarrow', 'fastparquet']:
        path = parquet_p('features_full')
        if path.exists():
            log(f"Attempting to load features_full.parquet with engine={engine}")
            try:
                df = pd.read_parquet(path, engine=engine)
                log(f"Loaded features_full.parquet: shape={df.shape}")
                return df
            except Exception as e:
                log(f"  ❗ Failed ({engine}): {e}")

    # Final fallback to CSV
    csv = res / 'materials_features.csv'
    if csv.exists():
        log("Falling back to materials_features.csv")
        df = pd.read_csv(csv, index_col=0)
        log(f"Loaded materials_features.csv: shape={df.shape}")
        return df

    # If none of the above succeeded, exit with error
    sys.exit("ERROR: No feature file found in results/")

# Execute feature loading
df_feat = load_features()
log(f"FEATURE loading completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 4) Load structural descriptors and rename collisions
log("STARTING structural descriptor loading")
start = time.time()

struct_path = parquet_p('structural_descriptors')
if not struct_path.exists():
    sys.exit(f"ERROR: {struct_path} not found.")
# Read structural descriptors Parquet
df_struc = pd.read_parquet(struct_path)
log(f"Loaded structural_descriptors.parquet: shape={df_struc.shape}")

# Rename 'volume' to 'volume_cell' to avoid column name conflict\if 'volume' in df_struc.columns:
    df_struc = df_struc.rename(columns={'volume':'volume_cell'})
    log("Renamed 'volume' → 'volume_cell' in structural descriptors")

log(f"STRUCTURAL loading completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 5) Merge feature and structural tables
log("MERGING feature + structural tables")
start = time.time()

# Use inner join to retain only IDs present in both dataframes, fill any NaNs with 0
df = df_feat.join(df_struc, how='inner').fillna(0)
log(f"Merged dataset shape: {df.shape}")

# Identify feature columns (X_cols) vs element target columns (elem_cols)
# Assumes element names are alphabetic strings\meta_cols = set(df_struc.columns)
X_cols    = [c for c in df.columns if c not in meta_cols and not c.isalpha()]
elem_cols = [c for c in df.columns if c not in X_cols]
log(f"Detected {len(X_cols)} feature columns and {len(elem_cols)} element target columns")

log(f"MERGE completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 6) Prepare training & test data splits
log("PREPARING training & test splits")
start = time.time()

# X matrix: predictor features
df_X = df[X_cols]
# Counts: raw element counts from structural descriptors
df_counts = df[elem_cols].astype(float).fillna(0)
# Convert counts to fractional composition per sample
# Divide each row by its sum; handle zero totals to avoid NaNs
y = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

log(f"Feature matrix X shape: {df_X.shape}")
log(f"Target matrix y shape:   {y.shape}")

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df_X, y, test_size=0.2, random_state=42
)
log(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

log(f"SPLIT completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 7) Configure and train the model
log("CONFIGURING RandomForestRegressor")
start = time.time()

# Allow override of n_estimators via CLI argument
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_ESTIMATORS
max_depth    = DEFAULT_MAX_DEPTH
max_samples  = DEFAULT_MAX_SAMPLES

log(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, max_samples={max_samples}")

# Create base regressor and wrap for multi-output
base = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    max_samples=max_samples,
    random_state=42,
    n_jobs=-1             # Use all cores for parallel training
)
model = MultiOutputRegressor(base)

log("STARTING model.fit()")
model.fit(X_train, y_train)
log(f"TRAINING completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 8) Evaluate the trained model
log("EVALUATING model on test set")
start = time.time()

y_pred = model.predict(X_test)
# Compute overall R^2 and MAE metrics
total_r2  = r2_score(y_test, y_pred, multioutput='uniform_average')
total_mae = mean_absolute_error(y_test, y_pred)
log(f"Overall R²:  {total_r2:.3f}")
log(f"Overall MAE: {total_mae:.3f}")

# Compute per-element R^2 and report top 5
r2_per = {
    elem: r2_score(y_test[elem], y_pred[:, idx])
    for idx, elem in enumerate(elem_cols)
}
top5 = sorted(r2_per.items(), key=lambda kv: kv[1], reverse=True)[:5]
log("Top 5 elements by R²:")
for el, score in top5:
    print(f"  • {el:>4} → R² = {score:.3f}")

log(f"EVALUATION completed in {time.time() - start:.1f}s\n")

# --------------------------------------
# 9) Save the trained model and metadata
log("SAVING trained model to composition_model.joblib")
out_model = res / 'composition_model.joblib'
# Store model object alongside feature/target column names
joblib.dump({
    'model': model,
    'features': X_cols,
    'elements': elem_cols
}, out_model)
log("Model successfully saved.")

