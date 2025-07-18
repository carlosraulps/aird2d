
#!/usr/bin/env python3
# train_composition_model.py

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# 1) Ensure scikit-learn is installed
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

# --- Configuration & Paths ---
root      = Path(__file__).resolve().parent.parent
res       = root / 'results'
parquet_p = lambda name: res / f"{name}.parquet"

# Default hyperparameters -- tuned for speed
DEFAULT_N_ESTIMATORS = 20
DEFAULT_MAX_DEPTH    = 8
DEFAULT_MAX_SAMPLES  = 0.5

def log(msg: str):
    """Print a timestamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --- 2) Load feature tables ---
log("STARTING feature loading")
start = time.time()

def load_features():
    # Try cleaned Parquet first
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

    # Fallback to full Parquet
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

    # Final fallback: CSV
    csv = res / 'materials_features.csv'
    if csv.exists():
        log("Falling back to materials_features.csv")
        df = pd.read_csv(csv, index_col=0)
        log(f"Loaded materials_features.csv: shape={df.shape}")
        return df

    sys.exit("ERROR: No feature file found in results/")

df_feat = load_features()
log(f"FEATURE loading completed in {time.time() - start:.1f}s\n")

# --- 3) Load structural descriptors ---
log("STARTING structural descriptor loading")
start = time.time()

struct_path = parquet_p('structural_descriptors')
if not struct_path.exists():
    sys.exit(f"ERROR: {struct_path} not found.")
df_struc = pd.read_parquet(struct_path)
log(f"Loaded structural_descriptors.parquet: shape={df_struc.shape}")

# Rename overlapping columns
if 'volume' in df_struc.columns:
    df_struc = df_struc.rename(columns={'volume':'volume_cell'})
    log("Renamed 'volume' → 'volume_cell' in structural descriptors")

log(f"STRUCTURAL loading completed in {time.time() - start:.1f}s\n")

# --- 4) Merge & inspect ---
log("MERGING feature + structural tables")
start = time.time()

df = df_feat.join(df_struc, how='inner').fillna(0)
log(f"Merged dataset shape: {df.shape}")

# Identify features vs. element targets
meta_cols = set(df_struc.columns)
X_cols    = [c for c in df.columns if c not in meta_cols and not c.isalpha()]
elem_cols = [c for c in df.columns if c not in X_cols]
log(f"Detected {len(X_cols)} feature columns and {len(elem_cols)} element target columns")

log(f"MERGE completed in {time.time() - start:.1f}s\n")

# --- 5) Prepare training data ---
log("PREPARING training & test splits")
start = time.time()

X = df[X_cols]
counts = df[elem_cols].astype(float).fillna(0)
y = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

log(f"Feature matrix X shape: {X.shape}")
log(f"Target matrix y shape:   {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
log(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

log(f"SPLIT completed in {time.time() - start:.1f}s\n")

# --- 6) Model configuration & training ---
log("CONFIGURING RandomForestRegressor")
start = time.time()

# You can override these via environment variables or CLI args if desired
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_ESTIMATORS
max_depth    = DEFAULT_MAX_DEPTH
max_samples  = DEFAULT_MAX_SAMPLES

log(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, max_samples={max_samples}")

base = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    max_samples=max_samples,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputRegressor(base)

log("STARTING model.fit()")
model.fit(X_train, y_train)
log(f"TRAINING completed in {time.time() - start:.1f}s\n")

# --- 7) Evaluation ---
log("EVALUATING model on test set")
start = time.time()

y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred, multioutput='uniform_average')
mae  = mean_absolute_error(y_test, y_pred)
log(f"Overall R²:  {r2:.3f}")
log(f"Overall MAE: {mae:.3f}")

# Top 5 per-element R²
r2_per = {
    elem: r2_score(y_test[elem], y_pred[:, i])
    for i, elem in enumerate(elem_cols)
}
top5 = sorted(r2_per.items(), key=lambda kv: kv[1], reverse=True)[:5]
log("Top 5 elements by R²:")
for el, score in top5:
    print(f"  • {el:>4} → R² = {score:.3f}")

log(f"EVALUATION completed in {time.time() - start:.1f}s\n")

# --- 8) Save the model ---
log("SAVING trained model to composition_model.joblib")
out_model = res / 'composition_model.joblib'
joblib.dump({
    'model': model,
    'features': X_cols,
    'elements': elem_cols
}, out_model)
log("Model successfully saved.\n")

