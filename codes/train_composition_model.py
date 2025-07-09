
#!/usr/bin/env python3
# train_composition_model.py

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 1) Ensure scikit-learn is installed
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
except ImportError as e:
    sys.exit(
        "ERROR: scikit-learn is not installed. "
        "Please run `pip install scikit-learn` and try again."
    )
# scikit-learn is packaged as scikit-learn, not sklearn :contentReference[oaicite:1]{index=1}

import joblib

# Base paths
root = Path(__file__).resolve().parent.parent
res  = root / 'results'

# --- 2) Robust feature loading with Parquet engine fallback ---
def load_features():
    # Try cleaned Parquet with PyArrow (may hit Thrift limit) :contentReference[oaicite:2]{index=2}
    for engine in ['pyarrow', 'fastparquet']:
        fp = res / 'features_clean.parquet'
        if fp.exists():
            try:
                return pd.read_parquet(fp, engine=engine)
            except Exception:
                print(f"❗ features_clean.parquet failed with {engine}, trying next…")
    # Try full Parquet
    for engine in ['pyarrow', 'fastparquet']:
        fp = res / 'features_full.parquet'
        if fp.exists():
            try:
                return pd.read_parquet(fp, engine=engine)
            except Exception:
                print(f"❗ features_full.parquet failed with {engine}, trying next…")
    # Fallback to CSV
    csv = res / 'materials_features.csv'
    if csv.exists():
        print("⚠️ Falling back to CSV load")
        return pd.read_csv(csv, index_col=0)
    sys.exit("ERROR: No feature file found for training.")

df_feat = load_features()
print(f"Loaded feature table with shape {df_feat.shape}")

# --- 3) Load structural descriptors & rename overlaps ---
try:
    df_struc = pd.read_parquet(res / 'structural_descriptors.parquet')
    print("Loaded structural_descriptors.parquet")
except Exception as e:
    sys.exit(f"ERROR: cannot load structural_descriptors.parquet ({e})")
# Rename 'volume' to avoid collision 
df_struc = df_struc.rename(columns={'volume': 'volume_cell'})

# --- 4) Merge tables ---
df = df_feat.join(df_struc, how='inner')
print(f"Merged dataset shape: {df.shape}")

# --- 5) Prepare X (features) and y (element fractions) ---
# Identify element columns (assuming one-letter or two-letter symbols) :contentReference[oaicite:4]{index=4}
all_cols = set(df.columns)
meta_and_struct = set(df_struc.columns) | set(['id'])  # structural + metadata origin
elem_cols = [c for c in df.columns if c not in meta_and_struct and c.isalpha()]
X = df.drop(columns=elem_cols)
counts = df[elem_cols].fillna(0)
y = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

print(f"Training to predict {len(elem_cols)} element fractions")

# --- 6) Train/Test Split & Model Training ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model = MultiOutputRegressor(base)
print("Training RandomForest multi-output regressor…")
model.fit(X_train, y_train)

# --- 7) Evaluation ---
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred, multioutput='uniform_average')
mae  = mean_absolute_error(y_test, y_pred)
print(f"\nOverall R²:  {r2:.3f}")
print(f"Overall MAE: {mae:.3f}")

# Per-element R²
r2_per = {
    elem: r2_score(y_test[elem], y_pred[:, i])
    for i, elem in enumerate(elem_cols)
}
print("\nTop 5 elements by R²:")
for elem, score in sorted(r2_per.items(), key=lambda kv: kv[1], reverse=True)[:5]:
    print(f"  {elem:>4} : {score:.3f}")

# --- 8) Save the model ---
out_model = res / 'composition_model.joblib'
joblib.dump({
    'model': model,
    'features': X.columns.tolist(),
    'elements': elem_cols
}, out_model)
print(f"\nSaved trained model to {out_model}")

