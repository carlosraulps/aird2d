
#!/usr/bin/env python3
"""
app.py

End-to-end materials recommender using FAISS + Ollama local LLM.

Usage:
    python app.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

import os
import sys
import json
import time
from pathlib import Path

import joblib                                     # for loading the trained model :contentReference[oaicite:6]{index=6}
import pandas as pd                               # for DataFrame operations
import numpy as np                                # for numeric arrays
import faiss                                      # for nearest-neighbor search :contentReference[oaicite:7]{index=7}
from ollama import chat                           # for local LLM inference :contentReference[oaicite:8]{index=8}

# Helper: timestamped logging
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# Paths
ROOT = Path(__file__).resolve().parent
RES  = ROOT.parent / 'results'

# ——————————————————————————————
# 1. Load composition model + metadata
# ——————————————————————————————
log("Loading composition model…")
model_data = joblib.load(RES / 'composition_model.joblib')
model      = model_data['model']
features   = model_data['features']
elements   = model_data['elements']
log(f"Model loaded: {len(features)} features → {len(elements)} elements\n")

# ——————————————————————————————
# 2. Robust feature-table loading
# ——————————————————————————————
def load_feature_table():
    """Attempt to load the feature table via pyarrow → fastparquet → CSV."""
    fp = RES / 'features_clean.parquet'
    if fp.exists():
        for engine in ('pyarrow', 'fastparquet'):
            log(f"Trying features_clean.parquet with engine={engine}…")
            try:
                df = pd.read_parquet(fp, engine=engine)
                log(f"✔ Loaded via {engine}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"  ✖ {engine} failed: {e}")

    # Fallback to features_full.parquet
    fp_full = RES / 'features_full.parquet'
    if fp_full.exists():
        for engine in ('pyarrow', 'fastparquet'):
            log(f"Trying features_full.parquet with engine={engine}…")
            try:
                df = pd.read_parquet(fp_full, engine=engine)
                log(f"✔ Loaded features_full via {engine}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"  ✖ {engine} failed: {e}")

    # Final fallback to CSV
    csv = RES / 'materials_features.csv'
    if csv.exists():
        log("Falling back to materials_features.csv…")
        df = pd.read_csv(csv, index_col=0)
        log(f"✔ Loaded CSV, shape={df.shape}\n")
        return df

    sys.exit("ERROR: No feature table found in results/")

log("Loading feature table…")
df_feat = load_feature_table()

# ——————————————————————————————
# 3. Load structural descriptors
# ——————————————————————————————
log("Loading structural descriptors…")
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
if 'volume' in df_struc.columns:
    df_struc = df_struc.rename(columns={'volume':'volume_cell'})
    log("Renamed 'volume' → 'volume_cell'")
log(f"Structural descriptors shape: {df_struc.shape}\n")

# ——————————————————————————————
# 4. Merge everything
# ——————————————————————————————
log("Merging feature + structural DataFrames…")
df_all = df_feat.join(df_struc, how='inner').fillna(0)
log(f"Merged DataFrame shape: {df_all.shape}\n")

# ——————————————————————————————
# 5. Build FAISS index
# ——————————————————————————————
log("Building FAISS index…")
X    = df_all[features].values.astype('float32')       # feature matrix :contentReference[oaicite:9]{index=9}
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
ids   = df_all.index.to_numpy()
log(f"FAISS index ready with {index.ntotal} vectors\n")

# ——————————————————————————————
# 6. Prediction & retrieval
# ——————————————————————————————
def predict_composition(props: dict) -> dict:
    """Predict element fractions from property dict."""
    x    = np.array([[props.get(f,0) for f in features]], dtype='float32')
    frac = model.predict(x)[0]
    return dict(zip(elements, frac))

def retrieve_neighbors(props: dict, k: int=5):
    """Retrieve top-k nearest system IDs & distances."""
    x, = np.array([[props.get(f,0) for f in features]], dtype='float32'),
    D, I = index.search(x, k)
    return ids[I[0]], D[0]

# ——————————————————————————————
# 7. Ollama-based summarization
# ——————————————————————————————
def summarize(props, comp, neigh_ids, dists) -> str:
    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "The model predicts these element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"Top-5 nearest known materials: IDs {neigh_ids.tolist()} with distances {dists.tolist()}.\n"
        "Provide a concise recommendation explaining these results."
    )
    resp = chat(model='qwen2.5:7b', messages=[{'role':'user','content':prompt}], stream=False)
    return resp.message.content

# ——————————————————————————————
# 8. CLI parsing & execution
# ——————————————————————————————
if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv)%2!=1:
        print(__doc__)
        sys.exit(1)

    # Parse --feature value args
    props = {sys.argv[i].lstrip('-'): float(sys.argv[i+1])
             for i in range(1, len(sys.argv), 2)}
    log(f"Input properties: {props}\n")

    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props)

    log("Predicted composition fractions:")
    for el,v in comp.items():
        print(f"  • {el:>2}: {v:.3f}")

    log("Nearest neighbors (ID : distance):")
    for sid,dist in zip(neighs,dists):
        print(f"  • {sid}: {dist:.3f}")

    log("Generating Ollama summary…")
    summary = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", summary)

