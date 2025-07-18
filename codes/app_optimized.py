
#!/usr/bin/env python3
"""
app_optimized.py

End-to-end materials recommender using:
 1) GaussianRandomProjection to reduce ~180k features → 256 dims in seconds
 2) FAISS IVF-PQ approximate index to keep memory footprint low
 3) Ollama local LLM for concise, human-readable summaries

Usage:
    python app_optimized.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

import sys
import time
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import faiss
from ollama import chat

# ------------------------------------------------------------------------------
# Helper: timestamped logging
# ------------------------------------------------------------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ------------------------------------------------------------------------------
# 1) Paths & Model Loading
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RES  = ROOT.parent / 'results'

log("1) Loading trained composition model …")
model_data = joblib.load(RES / 'composition_model.joblib')
model      = model_data['model']
features   = model_data['features']
elements   = model_data['elements']
log(f"   • Model loaded: {len(features)} features → {len(elements)} elements\n")

# ------------------------------------------------------------------------------
# 2) Robust feature-table loading (pyarrow → fastparquet → CSV)
# ------------------------------------------------------------------------------

def load_feature_table():
    fp = RES / 'features_clean.parquet'
    if fp.exists():
        for engine in ('pyarrow', 'fastparquet'):
            log(f"2) Trying features_clean.parquet with engine='{engine}' …")
            try:
                df = pd.read_parquet(fp, engine=engine)
                log(f"   ✔ Loaded features_clean.parquet via {engine}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"   ✖ {engine} failed: {e}")
    # fallback to CSV
    csv = RES / 'materials_features.csv'
    if csv.exists():
        log("2) Falling back to materials_features.csv …")
        df = pd.read_csv(csv, index_col=0)
        log(f"   ✔ Loaded CSV features, shape={df.shape}\n")
        return df

    sys.exit("ERROR: No feature table found under results/")

log("2) Loading feature table …")
df_feat = load_feature_table()

# ------------------------------------------------------------------------------
# 3) Load structural descriptors
# ------------------------------------------------------------------------------

log("3) Loading structural descriptors …")
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
if 'volume' in df_struc.columns:
    df_struc = df_struc.rename(columns={'volume':'volume_cell'})
    log("   • Renamed column 'volume' → 'volume_cell'")
log(f"   ✔ Structural descriptors loaded, shape={df_struc.shape}\n")

# ------------------------------------------------------------------------------
# 4) Merge feature & structure tables
# ------------------------------------------------------------------------------

log("4) Merging feature and structural DataFrames …")
df_all = df_feat.join(df_struc, how='inner').fillna(0)
log(f"   ✔ Merged DataFrame shape: {df_all.shape}\n")

# ------------------------------------------------------------------------------
# 5) Dimensionality reduction via Random Projection
# ------------------------------------------------------------------------------

TARGET_DIMS = 256

log(f"5) Reducing dimension from {len(features)} → {TARGET_DIMS} via GaussianRandomProjection …")
start = time.time()
grp = GaussianRandomProjection(n_components=TARGET_DIMS, random_state=42)
X_full    = df_all[features].values.astype('float32')
X_reduced = grp.fit_transform(X_full)
log(f"   ✔ Projection completed in {time.time() - start:.1f}s; reduced shape = {X_reduced.shape}\n")

# ------------------------------------------------------------------------------
# 6) Build FAISS IVF-PQ approximate index
# ------------------------------------------------------------------------------

# IVF-PQ config
nlist = 256   # number of Voronoi cells
m     = 16    # number of subquantizers
nbits = 8     # bits per subvector

log(f"6) Building FAISS IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits}) …")
quantizer = faiss.IndexFlatL2(TARGET_DIMS)
index     = faiss.IndexIVFPQ(quantizer, TARGET_DIMS, nlist, m, nbits)
log("   • Training IVF-PQ index …")
index.train(X_reduced)
log("   • Adding vectors to the index …")
index.add(X_reduced)
ids       = df_all.index.to_numpy()
log(f"   ✔ Index built; total vectors = {index.ntotal}\n")

# ------------------------------------------------------------------------------
# 7) Prediction & nearest-neighbor retrieval
# ------------------------------------------------------------------------------

def predict_composition(props: dict) -> dict:
    """Predict element fractions from input property dict."""
    x    = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    frac = model.predict(x)[0]
    return dict(zip(elements, frac))

def retrieve_neighbors(props: dict, k: int=5):
    """Retrieve the top-k nearest system IDs and their L2 distances."""
    x_full   = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    x_red    = grp.transform(x_full)
    D, I     = index.search(x_red, k)
    return ids[I[0]], D[0]

# ------------------------------------------------------------------------------
# 8) Ollama-based natural language summarization
# ------------------------------------------------------------------------------

def summarize(props, comp, neigh_ids, dists) -> str:
    """Generate a concise LLM-based recommendation via Ollama."""
    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "The model predicts these element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"The top-5 nearest known materials are system IDs {neigh_ids.tolist()} "
        f"with distances {dists.tolist()}.\n"
        "Provide a concise recommendation explaining these results."
    )
    resp = chat(model='qwen2.5:7b',
                messages=[{'role':'user','content':prompt}],
                stream=False)
    return resp.message.content

# ------------------------------------------------------------------------------
# 9) Command-line interface
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    # Parse --feature value pairs
    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0, len(args), 2)}
    log(f"INPUT properties: {props}\n")

    # Predict and retrieve
    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props)

    # Display predictions
    log("Predicted composition fractions:")
    for el, val in comp.items():
        print(f"  • {el:>2}: {val:.3f}")

    log("Nearest neighbors (ID : distance):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    # Summarize via Ollama
    log("Generating recommendation via Ollama …")
    summary = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", summary)

