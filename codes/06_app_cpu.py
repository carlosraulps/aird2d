
#!/usr/bin/env python3
# app_optimized_commented.py

"""
End-to-end materials recommender:
 1) Dimensionality reduction via GaussianRandomProjection
 2) Approximate nearest-neighbor search with FAISS IVF-PQ
 3) Human-readable summaries via local Ollama LLM

Usage example:
    python app_optimized.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

# --------------------------------------
# Standard library imports
import sys               # Handle CLI args and exits
import time              # Timestamping for logs
import json              # Formatting input/output dictionaries
from pathlib import Path # File path manipulations

# --------------------------------------
# Third-party imports
import joblib            # Load serialized model artifacts
import pandas as pd      # DataFrame handling and I/O
import numpy as np       # Numerical operations on arrays
from sklearn.random_projection import GaussianRandomProjection  # Fast projection
import faiss             # High-performance similarity search
from ollama import chat  # Interface to local Ollama LLM for summaries

# ----------------------------------------------------------------------------
# Helper: timestamped logging
# ----------------------------------------------------------------------------
def log(msg: str):
    """Print a message prefixed with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ----------------------------------------------------------------------------
# 1) Paths & Model Loading
# ----------------------------------------------------------------------------

# Determine root directory (where this script lives)
ROOT = Path(__file__).resolve().parent
# Results dir contains precomputed features, model, etc.
RES  = ROOT.parent / 'results'

# Load trained composition model and metadata
log("1) Loading trained composition model …")
model_data = joblib.load(RES / 'composition_model.joblib')
model      = model_data['model']      # MultiOutputRegressor wrapper
features   = model_data['features']   # List of feature column names
elements   = model_data['elements']   # List of element target names
log(f"   • Model loaded: {len(features)} features → {len(elements)} elements\n")

# ----------------------------------------------------------------------------
# 2) Robust feature-table loading (Parquet → CSV fallback)
# ----------------------------------------------------------------------------

def load_feature_table() -> pd.DataFrame:
    """
    Try loading cleaned features in Parquet format with pyarrow or fastparquet.
    If that fails or file missing, fall back to CSV.
    Exits if no file found.
    """
    fp = RES / 'features_clean.parquet'
    if fp.exists():
        # Attempt both common engines
        for engine in ('pyarrow', 'fastparquet'):
            log(f"2) Trying features_clean.parquet with engine='{engine}' …")
            try:
                df = pd.read_parquet(fp, engine=engine)
                log(f"   ✔ Loaded features_clean.parquet via {engine}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"   ✖ {engine} failed: {e}")
    # CSV fallback if Parquet loads not available
    csv = RES / 'materials_features.csv'
    if csv.exists():
        log("2) Falling back to materials_features.csv …")
        df = pd.read_csv(csv, index_col=0)
        log(f"   ✔ Loaded CSV features, shape={df.shape}\n")
        return df
    # No file found: exit with error
    sys.exit("ERROR: No feature table found under results/")

log("2) Loading feature table …")
df_feat = load_feature_table()

# ----------------------------------------------------------------------------
# 3) Load structural descriptors and rename conflicts
# ----------------------------------------------------------------------------

log("3) Loading structural descriptors …")
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
# Rename 'volume' to avoid collision with feature column of same name
df_struc = df_struc.rename(columns={'volume':'volume_cell'})
log(f"   ✔ Structural descriptors loaded, shape={df_struc.shape}\n")

# ----------------------------------------------------------------------------
# 4) Merge feature & structure tables
# ----------------------------------------------------------------------------

log("4) Merging feature and structural DataFrames …")
# Inner join retains only entries present in both dataframes
# Fill missing with zero to allow numeric processing
df_all = df_feat.join(df_struc, how='inner').fillna(0)
log(f"   ✔ Merged DataFrame shape: {df_all.shape}\n")

# ----------------------------------------------------------------------------
# 5) Dimensionality reduction via Random Projection
# ----------------------------------------------------------------------------

# Target dimensionality for projection
target_dims = 256
log(f"5) Reducing dimension from {len(features)} → {target_dims} via GaussianRandomProjection …")
start = time.time()
# Instantiate and apply projection to feature vectors
grp = GaussianRandomProjection(n_components=target_dims, random_state=42)
# Extract raw feature matrix as float32 numpy array
X_full    = df_all[features].values.astype('float32')
X_reduced = grp.fit_transform(X_full)
log(f"   ✔ Projection completed in {time.time() - start:.1f}s; reduced shape = {X_reduced.shape}\n")

# ----------------------------------------------------------------------------
# 6) Build FAISS IVF-PQ approximate index
# ----------------------------------------------------------------------------

# IVF-PQ hyperparameters
nlist = 256  # number of Voronoi cells/clusters
m     = 16   # subquantizers per vector
nbits = 8    # bits per subvector (256 centroids each)

log(f"6) Building FAISS IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits}) …")
# Use a flat L2 index as coarse quantizer\quantizer = faiss.IndexFlatL2(target_dims)
# Create IndexIVFPQ with specified parameters
index = faiss.IndexIVFPQ(quantizer, target_dims, nlist, m, nbits)

log("   • Training IVF-PQ index …")
index.train(X_reduced)  # fit codebooks
log("   • Adding vectors to the index …")
index.add(X_reduced)    # add all reduced vectors
# Store system IDs aligned with index positions
ids = df_all.index.to_numpy()
log(f"   ✔ Index built; total vectors = {index.ntotal}\n")

# ----------------------------------------------------------------------------
# 7) Prediction & nearest-neighbor retrieval
# ----------------------------------------------------------------------------

def predict_composition(props: dict) -> dict:
    """Predict element fraction composition given property inputs."""
    # Build 1×D feature vector in correct order\    x = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    # Predict fractions with trained model
    frac = model.predict(x)[0]
    return dict(zip(elements, frac))


def retrieve_neighbors(props: dict, k: int = 5):
    """Retrieve top-k nearest system IDs and distances for given props."""
    x_full = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    # Project into reduced space
    x_red  = grp.transform(x_full)
    # Search in FAISS index
    D, I    = index.search(x_red, k)
    return ids[I[0]], D[0]

# ----------------------------------------------------------------------------
# 8) Ollama-based natural language summarization
# ----------------------------------------------------------------------------

def summarize(props: dict, comp: dict, neigh_ids: np.ndarray, dists: np.ndarray) -> str:
    """Generate a concise recommendation using local Ollama LLM."""
    # Construct prompt including input props, predicted composition,
    # and nearest neighbor IDs/distances\    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "The model predicts these element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"The top-5 nearest known materials are system IDs {neigh_ids.tolist()} "
        f"with distances {dists.tolist()}.\n"
        "Provide a concise recommendation explaining these results."
    )
    # Call Ollama chat API to generate text
    resp = chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=False
    )
    return resp.message.content

# ----------------------------------------------------------------------------
# 9) Command-line interface
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    # Expect even number of args: --feature value pairs
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    # Parse CLI into dict of property: float value
    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0, len(args), 2)}
    log(f"INPUT properties: {props}\n")

    # Run prediction and neighbor retrieval
    comp, neighs, dists = predict_composition(props), *retrieve_neighbors(props)

    # Display predicted composition fractions
    log("Predicted composition fractions:")
    for el, val in comp.items():
        print(f"  • {el:>2}: {val:.3f}")

    # Display nearest neighbor IDs and distances
    log("Nearest neighbors (ID : distance):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    # Generate and print summary
    log("Generating recommendation via Ollama …")
    summary = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", summary)

