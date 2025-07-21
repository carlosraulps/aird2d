
#!/usr/bin/env python3
# app_gpu_optimized_commented.py

"""
GPU-accelerated materials recommender:
 1) cuML PCA for fast, in-GPU dimensionality reduction
 2) FAISS IVF-PQ index built on CPU then moved to GPU
 3) Ollama local LLM for summaries

Usage:
    python app_gpu_optimized.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

# --------------------------------------
# Standard library imports
# --------------------------------------
import sys               # Handle CLI args and graceful exits
import time              # Timestamping for logs
import json              # Serialize prompts for LLM
from pathlib import Path # File path manipulations

# --------------------------------------
# Third-party imports
# --------------------------------------
import joblib            # Load saved model and metadata
import pandas as pd      # DataFrame I/O and manipulation
import numpy as np       # Numerical array operations
import faiss             # High-performance vector search library
import cudf              # RAPIDS GPU DataFrames for in-GPU operations
from cuml.decomposition import PCA as cuPCA  # GPU-accelerated PCA implementation
from ollama import chat  # Interface to local Ollama LLM for text generation

# --------------------------------------
# Helper: timestamped logging
# --------------------------------------
def log(msg: str):
    """Print a message prefixed with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --------------------------------------
# 1) Paths & Model Loading
# --------------------------------------
# Determine script root and results directory
ROOT = Path(__file__).resolve().parent
RES  = ROOT.parent / 'results'

# Load pretrained composition model and metadata
log("1) Loading trained composition model…")
md       = joblib.load(RES / 'composition_model.joblib')
model    = md['model']     # MultiOutputRegressor with RandomForest base
features = md['features']  # List of feature column names
elements = md['elements']  # List of element target column names
log(f"   • Model ready: {len(features)} features → {len(elements)} elements\n")

# --------------------------------------
# 2) Robust Feature Loading (Parquet → CSV fallback)
# --------------------------------------
def load_features():
    """
    Attempt to load cleaned features from Parquet using pyarrow or fastparquet.
    If unsuccessful, fall back to CSV.
    Exit with error if no file is found.
    """
    # Path to cleaned features parquet
    fp = RES / 'features_clean.parquet'
    if fp.exists():
        # Try common Parquet engines
        for eng in ('pyarrow', 'fastparquet'):
            log(f"2) Trying {fp.name} via {eng}…")
            try:
                df = pd.read_parquet(fp, engine=eng)
                log(f"   ✔ Loaded via {eng}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"   ✖ {eng} failed: {e}")
    # Fallback to CSV if Parquet is unavailable
    csv = RES / 'materials_features.csv'
    if csv.exists():
        log(f"2) Loading {csv.name} as CSV…")
        df = pd.read_csv(csv, index_col=0)
        log(f"   ✔ CSV loaded, shape={df.shape}\n")
        return df
    # No valid file found: abort
    sys.exit("ERROR: No feature table found in results/")

log("2) Loading feature table…")
df_feat = load_features()

# --------------------------------------
# 3) Load Structural Descriptors
# --------------------------------------
log("3) Loading structural descriptors…")
# Read structural descriptor Parquet file using pyarrow engine
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
# Rename 'volume' to avoid collision with feature columns
df_struc = df_struc.rename(columns={'volume': 'volume_cell'})
log(f"   ✔ Loaded structural: {df_struc.shape}\n")

# --------------------------------------
# 4) Merge Feature & Structure Tables
# --------------------------------------
log("4) Merging feature + structural DataFrames…")
# Inner join on index; fill missing values with zero for numeric safety
df_all = df_feat.join(df_struc, how='inner').fillna(0)
log(f"   ✔ Merged DataFrame shape: {df_all.shape}\n")

# --------------------------------------
# 5) GPU-Accelerated Dimensionality Reduction (cuML PCA)
# --------------------------------------
TARGET_DIMS = 256  # Desired reduced dimension
log(f"5) Projecting from {len(features)} → {TARGET_DIMS} dims on GPU via cuML PCA…")
start = time.time()
# Convert selected features to a RAPIDS GPU DataFrame
gdf = cudf.DataFrame.from_pandas(df_all[features])
# Initialize cuML PCA with Jacobi solver for stability
pca = cuPCA(n_components=TARGET_DIMS, svd_solver='jacobi', random_state=42)
# Fit and transform entirely on GPU; returns cuDF
X_reduced_gpu = pca.fit_transform(gdf)
# Transfer reduced data back to NumPy array for FAISS and model
X_reduced = X_reduced_gpu.to_numpy().astype('float32')
log(f"   ✔ Projection done in {time.time() - start:.1f}s; reduced shape = {X_reduced.shape}\n")

# --------------------------------------
# 6) Build & Move IVF-PQ Index to GPU (FAISS)
# --------------------------------------
nlist = 256  # Number of Voronoi cells (coarse quantizer)
m     = 16   # Number of sub-quantizers for product quantization
nbits = 8    # Bits per subvector (256 possible centroids)

log(f"6) Building CPU IVF-PQ index (nlist={nlist}, m={m}, nbits={nbits})…")
# Create flat L2 index as quantizer
gpu_quant = faiss.IndexFlatL2(TARGET_DIMS)
cpu_index = faiss.IndexIVFPQ(gpu_quant, TARGET_DIMS, nlist, m, nbits)
# Train codebooks on CPU data
cpu_index.train(X_reduced)
# Add all reduced vectors to the index
cpu_index.add(X_reduced)
log(f"   • CPU index trained & added {cpu_index.ntotal} vectors\n")

# Transfer the trained index from CPU to GPU
log("   • Transferring index to GPU…")
res_gpu  = faiss.StandardGpuResources()           # Allocate GPU resources
gpu_index = faiss.index_cpu_to_gpu(res_gpu, 0, cpu_index)  # Copy index to GPU
log("   ✔ GPU index ready\n")

# Preserve mapping from FAISS positions to original system IDs
ids = df_all.index.to_numpy()

# --------------------------------------
# 7) Prediction & Nearest-Neighbor Search
# --------------------------------------
def predict_composition(props: dict) -> dict:
    """Return predicted element fractions for given property dict."""
    # Build input vector in correct order, filling missing props with 0
    x = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    frac = model.predict(x)[0]
    return dict(zip(elements, frac))


def retrieve_neighbors(props: dict, k: int = 5):
    """Return top-k nearest system IDs and L2 distances for given props."""
    # Create 1×D input array and convert to GPU frame
    x = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    x_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(x, columns=features))
    # Project on GPU via the same PCA transformer
    x_red = pca.transform(x_gpu).to_numpy().astype('float32')
    # Perform GPU-accelerated nearest neighbor search
    D, I = gpu_index.search(x_red, k)
    return ids[I[0]], D[0]

# --------------------------------------
# 8) Ollama Summarization
# --------------------------------------
def summarize(props: dict, comp: dict, neigh_ids: np.ndarray, dists: np.ndarray) -> str:
    """Generate a concise materials recommendation via Ollama LLM."""
    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "The model predicts these element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"Top-5 nearest known systems: IDs {neigh_ids.tolist()} distances {dists.tolist()}.\n"
        "Provide a concise recommendation explaining these findings."
    )
    # Send prompt to local Ollama chat model and return content
    resp = chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=False
    )
    return resp.message.content

# --------------------------------------
# 9) CLI Parsing & Run Pipeline
# --------------------------------------
if __name__ == "__main__":
    # Expect feature-value pairs as CLI args (e.g., --hform -1.2)
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    # Parse properties dict from args
    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0, len(args), 2)}
    log(f"INPUT properties: {props}\n")

    # Execute prediction and neighbor retrieval
    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props, k=5)

    # Log and display predicted compositions
    log("Predicted composition fractions:")
    for el, val in comp.items():
        print(f"  • {el:>2}: {val:.3f}")

    # Log and display nearest neighbor IDs & distances
    log("Nearest neighbors (ID : distance):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    # Generate and print human-readable recommendation
    log("Generating LLM summary via Ollama…")
    summary = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", summary)

