
#!/usr/bin/env python3
# app_cached_pipeline_commented.py

"""
Highly optimized, cached end-to-end materials recommender pipeline.

Features:
 0) Force all computations and libraries to utilize all CPU cores.
 1) Load pre-trained composition model and metadata.
 2) Robust Parquet loading with caching, corruption detection, and fallback I/O.
 3) Merge cleaned features and structural descriptors into a unified DataFrame.
 4) Dimensionality reduction via cached GaussianRandomProjection.
 5) FAISS IndexIVFPQ build or load with caching for approximate nearest neighbors.
 6) Fast composition prediction and neighbor retrieval functions.
 7) Optional Ollama LLM summarization with failure handling.
 8) Simple CLI interface parsing feature–value pairs.

Usage:
    python app_cached_pipeline.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0) Parallelism setup: ensure all relevant libraries use maximum CPU threads
# ─────────────────────────────────────────────────────────────────────────────
import os
# Determine number of available CPU cores (fallback to 1 if detection fails)
n_threads = os.cpu_count() or 1
# Environment variables control thread usage in various numeric libraries
ios.environ["OMP_NUM_THREADS"]      = str(n_threads)
ios.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
ios.environ["MKL_NUM_THREADS"]      = str(n_threads)
ios.environ["NUMEXPR_NUM_THREADS"]  = str(n_threads)
ios.environ["ARROW_NUM_THREADS"]    = str(n_threads)

# Additional imports for model loading and data I/O
import sys, time, json  # sys: CLI args & exit; time: logging timestamps; json: prompt serialization
from pathlib import Path  # Pathlib for robust path manipulations

# Third-party imports for caching, dataframes, projections, vector search, and LLM
import joblib                             # Persistent caching/loading of Python objects
import pandas as pd                       # DataFrame operations and Parquet I/O
import numpy as np                        # Numerical array operations
from sklearn.random_projection import GaussianRandomProjection  # Fast random projection
import faiss                              # Approximate nearest neighbor library
import pyarrow.parquet as pq              # Low-level Parquet reader for fallback
from ollama import chat                   # Local LLM interface for text summarization

# Configure FAISS to also use all threads for CPU-based index operations
if hasattr(faiss, 'omp_set_num_threads'):
    faiss.omp_set_num_threads(n_threads)

# ─────────────────────────────────────────────────────────────────────────────
# Logging helper: prints timestamped messages for progress tracking
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str):
    """Print a message prefixed with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Cache Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Define project root and results directory relative to this script
ROOT       = Path(__file__).resolve().parent
RES        = ROOT.parent / "results"
# Create a dedicated cache directory for intermediate artifacts
CACHE_DIR  = RES / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Paths for cached objects
GRP_FILE    = CACHE_DIR / "grp.pkl"         # GaussianRandomProjection cache
INDEX_FILE  = CACHE_DIR / "faiss.idx"      # FAISS index cache
FEAT_CACHE  = CACHE_DIR / "features.pkl"    # Cleaned features DataFrame cache
STRUC_CACHE = CACHE_DIR / "structural.pkl"  # Structural descriptors cache

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load pre-trained composition model and its metadata
# ─────────────────────────────────────────────────────────────────────────────
log("1) Loading composition model …")
# The composition_model.joblib contains a dict with 'model', 'features', 'elements'
model_data = joblib.load(RES / "composition_model.joblib")
model      = model_data["model"]      # Trained MultiOutputRegressor
features   = model_data["features"]   # Ordered feature column names
elements   = model_data["elements"]   # Element target column names
log(f"   • Loaded: {len(features)} features → {len(elements)} elements")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Robust Parquet loading with caching and corruption handling
# ─────────────────────────────────────────────────────────────────────────────
def load_parquet_with_cache(parquet_path: Path, pickle_path: Path) -> pd.DataFrame:
    """
    Load a DataFrame from Parquet, with persistent caching to pickle.
    If cached pickle exists, try loading; on failure delete cache and retry Parquet.
    Use fastparquet first; on error, fall back to pyarrow reader.
    """
    # 2a) Attempt to load from pickle cache if present
    if pickle_path.exists():
        try:
            log(f"   • Loading cached {pickle_path.name}")
            return joblib.load(pickle_path)
        except Exception as e:
            log(f"   • Cache load failed ({type(e).__name__}); removing cache & retry")
            pickle_path.unlink(missing_ok=True)
    # 2b) Read raw Parquet file via fastparquet for speed
    log(f"   • Reading Parquet {parquet_path.name} with fastparquet")
    try:
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    except Exception as e:
        # On failure, use pyarrow low-level API with large thrift container for big files
        log(f"   • fastparquet failed ({e}); using pyarrow fallback")
        pf = pq.ParquetFile(
            str(parquet_path),
            memory_map=True,
            thrift_container_size_limit=10**9,
        )
        df = pf.read(use_threads=True).to_pandas()
    # 2c) Attempt to write to pickle cache for next runs
    try:
        joblib.dump(df, pickle_path)
    except Exception as e:
        log(f"   • Warning: could not write cache ({e})")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3) Load cleaned feature table using the caching loader
# ─────────────────────────────────────────────────────────────────────────────
log("2) Loading feature table …")
df_feat = load_parquet_with_cache(RES / "features_clean.parquet", FEAT_CACHE)
log(f"   • Features shape: {df_feat.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4) Load structural descriptors with caching and rename column conflict
# ─────────────────────────────────────────────────────────────────────────────
log("3) Loading structural descriptors …")
df_struc = load_parquet_with_cache(RES / "structural_descriptors.parquet", STRUC_CACHE)
# Rename 'volume' to 'volume_cell' to avoid column name collision
if "volume" in df_struc.columns:
    df_struc = df_struc.rename(columns={"volume": "volume_cell"})
log(f"   • Structural shape: {df_struc.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Merge cleaned features and structural descriptors into a unified DataFrame
# ─────────────────────────────────────────────────────────────────────────────
log("4) Merging features + structures …")
# Use inner join on index 'id', then fill any missing values with 0 for numerical safety
df_all = df_feat.join(df_struc, how="inner").fillna(0)
log(f"   • Merged shape: {df_all.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Dimensionality reduction via GaussianRandomProjection with persistent caching
# ─────────────────────────────────────────────────────────────────────────────
TARGET_DIMS = 256  # Project down to this many dimensions
if GRP_FILE.exists():
    log("5) Loading cached RandomProjection model …")
    grp = joblib.load(GRP_FILE)
else:
    log(f"5) Computing RandomProjection → {TARGET_DIMS} dims …")
    # Instantiate and fit on full feature matrix (float32 for efficiency)
    grp = GaussianRandomProjection(n_components=TARGET_DIMS, random_state=42)
    grp.fit(df_all[features].values.astype("float32"))
    # Cache for future runs to avoid recomputation
    joblib.dump(grp, GRP_FILE)
    log("   • Cached RandomProjection model")

# ─────────────────────────────────────────────────────────────────────────────
# 7) Build or load FAISS IVF-PQ index with caching
# ─────────────────────────────────────────────────────────────────────────────
if INDEX_FILE.exists():
    log("6) Loading cached FAISS index …")
    index = faiss.read_index(str(INDEX_FILE))
else:
    log("6) Building FAISS IndexIVFPQ …")
    # Transform features into reduced space for indexing
    X = df_all[features].values.astype("float32")
    X_red = grp.transform(X)
    # Instantiate index with hyperparameters: nlist=256 cells, m=16 subvectors, 8 bits
    quantizer = faiss.IndexFlatL2(TARGET_DIMS)
    index = faiss.IndexIVFPQ(quantizer, TARGET_DIMS, 256, 16, 8)
    # Set probing parameter for queries (max 10 clusters)
    index.nprobe = min(10, getattr(index, 'nlist', 10))
    # Train on reduced dataset then add all vectors to index
    log("   • Training index …")
    index.train(X_red)
    log("   • Adding vectors …")
    index.add(X_red)
    # Persist index to disk for reuse
    faiss.write_index(index, str(INDEX_FILE))
    log("   • Cached FAISS index")

# Preserve mapping of FAISS positions back to original system IDs
ids = df_all.index.values

# ─────────────────────────────────────────────────────────────────────────────
# 8) Prediction and neighbor retrieval functions
# ─────────────────────────────────────────────────────────────────────────────
def predict_composition(props: dict) -> dict:
    """Return predicted element fractions for a given properties dict."""
    # Build input array of shape (1, n_features), defaulting missing props to 0
    x = np.array([[props.get(f, 0) for f in features]], dtype="float32")
    preds = model.predict(x)[0]  # Get 1D array of predicted fractions
    return dict(zip(elements, preds))


def retrieve_neighbors(props: dict, k: int = 5):
    """Return the top-k nearest system IDs and distances for a given props."""
    # Construct input vector and reduce dimensionality
    x = np.array([[props.get(f, 0) for f in features]], dtype="float32")
    Xq = grp.transform(x)  # Project into reduced space
    # Perform Faiss search: returns (indices, distances)
    I, D = index.search(Xq, k)
    idx = I[0].astype(int)  # Convert indices to Python ints
    return ids[idx], D[0]

# ─────────────────────────────────────────────────────────────────────────────
# 9) Summarization via Ollama LLM with error handling
# ─────────────────────────────────────────────────────────────────────────────
def summarize(props: dict, comp: dict, neigh_ids: np.ndarray, dists: np.ndarray) -> str:
    """Generate a concise recommendation using Ollama local LLM."""
    # Build a user prompt that includes input properties, predictions, and neighbors
    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "Predicted element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"Nearest IDs {list(neigh_ids)} with distances {list(dists)}.\n"
        "Provide a concise recommendation explaining these results."
    )
    try:
        resp = chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.message.content
    except Exception as e:
        log(f"   • Ollama chat failed: {e}")
        # Fallback message if LLM call fails
        return "Recommendation unavailable due to chat error."

# ─────────────────────────────────────────────────────────────────────────────
# 10) Simple command-line interface parsing and pipeline execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Expect even number of args: --feature value pairs
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    # Parse CLI into a dict of property names and float values
    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0, len(args), 2)}
    log(f"INPUT: {props}")

    # Run prediction and retrieval
    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props)

    # Display predicted composition fractions
    log("Composition fractions:")
    for el, val in comp.items():
        print(f"  • {el}: {val:.3f}")

    # Display nearest neighbors and distances
    log("Nearest neighbors (ID:dist):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    # Generate and display LLM-based summary
    log("Summarizing via Ollama …")
    recommendation = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", recommendation)

