#!/usr/bin/env python3
import os
import sys, time, json
from pathlib import Path

# ─── 0) Force all libraries to use all cores ─────────────────────────────────
n_threads = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"]      = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"]      = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"]  = str(n_threads)
os.environ["ARROW_NUM_THREADS"]    = str(n_threads)

import joblib
import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import faiss
import pyarrow.parquet as pq
from ollama import chat

# Ensure Faiss uses all threads
dll = faiss.omp_get_max_threads if hasattr(faiss, 'omp_get_max_threads') else None
faiss.omp_set_num_threads(n_threads)

# ─── Logging helper ──────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ─── Paths & cache files ─────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
RES        = ROOT.parent / "results"
CACHE_DIR  = RES / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRP_FILE    = CACHE_DIR / "grp.pkl"
INDEX_FILE  = CACHE_DIR / "faiss.idx"
FEAT_CACHE  = CACHE_DIR / "features.pkl"
STRUC_CACHE = CACHE_DIR / "structural.pkl"

# ─── 1) Load composition model ────────────────────────────────────────────────
log("1) Loading composition model …")
model_data = joblib.load(RES / "composition_model.joblib")
model      = model_data["model"]
features   = model_data["features"]
elements   = model_data["elements"]
log(f"   • Loaded: {len(features)} features → {len(elements)} elements")

# ─── 2) Robust Parquet loader with caching and corruption handling ────────────
def load_parquet_with_cache(parquet_path: Path, pickle_path: Path) -> pd.DataFrame:
    if pickle_path.exists():
        try:
            log(f"   • Loading cached {pickle_path.name}")
            return joblib.load(pickle_path)
        except Exception as e:
            log(f"   • Failed to load cache ({type(e).__name__}); deleting and retrying")
            pickle_path.unlink(missing_ok=True)
    log(f"   • Reading parquet {parquet_path.name} with fastparquet")
    try:
        df = pd.read_parquet(parquet_path, engine="fastparquet")
    except Exception as e:
        log(f"   • fastparquet failed ({e}); using pyarrow")
        pf = pq.ParquetFile(
            str(parquet_path),
            memory_map=True,
            thrift_container_size_limit=10**9,
        )
        df = pf.read(use_threads=True).to_pandas()
    try:
        joblib.dump(df, pickle_path)
    except Exception as e:
        log(f"   • Warning: could not write cache ({e})")
    return df

# ─── 3) Load feature table ───────────────────────────────────────────────────
log("2) Loading feature table …")
df_feat = load_parquet_with_cache(RES / "features_clean.parquet", FEAT_CACHE)
log(f"   • Features shape: {df_feat.shape}")

# ─── 4) Load structural descriptors ────────────────────────────────────────────
log("3) Loading structural descriptors …")
df_struc = load_parquet_with_cache(RES / "structural_descriptors.parquet", STRUC_CACHE)
if "volume" in df_struc.columns:
    df_struc = df_struc.rename(columns={"volume": "volume_cell"})
log(f"   • Structural shape: {df_struc.shape}")

# ─── 5) Merge tables ─────────────────────────────────────────────────────────
log("4) Merging features + structures …")
df_all = df_feat.join(df_struc, how="inner").fillna(0)
log(f"   • Merged shape: {df_all.shape}")

# ─── 6) Dimensionality reduction (RandomProjection) ──────────────────────────
TARGET_DIMS = 256
if GRP_FILE.exists():
    log("5) Loading cached RandomProjection …")
    grp = joblib.load(GRP_FILE)
else:
    log(f"5) Computing RandomProjection → {TARGET_DIMS} dims …")
    grp = GaussianRandomProjection(n_components=TARGET_DIMS, random_state=42)
    grp.fit(df_all[features].values.astype("float32"))
    joblib.dump(grp, GRP_FILE)
    log("   • Cached RandomProjection")

# ─── 7) Build / load FAISS index ──────────────────────────────────────────────
if INDEX_FILE.exists():
    log("6) Loading cached FAISS index …")
    index = faiss.read_index(str(INDEX_FILE))
else:
    log("6) Building FAISS IndexIVFPQ …")
    X = df_all[features].values.astype("float32")
    X_red = grp.transform(X)
    quantizer = faiss.IndexFlatL2(TARGET_DIMS)
    index = faiss.IndexIVFPQ(quantizer, TARGET_DIMS, 256, 16, 8)
    index.nprobe = min(10, getattr(index, 'nlist', 10))
    log("   • Training index …")
    index.train(X_red)
    log("   • Adding vectors …")
    index.add(X_red)
    faiss.write_index(index, str(INDEX_FILE))
    log("   • Cached FAISS index")

ids = df_all.index.values

# ─── 8) Prediction and neighbor retrieval ────────────────────────────────────
def predict_composition(props: dict) -> dict:
    x = np.array([[props.get(f, 0) for f in features]], dtype="float32")
    preds = model.predict(x)[0]
    return dict(zip(elements, preds))

def retrieve_neighbors(props: dict, k: int =5):
    x = np.array([[props.get(f, 0) for f in features]], dtype="float32")
    I, D = index.search(grp.transform(x), k)
    idx = I[0].astype(int)
    return ids[idx], D[0]

# ─── 9) Summarize via Ollama ─────────────────────────────────────────────────
def summarize(props: dict, comp: dict, neigh_ids: np.ndarray, dists: np.ndarray) -> str:
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
        return "Recommendation unavailable due to chat error."

# ─── 10) Simple CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0, len(args), 2)}
    log(f"INPUT: {props}")

    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props)

    log("Composition fractions:")
    for el, val in comp.items():
        print(f"  • {el}: {val:.3f}")

    log("Nearest neighbors (ID:dist):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    log("Summarizing via Ollama …")
    print("\nRecommendation:\n", summarize(props, comp, neighs, dists))
