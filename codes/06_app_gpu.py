
#!/usr/bin/env python3
"""
app_gpu_optimized.py

End-to-end materials recommender using GPU-accelerated workflows:

 1) cuML PCA (Jacobi) for fast, in-GPU dimensionality reduction 
    (no more hours-long TruncatedSVD) :contentReference[oaicite:0]{index=0}
 2) FAISS IVF-PQ index built on CPU then transferred to GPU
    (exact search on reduced dims, minimal GPU memory) :contentReference[oaicite:1]{index=1}
 3) Ollama local LLM for human-readable summaries :contentReference[oaicite:2]{index=2}

Usage:
    python app_gpu_optimized.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

import sys, time, json
from pathlib import Path

import joblib                              # load the trained model :contentReference[oaicite:3]{index=3}
import pandas as pd
import numpy as np
import faiss                                # vector search library :contentReference[oaicite:4]{index=4}
import cudf                                 # RAPIDS GPU DataFrames :contentReference[oaicite:5]{index=5}
from cuml.decomposition import PCA as cuPCA # GPU-accelerated PCA :contentReference[oaicite:6]{index=6}
from ollama import chat                    # local LLM interface :contentReference[oaicite:7]{index=7}

def log(msg: str):
    """Timestamped print."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# —————————————————————————————————————————————————————————
# 1) Paths & Model Loading
# —————————————————————————————————————————————————————————
ROOT = Path(__file__).resolve().parent
RES  = ROOT.parent / 'results'

log("1) Loading trained composition model…")
md      = joblib.load(RES / 'composition_model.joblib')
model   = md['model']
features= md['features']
elements= md['elements']
log(f"   • Model ready: {len(features)} features → {len(elements)} elements\n")

# —————————————————————————————————————————————————————————
# 2) Robust Feature Loading (PyArrow → FastParquet → CSV)
# —————————————————————————————————————————————————————————
def load_features():
    fp = RES / 'features_clean.parquet'
    if fp.exists():
        for eng in ('pyarrow','fastparquet'):
            log(f"2) Trying {fp.name} via {eng}…")
            try:
                df = pd.read_parquet(fp, engine=eng)
                log(f"   ✔ Loaded via {eng}, shape={df.shape}\n")
                return df
            except Exception as e:
                log(f"   ✖ {eng} failed: {e}")
    # Fallback to CSV
    csv = RES / 'materials_features.csv'
    if csv.exists():
        log(f"2) Loading {csv.name} as CSV…")
        df = pd.read_csv(csv, index_col=0)
        log(f"   ✔ CSV loaded, shape={df.shape}\n")
        return df
    sys.exit("ERROR: No feature table found in results/")

log("2) Loading feature table…")
df_feat = load_features()

# —————————————————————————————————————————————————————————
# 3) Load Structural Descriptors
# —————————————————————————————————————————————————————————
log("3) Loading structural descriptors…")
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
if 'volume' in df_struc.columns:
    df_struc = df_struc.rename(columns={'volume':'volume_cell'})
    log("   • Renamed 'volume'→'volume_cell'")
log(f"   ✔ Loaded structural: {df_struc.shape}\n")

# —————————————————————————————————————————————————————————
# 4) Merge Feature & Structure Tables
# —————————————————————————————————————————————————————————
log("4) Merging feature + structural DataFrames…")
df_all = df_feat.join(df_struc, how='inner').fillna(0)
log(f"   ✔ Merged DataFrame shape: {df_all.shape}\n")

# —————————————————————————————————————————————————————————
# 5) GPU-Accelerated Dimensionality Reduction (cuML PCA)
# —————————————————————————————————————————————————————————
TARGET_DIMS = 256
log(f"5) Projecting ~{len(features)} dims → {TARGET_DIMS} dims on GPU via cuML PCA…")
start = time.time()

# Convert to GPU dataframe
gdf = cudf.DataFrame.from_pandas(df_all[features])
pca = cuPCA(n_components=TARGET_DIMS, svd_solver='jacobi', random_state=42)
X_reduced_gpu = pca.fit_transform(gdf)  # returns cuDF
X_reduced = X_reduced_gpu.to_numpy().astype('float32')  # back to NumPy
log(f"   ✔ Projection done in {time.time()-start:.1f}s; reduced shape = {X_reduced.shape}\n")

# —————————————————————————————————————————————————————————
# 6) Build & Move IVF-PQ Index to GPU (FAISS)
# —————————————————————————————————————————————————————————
nlist = 256  # Voronoi cells
m     = 16   # PQ subquantizers
nbits = 8    # bits per code

log(f"6) Building CPU IVF-PQ index (nlist={nlist}, m={m}, nbits={nbits})…")
cpu_quant = faiss.IndexFlatL2(TARGET_DIMS)
cpu_index = faiss.IndexIVFPQ(cpu_quant, TARGET_DIMS, nlist, m, nbits)
cpu_index.train(X_reduced)  # train on CPU :contentReference[oaicite:8]{index=8}
cpu_index.add(X_reduced)
log(f"   • CPU index trained & added {cpu_index.ntotal} vectors\n")

# Transfer CPU index to GPU
log("   • Transferring index to GPU…")
res_gpu = faiss.StandardGpuResources()  # allocate GPU resources :contentReference[oaicite:9]{index=9}
gpu_index = faiss.index_cpu_to_gpu(res_gpu, 0, cpu_index)
log("   ✔ GPU index ready\n")

# Prepare ID mapping
ids = df_all.index.to_numpy()

# —————————————————————————————————————————————————————————
# 7) Prediction & Nearest-Neighbor Search
# —————————————————————————————————————————————————————————
def predict_composition(props: dict) -> dict:
    x = np.array([[props.get(f,0) for f in features]], dtype='float32')
    frac = model.predict(x)[0]
    return dict(zip(elements, frac))

def retrieve_neighbors(props: dict, k: int=5):
    x = np.array([[props.get(f,0) for f in features]], dtype='float32')
    x_red = pca.transform(cudf.DataFrame.from_pandas(pd.DataFrame(x, columns=features))).to_numpy().astype('float32')
    D, I = gpu_index.search(x_red, k)  # GPU search :contentReference[oaicite:10]{index=10}
    return ids[I[0]], D[0]

# —————————————————————————————————————————————————————————
# 8) Ollama Summarization
# —————————————————————————————————————————————————————————
def summarize(props, comp, neigh_ids, dists) -> str:
    prompt = (
        "You are a materials scientist. A user requested a 2D material with properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "The model predicts these element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        f"Top-5 nearest known systems: IDs {neigh_ids.tolist()} distances {dists.tolist()}.\n"
        "Provide a concise recommendation explaining these findings."
    )
    resp = chat(model='qwen2.5:7b', messages=[{'role':'user','content':prompt}], stream=False)
    return resp.message.content

# —————————————————————————————————————————————————————————
# 9) CLI Parsing & Run Pipeline
# —————————————————————————————————————————————————————————
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)<2 or len(args)%2!=0:
        print(__doc__)
        sys.exit(1)

    props = {args[i].lstrip('-'): float(args[i+1]) for i in range(0,len(args),2)}
    log(f"INPUT properties: {props}\n")

    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props, k=5)

    log("Predicted composition fractions:")
    for el,val in comp.items():
        print(f"  • {el:>2}: {val:.3f}")

    log("Nearest neighbors (ID : distance):")
    for sid,dist in zip(neighs, dists):
        print(f"  • {sid}: {dist:.3f}")

    log("Generating LLM summary via Ollama…")
    summary = summarize(props, comp, neighs, dists)
    print("\nRecommendation:\n", summary)
