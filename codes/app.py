
#!/usr/bin/env python3
"""
app.py

End-to-end materials recommender using FAISS + a local Ollama model.

Steps:
 1. Load trained composition model (joblib).
 2. Load cleaned features & structural descriptors (Parquet).
 3. Build FAISS index over feature vectors.
 4. Predict composition fractions from user inputs.
 5. Retrieve nearest known materials.
 6. Summarize recommendations via Ollama chat.

Usage:
  python app.py --hform -1.2 --Egap 1.5 --thickness 3.2
"""

import os
import sys
import json
from pathlib import Path

import joblib                              # for loading the trained model :contentReference[oaicite:0]{index=0}
import pandas as pd                       # for data handling
import numpy as np                        # for numeric operations
import faiss                              # FAISS for nearest-neighbor search :contentReference[oaicite:1]{index=1}
from ollama import chat                   # Ollama Python client for local LLM :contentReference[oaicite:2]{index=2}

# ---------------------------
# 1. Paths and Model Loading
# ---------------------------

ROOT = Path(__file__).resolve().parent
RES  = ROOT.parent / './results'

print("🔍 Loading composition model and metadata…")
# Load the composition model (MultiOutputRegressor + feature & element lists)
model_data = joblib.load(RES / 'composition_model.joblib')
model      = model_data['model']
features   = model_data['features']
elements   = model_data['elements']
print(f"   • Model loaded. Predicting {len(elements)} element fractions from {len(features)} features.")

# Load cleaned feature table (metadata pivots)
print("🔍 Loading cleaned features (features_clean.parquet)…")
df_feat = pd.read_parquet(RES / 'features_clean.parquet', engine='pyarrow')
print(f"   • features_clean shape: {df_feat.shape}")

# Load structural descriptors and rename to avoid conflicts
print("🔍 Loading structural descriptors (structural_descriptors.parquet)…")
df_struc = pd.read_parquet(RES / 'structural_descriptors.parquet', engine='pyarrow')
df_struc = df_struc.rename(columns={'volume': 'volume_cell'})
print(f"   • structural_descriptors shape: {df_struc.shape}")

# Merge into a single DataFrame
print("🔗 Merging feature tables…")
df_all = df_feat.join(df_struc, how='inner').fillna(0)
print(f"   • Merged shape: {df_all.shape}\n")

# --------------------------------
# 2. Build FAISS Index for Retrieval
# --------------------------------

print("⚙️  Building FAISS index for retrieval…")
X = df_all[features].values.astype('float32')  # feature matrix :contentReference[oaicite:3]{index=3}
dim = X.shape[1]
index = faiss.IndexFlatL2(dim)                  # L2 distance index :contentReference[oaicite:4]{index=4}
index.add(X)
ids = df_all.index.to_numpy()                   # map back to system IDs
print(f"   • FAISS index built over {index.ntotal} vectors of dimension {dim}\n")

# ---------------------------
# 3. Prediction & Retrieval
# ---------------------------

def predict_composition(props: dict) -> dict:
    """
    Given a dict of property_name→value, returns element fractions.
    """
    x = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    frac = model.predict(x)[0]  # shape=(len(elements),)
    return dict(zip(elements, frac))

def retrieve_neighbors(props: dict, k: int = 5):
    """
    Returns top-k nearest system IDs and their L2 distances.
    """
    x = np.array([[props.get(f, 0) for f in features]], dtype='float32')
    D, I = index.search(x, k)
    return ids[I[0]], D[0]

# ---------------------------
# 4. Ollama Summarization
# ---------------------------

def summarize_recommendation(props, comp, neigh_ids, dists) -> str:
    """
    Uses Ollama to generate a concise, human-readable summary of:
     - the predicted composition
     - the retrieved nearest examples
    """
    prompt = (
        "You are a materials scientist. A user requested a 2D material with these properties:\n"
        f"{json.dumps(props, indent=2)}\n\n"
        "Your model predicts the following element fractions:\n"
        f"{json.dumps(comp, indent=2)}\n\n"
        "The top-5 nearest known materials by feature similarity are system IDs "
        f"{neigh_ids.tolist()} with distances {np.round(dists,3).tolist()}.\n\n"
        "Write a concise recommendation explaining the predicted composition "
        "and why the retrieved examples are relevant."
    )
    # Call local Ollama model (e.g. qwen2.5:7b) :contentReference[oaicite:5]{index=5}
    response = chat(
        model='qwen2.5:7b',
        messages=[{'role':'user', 'content':prompt}],
        stream=False
    )
    return response.message.content

# ---------------------------
# 5. Command-Line Interface
# ---------------------------

if __name__ == "__main__":
    # parse CLI args of form --feature value
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    # build properties dict
    props = {}
    for i in range(0, len(args), 2):
        key, val = args[i].lstrip('-'), float(args[i+1])
        props[key] = val
    print("🏁 Input properties:", props)

    # Run prediction + retrieval
    comp    = predict_composition(props)
    neighs, dists = retrieve_neighbors(props, k=5)

    # Display results
    print("\n🎯 Predicted composition fractions:")
    for el, frac in comp.items():
        print(f"  • {el:>2} : {frac:.3f}")

    print("\n🔍 Nearest neighbors (system ID : distance):")
    for sid, dist in zip(neighs, dists):
        print(f"  • {sid} : {dist:.3f}")

    # LLM summary
    print("\n📝 Generating human-readable recommendation via Ollama…")
    summary = summarize_recommendation(props, comp, neighs, dists)
    print("\n💡 Recommendation:\n", summary)
