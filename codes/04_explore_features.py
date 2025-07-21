
#!/usr/bin/env python3
# describe_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Base paths
root = Path(__file__).resolve().parent.parent
res  = root / 'results'

# --- 1) Load “clean” features, with fallbacks ---
def load_features():
    # 1a) try cleaned parquet
    try:
        df = pd.read_parquet(res / 'features_clean.parquet')
        print("Loaded features_clean.parquet")
        return df
    except Exception as e:
        print(f"❗ could not load features_clean.parquet ({e})")

    # 1b) try full parquet
    try:
        df = pd.read_parquet(res / 'features_full.parquet')
        print("Loaded features_full.parquet")
        return df
    except Exception as e:
        print(f"❗ could not load features_full.parquet ({e})")

    # 1c) fallback to CSV
    try:
        df = pd.read_csv(res / 'materials_features.csv', index_col=0)
        print("Loaded materials_features.csv")
        return df
    except Exception as e:
        print(f"✖︎ could not load materials_features.csv ({e})")
        sys.exit("ERROR: no feature table available to visualize.")

df_feat = load_features()

# --- 2) Load structural descriptors ---
try:
    df_struc = pd.read_parquet(res / 'structural_descriptors.parquet')
    print("Loaded structural_descriptors.parquet")
except Exception as e:
    sys.exit(f"ERROR: cannot load structural_descriptors.parquet ({e})")

# --- 2a) Rename overlapping columns to avoid merge conflicts ---
df_struc = df_struc.rename(columns={'volume': 'volume_cell'})

# --- 3) Merge on 'id' ---
df = df_feat.join(df_struc, how='left')
print(f"Merged feature table shape: {df.shape}")

# --- 4a) Normalize numeric columns ---
num   = df.select_dtypes(include='number').copy()
means = num.mean()
stds  = num.std(ddof=0)
num_norm = (num - means) / stds

# --- 4b) Compute & display stats ---
stats = pd.DataFrame({
    'mean':     means,
    'variance': num.var(ddof=0),
    'skew':     num.skew()
})
print("\nTop 10 features by variance:")
print(stats.sort_values('variance', ascending=False).head(10))

# Save the stats table
stats_out = res / 'feature_statistics.csv'
stats.to_csv(stats_out)
print(f"\nSaved feature statistics to {stats_out}")

# --- 4c) Plot distributions of key features ---
to_plot = ['hform', 'Egap', 'thickness', 'a', 'density_amu_per_A3', 'cn_mean']
for feat in to_plot:
    if feat not in num_norm.columns:
        print(f"– skipping '{feat}': not found in data")
        continue
    plt.figure(figsize=(6,4))
    num_norm[feat].hist(bins=50, density=True)
    plt.title(f"{feat} (normalized)")
    plt.xlabel("z-score")
    plt.ylabel("density")
    png = res / f"{feat}_dist.png"
    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    print(f"→ saved distribution plot: {png}")

print("\n✅ Step 4 complete: normalized, computed stats, and plotted distributions.")

