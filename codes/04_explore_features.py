
#!/usr/bin/env python3
# describe_visualize_commented.py

# --------------------------------------
# Script to load cleaned feature data and
# structural descriptors, merge them,
# compute basic statistics, normalize,
# and visualize selected feature distributions.
#
# Steps:
# 1) Load features with fallback order: cleaned Parquet → full Parquet → CSV
# 2) Load structural descriptors and rename overlapping columns
# 3) Merge features and descriptors on system ID
# 4) Normalize numeric data, compute and save summary statistics
# 5) Plot distribution histograms for key features
# --------------------------------------

# Standard library imports
import sys                     # For clean exit on critical errors
from pathlib import Path       # For file system path manipulations

# Third-party imports
import pandas as pd           # For DataFrame I/O and manipulation
import matplotlib.pyplot as plt  # For plotting feature distributions

# --------------------------------------
# Define base directories for input and output files
root = Path(__file__).resolve().parent.parent  # Project root (two levels up)
res  = root / 'results'                        # Results directory for I/O

# --------------------------------------
# 1) Load feature DataFrame with multiple fallbacks

def load_features() -> pd.DataFrame:
    """
    Attempt to load cleaned features first (Parquet),
    then fall back to full features (Parquet),
    and finally CSV if necessary.
    Exits on complete failure.
    """
    # 1a) Try cleaned Parquet
    try:
        df = pd.read_parquet(res / 'features_clean.parquet')
        print("Loaded features_clean.parquet")
        return df
    except Exception as e:
        print(f"❗ could not load features_clean.parquet ({e})")

    # 1b) Try full Parquet
    try:
        df = pd.read_parquet(res / 'features_full.parquet')
        print("Loaded features_full.parquet")
        return df
    except Exception as e:
        print(f"❗ could not load features_full.parquet ({e})")

    # 1c) Fallback to CSV
    try:
        df = pd.read_csv(res / 'materials_features.csv', index_col=0)
        print("Loaded materials_features.csv")
        return df
    except Exception as e:
        print(f"✖︎ could not load materials_features.csv ({e})")
        sys.exit("ERROR: no feature table available to visualize.")

# Load features into df_feat
df_feat = load_features()

# --------------------------------------
# 2) Load structural descriptors and handle errors
try:
    df_struc = pd.read_parquet(res / 'structural_descriptors.parquet')
    print("Loaded structural_descriptors.parquet")
except Exception as e:
    sys.exit(f"ERROR: cannot load structural_descriptors.parquet ({e})")

# 2a) Rename any overlapping columns to avoid conflicts on merge
# For example, rename 'volume' to 'volume_cell'
df_struc = df_struc.rename(columns={'volume': 'volume_cell'})

# --------------------------------------
# 3) Merge feature and structural DataFrames on their index ('id')
# Left join ensures all features retained even if descriptors missing
# The result df has all numeric and categorical features together
df = df_feat.join(df_struc, how='left')
print(f"Merged feature table shape: {df.shape}")

# --------------------------------------
# 4a) Normalize numeric columns using z-score (mean=0, std=1)
# Copy ensures original df remains unchanged
num = df.select_dtypes(include='number').copy()
means = num.mean()                  # Series of column means
stds  = num.std(ddof=0)             # Population standard deviation
# Apply normalization formula: (x - mean) / std
num_norm = (num - means) / stds

# --------------------------------------
# 4b) Compute summary statistics (mean, variance, skewness) on raw data
stats = pd.DataFrame({
    'mean':     means,
    'variance': num.var(ddof=0),
    'skew':     num.skew()
})
print("\nTop 10 features by variance:")
print(stats.sort_values('variance', ascending=False).head(10))

# Save statistics to CSV for later reference
stats_out = res / 'feature_statistics.csv'
stats.to_csv(stats_out)
print(f"\nSaved feature statistics to {stats_out}")

# --------------------------------------
# 4c) Plot normalized distribution for selected key features
# Define list of features to visualize
to_plot = ['hform', 'Egap', 'thickness', 'a', 'density_amu_per_A3', 'cn_mean']
for feat in to_plot:
    # Check existence before plotting
    if feat not in num_norm.columns:
        print(f"– skipping '{feat}': not found in data")
        continue
    # Create a new figure for each feature
    plt.figure(figsize=(6,4))
    # Plot histogram with density (area=1)
    num_norm[feat].hist(bins=50, density=True)
    plt.title(f"{feat} (normalized)")
    plt.xlabel("z-score")
    plt.ylabel("density")
    # Save each plot to PNG in results directory
    png = res / f"{feat}_dist.png"
    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    print(f"→ saved distribution plot: {png}")

print("\n✅ Step 4 complete: normalized, computed stats, and plotted distributions.")

