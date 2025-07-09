
#!/usr/bin/env python3
# wrangle_features.py

import pandas as pd
from pathlib import Path

# 1) Load the full feature matrix
out = Path(__file__).resolve().parent.parent / 'results' / 'features_full.parquet'
df = pd.read_parquet(out)
N = len(df)  # number of systems

# 2) Compute missing‐value fractions
#    m_j = (# of NaNs in col j) / N
miss_frac = df.isna().mean().sort_values(ascending=False)
print("Top 10 features by % missing:\n", (miss_frac*100).head(10))

# 3) Drop features with >50% missing
threshold = 0.5
to_drop = miss_frac[miss_frac > threshold].index
df = df.drop(columns=to_drop)
print(f"Dropped {len(to_drop)} features; remaining shape: {df.shape}")

# 4) Impute numeric columns with median
num_cols = df.select_dtypes(include='number').columns
medians = df[num_cols].median()
df[num_cols] = df[num_cols].fillna(medians)

# 5) One‐hot encode textual columns
txt_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=txt_cols, dummy_na=True)

# 6) Save the cleaned version
clean_out = out.with_name('features_clean.parquet')
df.to_parquet(clean_out)
print("Cleaned features saved to:", clean_out)
