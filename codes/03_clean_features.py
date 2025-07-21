
#!/usr/bin/env python3
# wrangle_features_commented.py

# --------------------------------------
# Script to clean and wrangle feature matrix
# Steps:
# 1) Load the full feature matrix from Parquet
# 2) Compute missing-value fractions per column
# 3) Drop features with >50% missing values
# 4) Impute numeric features using column medians
# 5) One-hot encode textual (object dtype) features
# 6) Save the cleaned feature matrix back to Parquet
# --------------------------------------

# Standard library imports
from pathlib import Path  # For filesystem path handling

# Third-party imports
import pandas as pd  # For DataFrame manipulation and I/O


# --------------------------------------
# 1) Load the full feature matrix
# Determine path to the Parquet file containing raw features
out = (
    Path(__file__).resolve()      # Absolute path of this script
         .parent.parent             # Up two levels to project root
    / 'results' / 'features_full.parquet'  # Location of raw feature matrix
)
# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet(out)
# Total number of systems (rows) for missing fraction calculations
N = len(df)

# --------------------------------------
# 2) Compute missing-value fractions for each feature
# miss_frac[j] = (# missing in column j) / N
miss_frac = df.isna().mean().sort_values(ascending=False)
# Display top 10 features by percentage missing
print("Top 10 features by % missing:\n", (miss_frac * 100).head(10))

# --------------------------------------
# 3) Drop features with more than 50% missing values
threshold = 0.5  # Fraction threshold for dropping features
to_drop = miss_frac[miss_frac > threshold].index  # Column names to drop
df = df.drop(columns=to_drop)  # Remove these columns
print(f"Dropped {len(to_drop)} features; remaining shape: {df.shape}")

# --------------------------------------
# 4) Impute missing values in numeric columns with the median
# Identify numeric columns by dtype
num_cols = df.select_dtypes(include='number').columns
# Compute column medians
die medians = df[num_cols].median()
# Fill NaNs in numeric columns with median values
df[num_cols] = df[num_cols].fillna(medians)

# --------------------------------------
# 5) One-hot encode textual (object dtype) columns
# Get columns of object (string) type
txt_cols = df.select_dtypes(include='object').columns
# Use pandas.get_dummies to create binary indicator columns
# dummy_na=True adds a column to indicate original NaNs
# This expands the DataFrame horizontally
df = pd.get_dummies(df, columns=txt_cols, dummy_na=True)

# --------------------------------------
# 6) Save the cleaned DataFrame to Parquet
# Construct output path in the same results directory
clean_out = out.with_name('features_clean.parquet')
# Write the cleaned DataFrame as Parquet
# Parquet preserves data types and is efficient for storage and I/O
df.to_parquet(clean_out)
# Inform user of save location
print("Cleaned features saved to:", clean_out)

