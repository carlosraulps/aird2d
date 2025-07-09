
#!/usr/bin/env python3
# pivot_merge_features.py

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# 1) adjust or override DB path via CLI
def get_db_path():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists(): return p
        sys.exit(f"ERROR: DB not found at {p}")
    root = Path(__file__).resolve().parent.parent
    for candidate in (root/'databases'/'c2db.db', root/'backup'/'c2db.db'):
        if candidate.exists():
            return candidate
    sys.exit("ERROR: could not locate c2db.db")

# 2) load tables into pandas
def load_tables(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    names = ['systems','number_key_values','text_key_values','species']
    return {n: pd.read_sql_table(n, engine) for n in names}

# 3) build the full feature DataFrame
def build_feature_df(dfs, Z2sym):
    # metadata
    df_sys = dfs['systems'].set_index('id')

    # numeric pivot: X_num ∈ R^{#systems × #unique_keys}
    df_nkv = dfs['number_key_values']
    X_num = df_nkv.pivot(index='id', columns='key', values='value')

    # textual pivot: X_txt ∈ {string}^{#systems × #unique_keys}
    df_tkv = dfs['text_key_values']
    X_txt = df_tkv.pivot(index='id', columns='key', values='value')

    # composition pivot: X_comp ∈ Z^{#systems × #elements}
    df_sp = dfs['species'].assign(elem=lambda d: d['Z'].map(Z2sym))
    X_comp = df_sp.pivot_table(
        index='id', columns='elem', values='n', fill_value=0
    )

    # join all
    df_full = df_sys.join([X_num, X_txt, X_comp], how='left')
    return df_full

# 4) inspect missingness & save
def inspect_and_save(df_full, out_path):
    print("\nFull feature matrix shape:", df_full.shape)
    miss_frac = df_full.isna().mean().sort_values(ascending=False)
    print("\nTop 10 columns by % missing:\n", miss_frac.head(10))

    out_path.parent.mkdir(exist_ok=True, parents=True)
    df_full.to_parquet(out_path.with_suffix('.parquet'))
    print(f"\nSaved feature matrix to {out_path.with_suffix('.parquet')}")

if __name__ == '__main__':
    db_path = get_db_path()
    print("Using DB:", db_path)

    # map atomic number → element symbol (you’ll want a full 1→H,2→He,… map here)
    Z2sym = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S',
        # … fill in the rest of the 1–118 mapping …
    }

    dfs     = load_tables(db_path)
    df_full = build_feature_df(dfs, Z2sym)
    out     = Path(db_path).parent.parent / 'results' / 'features_full'
    inspect_and_save(df_full, out)
