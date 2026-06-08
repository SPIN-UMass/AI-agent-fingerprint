"""
preprocess.py
-------------
Preprocesses four feature-type CSVs for tree-based agent classification.
Each feature type can be used independently, and a combined dataset is also
produced. All five datasets share the same train/test split (same indices),
so results are directly comparable.

Public API
----------
    from preprocess import preprocess_all

    datasets, le = preprocess_all()

    # datasets is a dict with keys:
    #   "temporal", "http", "tls", "behavioral", "combined"
    # Each value is a dict:
    #   {
    #     "X_train": pd.DataFrame,
    #     "X_test":  pd.DataFrame,
    #     "y_train": pd.Series,
    #     "y_test":  pd.Series,
    #     "feature_names": list[str],
    #   }
    # le is a fitted LabelEncoder shared across all datasets.

Saved outputs (when run as __main__)
--------------------------------------
    temporal_preprocessed.csv
    http_preprocessed.csv
    tls_preprocessed.csv
    behavioral_preprocessed.csv
    combined_preprocessed.csv

    Each saved file contains all rows (train + test) with columns:
        agent_label  |  feature_0  |  feature_1  |  ...
    The original agent name can be recovered via le.inverse_transform().
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
TEMPORAL_PATH   = "temporal_features.csv"
HTTP_PATH       = "http_features.csv"
TLS_PATH        = "tls_features.csv"
BEHAVIORAL_PATH = "behavioral_features.csv"

TEST_SIZE   = 0.2
RANDOM_SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def _multihot_pipe_column(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Expand a pipe-delimited string column into binary indicator columns.
    NaN / empty rows produce all-zero rows.

    Example:
        "Pragma|X-Forwarded-Proto"
        → {ns_hdr_Pragma: 1, ns_hdr_X-Forwarded-Proto: 1, ...}
    """
    filled = series.fillna("")
    all_values: set = set()
    for cell in filled:
        if cell:
            all_values.update(cell.split("|"))

    records = []
    for cell in filled:
        tokens = set(cell.split("|")) if cell else set()
        records.append({v: int(v in tokens) for v in all_values})

    result = pd.DataFrame(records, index=series.index)
    result.columns = [f"{prefix}_{c}" for c in result.columns]
    return result


def _drop_zero_variance(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    """Drop numeric columns whose values never change (std == 0)."""
    exclude = exclude or []
    numeric = df.select_dtypes(include="number").columns.difference(exclude)
    zero_var = [c for c in numeric if df[c].std() == 0]
    if zero_var:
        print(f"  Dropping zero-variance columns: {zero_var}")
    return df.drop(columns=zero_var)


def _assert_all_numeric(df: pd.DataFrame, name: str) -> None:
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"[{name}] Non-numeric columns remain: {non_numeric}")


# ── per-file preprocessors ────────────────────────────────────────────────────

def preprocess_temporal(path: str = TEMPORAL_PATH) -> pd.DataFrame:
    """
    Returns a clean DataFrame with columns: agent, trial, <features...>
    All numeric; single missing row imputed with column medians.
    """
    print("\n[temporal]")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = _drop_zero_variance(df, exclude=[])
    _assert_all_numeric(df.drop(columns=["agent", "trial"]), "temporal")
    print(f"  Clean shape: {df.shape}  |  features: {df.shape[1] - 2}")
    return df


def preprocess_http(path: str = HTTP_PATH) -> pd.DataFrame:
    """
    Returns a clean DataFrame with columns: agent, trial, <features...>

    Steps
    -----
    - Drop zero-variance columns (nav_order_unique_ratio, has_dnt_rate, has_te_rate)
    - Multi-hot encode nonstandard_names (pipe-delimited header names)
    - Label-encode sf_site_mode, ua_family_mode, ua_os_mode
      (NaN treated as its own category "unknown")
    - Fill priority_correct_rate / priority_urgency_mode with 0
      (missing only when has_priority_rate == 0)
    - Median-impute remaining numeric missings
    """
    print("\n[http]")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # multi-hot encode nonstandard_names before dropping zero-variance
    multihot = _multihot_pipe_column(df["nonstandard_names"], prefix="ns_hdr")
    df = pd.concat([df.drop(columns=["nonstandard_names"]), multihot], axis=1)

    # label-encode categoricals; NaN → "unknown"
    for col in ["sf_site_mode", "ua_family_mode", "ua_os_mode"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("unknown"))

    # priority missings are structurally zero
    df["priority_correct_rate"] = df["priority_correct_rate"].fillna(0.0)
    df["priority_urgency_mode"] = df["priority_urgency_mode"].fillna(0.0)

    # median-impute any remaining numeric missings
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = _drop_zero_variance(df, exclude=[])
    _assert_all_numeric(df.drop(columns=["agent", "trial"]), "http")
    print(f"  Clean shape: {df.shape}  |  features: {df.shape[1] - 2}")
    return df


def preprocess_tls(path: str = TLS_PATH) -> pd.DataFrame:
    """
    Returns a clean DataFrame with columns: agent, trial, <features...>
    All numeric; median-impute any missings.
    """
    print("\n[tls]")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = _drop_zero_variance(df, exclude=[])
    _assert_all_numeric(df.drop(columns=["agent", "trial"]), "tls")
    print(f"  Clean shape: {df.shape}  |  features: {df.shape[1] - 2}")
    return df


def preprocess_behavioral(path: str = BEHAVIORAL_PATH) -> pd.DataFrame:
    """
    Returns a clean DataFrame with columns: agent, trial, <features...>

    Steps
    -----
    - Drop duplicate agent.1 column
    - Median-impute missings (structural: e.g. click stats absent when no clicks)
    """
    print("\n[behavioral]")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    df = df.drop(columns=["agent.1"], errors="ignore")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = _drop_zero_variance(df, exclude=[])
    _assert_all_numeric(df.drop(columns=["agent", "trial"]), "behavioral")
    print(f"  Clean shape: {df.shape}  |  features: {df.shape[1] - 2}")
    return df


# ── merge & split ─────────────────────────────────────────────────────────────

def preprocess_all(
    temporal_path:   str   = TEMPORAL_PATH,
    http_path:       str   = HTTP_PATH,
    tls_path:        str   = TLS_PATH,
    behavioral_path: str   = BEHAVIORAL_PATH,
    test_size:       float = TEST_SIZE,
    random_seed:     int   = RANDOM_SEED,
) -> tuple[dict, LabelEncoder]:
    """
    Preprocess all four feature sets and produce train/test splits.
    All five datasets (4 individual + 1 combined) share the same row indices
    so classifiers trained on each are directly comparable.

    Returns
    -------
    datasets : dict
        Keys: "temporal", "http", "tls", "behavioral", "combined"
        Each value is a dict with:
            X_train, X_test  : pd.DataFrame
            y_train, y_test  : pd.Series  (integer-encoded agent labels)
            feature_names    : list[str]
    le : LabelEncoder
        Fitted on agent names; use le.inverse_transform() to decode predictions.
    """
    # ── load each feature type ────────────────────────────────────────────────
    temporal   = preprocess_temporal(temporal_path)
    http       = preprocess_http(http_path)
    tls        = preprocess_tls(tls_path)
    behavioral = preprocess_behavioral(behavioral_path)

    # ── inner join on (agent, trial) ──────────────────────────────────────────
    # behavioral is missing 2 operator trials; inner join drops those rows
    # so all feature sets are aligned to the same 178 rows.
    print("\n[combining]")
    combined = (
        temporal
        .merge(http,       on=["agent", "trial"], how="inner", suffixes=("", "_http"))
        .merge(tls,        on=["agent", "trial"], how="inner", suffixes=("", "_tls"))
        .merge(behavioral, on=["agent", "trial"], how="inner", suffixes=("", "_beh"))
    )
    common_index = combined[["agent", "trial"]]  # 178 aligned rows
    print(f"  Aligned rows: {len(combined)}  ({combined['agent'].nunique()} agents)")

    # Trim each individual frame to the same 178 rows
    temporal   = common_index.merge(temporal,   on=["agent", "trial"], how="left")
    http       = common_index.merge(http,       on=["agent", "trial"], how="left")
    tls        = common_index.merge(tls,        on=["agent", "trial"], how="left")
    behavioral = common_index.merge(behavioral, on=["agent", "trial"], how="left")

    # ── encode target ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = pd.Series(
        le.fit_transform(combined["agent"]),
        index=combined.index,
        name="agent_label",
    )
    print(f"  Classes: {list(le.classes_)}")

    # ── shared stratified split ───────────────────────────────────────────────
    train_idx, test_idx = train_test_split(
        combined.index,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )
    print(f"  Train: {len(train_idx)} rows  |  Test: {len(test_idx)} rows")

    # ── build per-type datasets ───────────────────────────────────────────────
    def _make_dataset(df: pd.DataFrame, name: str) -> dict:
        X = df.drop(columns=["agent", "trial"], errors="ignore")
        return {
            "X_train":       X.loc[train_idx].reset_index(drop=True),
            "X_test":        X.loc[test_idx].reset_index(drop=True),
            "y_train":       y.loc[train_idx].reset_index(drop=True),
            "y_test":        y.loc[test_idx].reset_index(drop=True),
            "feature_names": X.columns.tolist(),
        }

    datasets = {
        "temporal":   _make_dataset(temporal,   "temporal"),
        "http":       _make_dataset(http,        "http"),
        "tls":        _make_dataset(tls,         "tls"),
        "behavioral": _make_dataset(behavioral,  "behavioral"),
        "combined":   _make_dataset(combined,    "combined"),
    }

    print("\n── Feature counts ──────────────────────────────────────")
    for name, ds in datasets.items():
        print(f"  {name:12s}  {len(ds['feature_names'])} features")

    return datasets, le


# ── save to CSV ───────────────────────────────────────────────────────────────

def save_preprocessed(
    datasets: dict,
    output_dir: str = ".",
) -> None:
    """
    Save each dataset (train + test combined) as a CSV.
    Columns: agent_label, feature_0, feature_1, ...
    Rows are sorted by their original index (train rows first, then test).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, ds in datasets.items():
        for split in ["train", "test"]:
            X = ds[f"X_{split}"]
            y = ds[f"y_{split}"]
            out = pd.concat([y.rename("agent_label"), X], axis=1)
            path = os.path.join(output_dir, f"{name}_{split}.csv")
            out.to_csv(path, index=False)
            print(f"  Saved {path}  {out.shape}")


# ── smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    INPUT_DIR  = "../features/"
    OUTPUT_DIR = "./model_input"

    datasets, le = preprocess_all(
        temporal_path   = os.path.join(INPUT_DIR, TEMPORAL_PATH),
        http_path       = os.path.join(INPUT_DIR, HTTP_PATH),
        tls_path        = os.path.join(INPUT_DIR, TLS_PATH),
        behavioral_path = os.path.join(INPUT_DIR, BEHAVIORAL_PATH),
    )

    print("\n── Class distribution (train) ──────────────────────────")
    counts = datasets["combined"]["y_train"].value_counts().sort_index()
    for idx, cnt in counts.items():
        print(f"  {le.classes_[idx]:25s}  {cnt}")

    print("\n── NaN check ───────────────────────────────────────────")
    for name, ds in datasets.items():
        nan_train = ds["X_train"].isnull().any().any()
        nan_test  = ds["X_test"].isnull().any().any()
        print(f"  {name:12s}  NaN in train={nan_train}  NaN in test={nan_test}")

    print("\n── Saving CSVs ─────────────────────────────────────────")
    save_preprocessed(datasets, output_dir=OUTPUT_DIR)