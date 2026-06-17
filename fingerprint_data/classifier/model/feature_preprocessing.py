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
) -> tuple[dict, LabelEncoder]:
    """
    Preprocess all four feature sets and return aligned DataFrames.
    No train/test split is performed here — the classifier handles that
    via leave-one-trial-out cross-validation.

    Returns
    -------
    datasets : dict
        Keys: "temporal", "http", "tls", "behavioral", "combined"
        Each value is a dict with:
            X            : pd.DataFrame  (features only, no agent/trial/label)
            y            : pd.Series     (integer-encoded agent labels)
            trial        : pd.Series     (trial identifiers, for CV fold assignment)
            feature_names: list[str]
    le : LabelEncoder
        Fitted on agent names; use le.inverse_transform() to decode predictions.
    """
    temporal   = preprocess_temporal(temporal_path)
    http       = preprocess_http(http_path)
    tls        = preprocess_tls(tls_path)
    behavioral = preprocess_behavioral(behavioral_path)

    print("\n[combining]")

    # ── Base index: rows present in all three complete sets ────────────────────
    base_index = (
        temporal[["agent", "trial"]]
        .merge(http[["agent", "trial"]], on=["agent", "trial"], how="inner")
        .merge(tls[["agent", "trial"]],  on=["agent", "trial"], how="inner")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"  Base rows (temporal∩http∩tls): {len(base_index)}  "
          f"({base_index['agent'].nunique()} agents)")

    # Re-align the three complete sets to base_index
    temporal = base_index.merge(temporal, on=["agent", "trial"], how="left")
    http     = base_index.merge(http,     on=["agent", "trial"], how="left")
    tls      = base_index.merge(tls,      on=["agent", "trial"], how="left")

    # ── Behavioral + combined: inner join — drop rows missing behavioral ────────
    beh_index = base_index.merge(
        behavioral[["agent", "trial"]],
        on=["agent", "trial"],
        how="inner",
    ).drop_duplicates().reset_index(drop=True)

    behavioral_aligned = beh_index.merge(behavioral, on=["agent", "trial"], how="left")

    combined = (
        temporal
        .merge(http,               on=["agent", "trial"], how="left", suffixes=("", "_http"))
        .merge(tls,                on=["agent", "trial"], how="left", suffixes=("", "_tls"))
        .merge(behavioral_aligned, on=["agent", "trial"], how="inner", suffixes=("", "_beh"))
    )

    n_dropped = len(base_index) - len(beh_index)
    print(f"  Behavioral/combined rows: {len(beh_index)}  "
          f"({n_dropped} dropped — missing behavioral data)")
    print(f"  Combined columns: {combined.shape[1]}")

    # ── Encode target ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(base_index["agent"])
    print(f"  Classes: {list(le.classes_)}")

    # ── Build datasets — no splitting ──────────────────────────────────────────
    def _make_dataset(df: pd.DataFrame) -> dict:
        feature_cols = [c for c in df.columns
                        if c not in ("agent", "trial", "agent_label")]
        return {
            "X":             df[feature_cols].reset_index(drop=True),
            "y":             pd.Series(
                                 le.transform(df["agent"]),
                                 name="agent_label",
                             ).reset_index(drop=True),
            "trial":         df["trial"].reset_index(drop=True),
            "feature_names": feature_cols,
        }

    datasets = {
        "temporal":   _make_dataset(temporal),
        "http":       _make_dataset(http),
        "tls":        _make_dataset(tls),
        "behavioral": _make_dataset(behavioral_aligned),
        "combined":   _make_dataset(combined),
    }

    print("\n── Feature counts ──────────────────────────────────────")
    for name, ds in datasets.items():
        print(f"  {name:12s}  {len(ds['feature_names'])} features  "
              f"{len(ds['X'])} rows")

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
        X = ds[f"X"]
        y = ds[f"y"]
        out = pd.concat([y.rename("agent_label"), X], axis=1)
        path = os.path.join(output_dir, f"{name}.csv")
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
    counts = datasets["combined"]["y"].value_counts().sort_index()
    for idx, cnt in counts.items():
        print(f"  {le.classes_[idx]:25s}  {cnt}")

    print("\n── NaN check ───────────────────────────────────────────")
    for name, ds in datasets.items():
        nan_train = ds["X"].isnull().any().any()
        nan_test  = ds["X"].isnull().any().any()
        print(f"  {name:12s}  NaN in train={nan_train}  NaN in test={nan_test}")

    print("\n── Saving CSVs ─────────────────────────────────────────")
    save_preprocessed(datasets, output_dir=OUTPUT_DIR)