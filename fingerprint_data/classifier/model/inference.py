"""
inference.py
------------
Loads the five saved ExtraTrees models, runs inference on the full dataset,
and prints evaluation results as:
    1. Console classification report (per feature type and per agent)
    2. LaTeX tables (per feature type summary + per-agent breakdown)

Usage
-----
    python inference.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from preprocess import (
    preprocess_temporal,
    preprocess_http,
    preprocess_tls,
    preprocess_behavioral,
)

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = "/mnt/user-data/uploads"
MODEL_DIR  = "/mnt/user-data/outputs"
OUTPUT_DIR = "/mnt/user-data/outputs"

TEMPORAL_PATH   = "temporal_features.csv"
HTTP_PATH       = "http_features.csv"
TLS_PATH        = "tls_features.csv"
BEHAVIORAL_PATH = "behavioral_features.csv"

FEATURE_TYPES = ["temporal", "http", "tls", "behavioral", "combined"]


# ── load data ──────────────────────────────────────────────────────────────────

def load_aligned_data() -> dict[str, pd.DataFrame]:
    """
    Load and preprocess all feature types, inner-join to align rows,
    and return a dict of DataFrames keyed by feature type.
    Each DataFrame has columns: agent | trial | <features...>
    """
    temporal   = preprocess_temporal(os.path.join(INPUT_DIR, TEMPORAL_PATH))
    http       = preprocess_http(os.path.join(INPUT_DIR, HTTP_PATH))
    tls        = preprocess_tls(os.path.join(INPUT_DIR, TLS_PATH))
    behavioral = preprocess_behavioral(os.path.join(INPUT_DIR, BEHAVIORAL_PATH))

    combined = (
        temporal
        .merge(http,       on=["agent", "trial"], how="inner", suffixes=("", "_http"))
        .merge(tls,        on=["agent", "trial"], how="inner", suffixes=("", "_tls"))
        .merge(behavioral, on=["agent", "trial"], how="inner", suffixes=("", "_beh"))
    )
    common_keys = combined[["agent", "trial"]]

    return {
        "temporal":   common_keys.merge(temporal,   on=["agent", "trial"], how="left"),
        "http":       common_keys.merge(http,        on=["agent", "trial"], how="left"),
        "tls":        common_keys.merge(tls,         on=["agent", "trial"], how="left"),
        "behavioral": common_keys.merge(behavioral,  on=["agent", "trial"], how="left"),
        "combined":   combined,
    }


# ── inference ──────────────────────────────────────────────────────────────────

def run_inference(
    data: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    For each feature type, load the saved model and run inference on
    all aligned rows.

    Returns
    -------
    results : dict
        Keys: feature type names
        Values: dict with keys
            "y_true"      : list of agent name strings
            "y_pred"      : list of predicted agent name strings
            "agent_names" : list of class names (from LabelEncoder)
    """
    results = {}

    for feat in FEATURE_TYPES:
        model_path = os.path.join(MODEL_DIR, f"extratrees_{feat}.joblib")
        clf, le, feature_cols = joblib.load(model_path)

        df     = data[feat]
        X      = df[feature_cols].values
        y_true = df["agent"].tolist()

        pred_labels = clf.predict(X)
        y_pred      = le.inverse_transform(pred_labels).tolist()

        results[feat] = {
            "y_true":      y_true,
            "y_pred":      y_pred,
            "agent_names": list(le.classes_),
        }

    return results


# ── console reporting ──────────────────────────────────────────────────────────

def print_results(results: dict[str, dict]) -> None:
    """Print per-feature-type and per-agent evaluation to the console."""

    agent_names = results["temporal"]["agent_names"]

    # ── per feature type ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION BY FEATURE TYPE")
    print("="*65)

    for feat in FEATURE_TYPES:
        res     = results[feat]
        y_true  = res["y_true"]
        y_pred  = res["y_pred"]
        acc     = accuracy_score(y_true, y_pred)

        print(f"\n── {feat.upper()} (accuracy={acc:.3f}) {'─'*(45-len(feat))}")
        print(classification_report(
            y_true, y_pred,
            target_names=agent_names,
            zero_division=0,
            digits=3,
        ))

    # ── per agent (across all feature types) ─────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION BY AGENT TYPE  (aggregated across feature types)")
    print("="*65)

    for agent in agent_names:
        print(f"\n── {agent.upper()} {'─'*(55-len(agent))}")
        print(f"  {'Feature type':<14} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'Support':>10}")
        print(f"  {'-'*50}")

        for feat in FEATURE_TYPES:
            res    = results[feat]
            y_true = res["y_true"]
            y_pred = res["y_pred"]

            # Binary classification view for this agent
            yt_bin = [1 if a == agent else 0 for a in y_true]
            yp_bin = [1 if a == agent else 0 for a in y_pred]

            support = sum(yt_bin)
            tp  = sum(t == 1 and p == 1 for t, p in zip(yt_bin, yp_bin))
            fp  = sum(t == 0 and p == 1 for t, p in zip(yt_bin, yp_bin))
            fn  = sum(t == 1 and p == 0 for t, p in zip(yt_bin, yp_bin))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            print(f"  {feat:<14} {prec:>10.3f} {rec:>10.3f} "
                  f"{f1:>10.3f} {support:>10}")


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    """Format a float to 3 decimal places for LaTeX."""
    return f"{val:.3f}"


def build_latex_tables(results: dict[str, dict]) -> str:
    """
    Build two LaTeX tables:
        Table 1 — per-feature-type summary (accuracy, macro P/R/F1)
        Table 2 — per-agent F1 scores across feature types
    Returns the full LaTeX string.
    """
    agent_names = results["temporal"]["agent_names"]
    # Readable agent labels for LaTeX
    agent_labels = {a: a.replace("_", r"\_") for a in agent_names}
    feat_labels  = {f: f.capitalize() for f in FEATURE_TYPES}

    lines = []

    # ── Table 1: per-feature-type summary ────────────────────────────────────
    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{ExtraTrees classification results by feature type "
        r"(inference on full dataset).}",
        r"\label{tab:feature_type_results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Feature Type & Accuracy & Precision & Recall & F1 \\",
        r"\midrule",
    ]

    for feat in FEATURE_TYPES:
        res    = results[feat]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        report = classification_report(
            y_true, y_pred,
            target_names=agent_names,
            output_dict=True,
            zero_division=0,
        )
        acc   = _fmt(accuracy_score(y_true, y_pred))
        prec  = _fmt(report["macro avg"]["precision"])
        rec   = _fmt(report["macro avg"]["recall"])
        f1    = _fmt(report["macro avg"]["f1-score"])
        label = feat_labels[feat]

        lines.append(f"{label} & {acc} & {prec} & {rec} & {f1} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table 2: per-agent F1 across feature types ────────────────────────────
    feat_cols = " & ".join(feat_labels[f] for f in FEATURE_TYPES)
    col_spec  = "l" + "r" * len(FEATURE_TYPES)

    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-agent F1 score by feature type "
        r"(ExtraTrees, inference on full dataset).}",
        r"\label{tab:agent_f1_results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Agent & {feat_cols} \\\\",
        r"\midrule",
    ]

    for agent in agent_names:
        row_vals = []
        for feat in FEATURE_TYPES:
            res    = results[feat]
            y_true = res["y_true"]
            y_pred = res["y_pred"]

            report = classification_report(
                y_true, y_pred,
                target_names=agent_names,
                output_dict=True,
                zero_division=0,
            )
            f1 = report[agent]["f1-score"]
            row_vals.append(_fmt(f1))

        label = agent_labels[agent]
        lines.append(f"{label} & {' & '.join(row_vals)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data = load_aligned_data()

    print("Running inference...")
    results = run_inference(data)

    # Console output
    print_results(results)

    # LaTeX tables
    latex = build_latex_tables(results)
    print("\n" + "="*65)
    print("  LATEX TABLES")
    print("="*65)
    print(latex)

    # Save LaTeX to file
    latex_path = os.path.join(OUTPUT_DIR, "results_tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX saved to {latex_path}")