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
from feature_preprocessing import (
    preprocess_temporal,
    preprocess_http,
    preprocess_tls,
    preprocess_behavioral,
    preprocess_all
)

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = "../features"
MODEL_DIR  = "../features/trained_model"
OUTPUT_DIR = "./eval_result"

TEMPORAL_PATH   = "temporal_features.csv"
HTTP_PATH       = "http_features.csv"
TLS_PATH        = "tls_features.csv"
BEHAVIORAL_PATH = "behavioral_features.csv"

FEATURE_TYPES = ["temporal", "http", "tls", "behavioral", "combined"]


# ── load data ──────────────────────────────────────────────────────────────────

def load_aligned_data() -> dict[str, pd.DataFrame]:
    """
    Load and preprocess all feature types.
    - temporal, http, tls: all rows (300)
    - behavioral, combined: only rows present in behavioral (210)
    Each DataFrame has columns: agent | trial | <features...>
    """
    datasets, le = preprocess_all(
        temporal_path   = os.path.join(INPUT_DIR, TEMPORAL_PATH),
        http_path       = os.path.join(INPUT_DIR, HTTP_PATH),
        tls_path        = os.path.join(INPUT_DIR, TLS_PATH),
        behavioral_path = os.path.join(INPUT_DIR, BEHAVIORAL_PATH),
    )

    # Reconstruct flat DataFrames expected by run_inference:
    #   agent | trial | <features...>
    data = {}
    for name, ds in datasets.items():
        df = ds["X"].copy()
        df.insert(0, "trial", ds["trial"])
        df.insert(0, "agent", le.inverse_transform(ds["y"]))  # agent name string
        data[name] = df

    return data


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
        y_pred = le.inverse_transform(pred_labels).tolist()

        present_names = sorted(set(y_true) | set(y_pred))  # only classes in this feature type

        results[feat] = {
            "y_true":      y_true,
            "y_pred":      y_pred,
            "agent_names": present_names,  # ← 7 for behavioral/combined, 10 for others
        }

    return results


# ── console reporting ──────────────────────────────────────────────────────────

def print_results(results: dict[str, dict]) -> None:

    # ── per feature type ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION BY FEATURE TYPE")
    print("="*65)

    for feat in FEATURE_TYPES:
        res    = results[feat]
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        acc    = accuracy_score(y_true, y_pred)

        # Use only classes present in this feature type
        present_names = res["agent_names"]

        print(f"\n── {feat.upper()} (accuracy={acc:.3f}) {'─'*(45-len(feat))}")
        print(classification_report(
            y_true, y_pred,
            target_names=present_names,  # ← per-feature-type classes
            labels=present_names,        # ← match labels to target_names
            zero_division=0,
            digits=3,
        ))

    # ── per agent (across all feature types) ─────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION BY AGENT TYPE  (aggregated across feature types)")
    print("="*65)

    # ── per agent (across all feature types) ─────────────────────────────────
    all_agent_names = sorted(set(
        name for res in results.values() for name in res["agent_names"]
    ))

    for agent in all_agent_names:
        print(f"\n── {agent.upper()} {'─'*(55-len(agent))}")
        print(f"  {'Feature type':<14} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'Support':>10}")
        print(f"  {'-'*50}")

        for feat in FEATURE_TYPES:
            res           = results[feat]
            present_names = res["agent_names"]

            # Agent not in this feature type → mark as missing
            if agent not in present_names:
                print(f"  {feat:<14} {'--':>10} {'--':>10} {'--':>10} {'--':>10}")
                continue

            y_true = res["y_true"]
            y_pred = res["y_pred"]

            yt_bin  = [1 if a == agent else 0 for a in y_true]
            yp_bin  = [1 if a == agent else 0 for a in y_pred]
            support = sum(yt_bin)
            tp = sum(t == 1 and p == 1 for t, p in zip(yt_bin, yp_bin))
            fp = sum(t == 0 and p == 1 for t, p in zip(yt_bin, yp_bin))
            fn = sum(t == 1 and p == 0 for t, p in zip(yt_bin, yp_bin))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            print(f"  {feat:<14} {prec:>10.3f} {rec:>10.3f} "
                  f"{f1:>10.3f} {support:>10}")


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    """Format a float to 3 decimal places for LaTeX."""
    return f"{val:.3f}"


def build_latex_tables(
    results: dict[str, dict],
    le: LabelEncoder,
) -> str:
    """
    Build two LaTeX tables from leave-one-trial-out CV results:
        Table 1 — per-feature-type summary (accuracy, macro P/R/F1)
        Table 2 — per-agent F1 scores across feature types
    """
    def _fmt(val: float) -> str:
        return f"{val:.3f}"

    # Global agent list — union across all feature types
    all_agent_names = sorted(set(
        le.inverse_transform([label])[ 0]
        for res in results.values()
        for label in set(res["all_true"]) | set(res["all_pred"])
    ))
    agent_labels = {a: a.replace("_", r"\_") for a in all_agent_names}
    feat_labels  = {f: f.capitalize() for f in results.keys()}

    lines = []

    # ── Table 1: per-feature-type summary ────────────────────────────────────
    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{ExtraTrees leave-one-trial-out CV results by feature type.}",
        r"\label{tab:feature_type_results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Feature Type & Accuracy & Precision & Recall & F1 \\",
        r"\midrule",
    ]

    for feat, res in results.items():
        # CV results store integer labels — decode to strings
        y_true        = le.inverse_transform(res["all_true"]).tolist()
        y_pred        = le.inverse_transform(res["all_pred"]).tolist()
        present_names = sorted(set(y_true) | set(y_pred))

        report = classification_report(
            y_true, y_pred,
            target_names=present_names,
            labels=present_names,
            output_dict=True,
            zero_division=0,
        )
        acc  = _fmt(accuracy_score(y_true, y_pred))
        prec = _fmt(report["macro avg"]["precision"])
        rec  = _fmt(report["macro avg"]["recall"])
        f1   = _fmt(report["macro avg"]["f1-score"])
        lines.append(
            f"{feat_labels[feat]} & {acc} & {prec} & {rec} & {f1} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ── Table 2: per-agent F1 across feature types ────────────────────────────
    feat_cols = " & ".join(feat_labels[f] for f in results.keys())
    col_spec  = "l" + "r" * len(results)

    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-agent F1 score by feature type "
        r"(leave-one-trial-out CV; -- indicates agent absent from that feature set).}",
        r"\label{tab:agent_f1_results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Agent & {feat_cols} \\\\",
        r"\midrule",
    ]

    for agent in all_agent_names:
        row_vals = []
        for feat, res in results.items():
            y_true        = le.inverse_transform(res["all_true"]).tolist()
            y_pred        = le.inverse_transform(res["all_pred"]).tolist()
            present_names = sorted(set(y_true) | set(y_pred))

            if agent not in present_names:
                row_vals.append("--")
                continue

            report = classification_report(
                y_true, y_pred,
                target_names=present_names,
                labels=present_names,
                output_dict=True,
                zero_division=0,
            )
            row_vals.append(_fmt(report[agent]["f1-score"]))

        lines.append(f"{agent_labels[agent]} & {' & '.join(row_vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
    # latex_path = os.path.join(OUTPUT_DIR, "results_tables.tex")
    # with open(latex_path, "w") as f:
    #     f.write(latex)
    # print(f"\nLaTeX saved to {latex_path}")