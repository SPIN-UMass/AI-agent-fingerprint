"""
classify_extratrees_cv.py
--------------------------
Classifies web agents using ExtraTreesClassifier with leave-one-trial-out
cross-validation. Each of the 30 trials is held out once as the test fold;
all other trials train the model. This prevents leakage from trial-specific
patterns that a random split would allow.

Evaluation is run separately for each feature type:
    temporal, http, tls, behavioral, combined

Outputs
-------
Printed
    - Per-fold accuracy for each feature type
    - Aggregated classification report (precision / recall / F1) across folds
    - Summary table: mean accuracy, precision, recall, F1 per feature type

Saved files  (to OUTPUT_DIR)
    - extratrees_<feature_type>.joblib   trained on ALL data (one per feature type)
    - cv_results.csv                     per-fold and summary metrics

Usage
-----
    python classify_extratrees_cv.py

Inference (after training)
--------------------------
    import joblib, pandas as pd
    from preprocess import preprocess_temporal   # or whichever type

    clf, le, feature_cols = joblib.load("extratrees_temporal.joblib")
    df  = preprocess_temporal("new_temporal.csv")
    X   = df[feature_cols].values
    pred_labels = clf.predict(X)
    pred_agents = le.inverse_transform(pred_labels)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from feature_preprocessing import (
    preprocess_temporal,
    preprocess_http,
    preprocess_tls,
    preprocess_behavioral,
    preprocess_all
)

# ── configuration ──────────────────────────────────────────────────────────────
TEMPORAL_PATH   = "temporal_features.csv"
HTTP_PATH       = "http_features.csv"
TLS_PATH        = "tls_features.csv"
BEHAVIORAL_PATH = "behavioral_features.csv"

INPUT_DIR  = "../features"
OUTPUT_DIR = "./trained_model"

N_ESTIMATORS = 200
RANDOM_SEED  = 42


# ── load & align ───────────────────────────────────────────────────────────────

def load_all() -> tuple[dict[str, pd.DataFrame], LabelEncoder]:

    datasets, le = preprocess_all(
        temporal_path   = os.path.join(INPUT_DIR, TEMPORAL_PATH),
        http_path       = os.path.join(INPUT_DIR, HTTP_PATH),
        tls_path        = os.path.join(INPUT_DIR, TLS_PATH),
        behavioral_path = os.path.join(INPUT_DIR, BEHAVIORAL_PATH),
    )

    # Reconstruct flat DataFrames expected by run_cv and train_and_save:
    #   agent_label | trial | <features...>
    frames = {}
    for name, ds in datasets.items():
        df = ds["X"].copy()
        df.insert(0, "trial", ds["trial"])
        df.insert(0, "agent_label", ds["y"])
        frames[name] = df

    print(f"\nAligned rows : {len(frames['combined'])}")
    print(f"Classes      : {list(le.classes_)}")
    print("Feature counts:")
    for name, df in frames.items():
        n_feat = df.shape[1] - 2  # exclude agent_label, trial
        print(f"  {name:12s}  {n_feat} features  {len(df)} rows")

    return frames, le


# ── trial normalisation ────────────────────────────────────────────────────────

def trial_number(trial_str: str) -> str:
    """
    Normalise trial identifiers to a zero-padded number string so that
    'trial-001' and 'trial_001' both map to '001'.
    Handles both hyphen and underscore separators.
    """
    return trial_str.replace("trial-", "").replace("trial_", "")


# ── cross-validation ───────────────────────────────────────────────────────────

def run_cv(
    frames: dict[str, pd.DataFrame],
    le: LabelEncoder,
) -> dict[str, dict]:

    # Use combined as the fold reference (smallest, 210 rows)
    ref      = frames["combined"]
    folds    = sorted(ref["trial"].map(trial_number).unique())

    print(f"\nCross-validation: {len(folds)} folds (leave-one-trial-out)\n")

    results = {
        name: {
            "fold_accuracies":  [],
            "fold_precisions":  [],
            "fold_recalls":     [],
            "fold_f1s":         [],
            "all_true":         [],
            "all_pred":         [],
        }
        for name in frames
    }

    for fold in folds:
        for name, df in frames.items():
            # Derive mask from each frame's own trial column
            fold_ids   = df["trial"].map(trial_number)
            test_mask  = fold_ids == fold
            train_mask = ~test_mask

            feature_cols = [c for c in df.columns
                            if c not in ("agent", "trial", "agent_label")]

            X_train = df.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, "agent_label"].values
            X_test  = df.loc[test_mask,  feature_cols].values
            y_test  = df.loc[test_mask,  "agent_label"].values

            clf = ExtraTreesClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_SEED,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            results[name]["fold_accuracies"].append(accuracy_score(y_test, preds))
            results[name]["fold_precisions"].append(
                precision_score(y_test, preds, average="macro", zero_division=0))
            results[name]["fold_recalls"].append(
                recall_score(y_test, preds, average="macro", zero_division=0))
            results[name]["fold_f1s"].append(
                f1_score(y_test, preds, average="macro", zero_division=0))
            results[name]["all_true"].extend(y_test.tolist())
            results[name]["all_pred"].extend(preds.tolist())

    return results


# ── train final models on all data ────────────────────────────────────────────

def train_and_save(
    frames: dict[str, pd.DataFrame],
    le: LabelEncoder,
) -> None:
    """
    Train one ExtraTreesClassifier per feature type on ALL available data
    and save each model as a .joblib file.

    Each saved file contains a tuple:
        (fitted_clf, fitted_le, feature_cols)
    so everything needed for inference is bundled together.

    Inference example
    -----------------
        clf, le, feature_cols = joblib.load("extratrees_temporal.joblib")
        X   = new_df[feature_cols].values
        pred_labels = clf.predict(X)
        pred_agents = le.inverse_transform(pred_labels)
    """
    print(f"\n── Saving trained models to {OUTPUT_DIR} ────────────────────")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, df in frames.items():
        feature_cols = [c for c in df.columns
                        if c not in ("agent", "trial", "agent_label")]

        X = df[feature_cols].values
        y = df["agent_label"].values

        clf = ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
        )
        clf.fit(X, y)

        path = os.path.join(OUTPUT_DIR, f"extratrees_{name}.joblib")
        joblib.dump((clf, le, feature_cols), path)
        print(f"  Saved {path}")


# ── reporting ──────────────────────────────────────────────────────────────────
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

def print_and_save_results(
    results: dict[str, dict],
    le: LabelEncoder,
) -> None:
    summary_rows = []
    fold_rows    = []

    for name, res in results.items():
        accs  = res["fold_accuracies"]
        precs = res["fold_precisions"]
        recs  = res["fold_recalls"]
        f1s   = res["fold_f1s"]

        # Derive classes actually present in this feature type's predictions
        present_labels = sorted(set(res["all_true"]) | set(res["all_pred"]))
        present_names  = le.inverse_transform(present_labels)

        print(f"\n{'='*65}")
        print(f"  ExtraTrees  |  {name}")
        print(f"{'='*65}")
        print(f"  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1':>10}")
        print(f"  {'-'*46}")
        for i, (a, p, r, f) in enumerate(zip(accs, precs, recs, f1s), 1):
            print(f"  {i:<6} {a:>10.3f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")
            fold_rows.append({
                "feature_type": name,
                "fold":         i,
                "accuracy":     round(a, 4),
                "precision":    round(p, 4),
                "recall":       round(r, 4),
                "f1":           round(f, 4),
            })

        print(f"\n  Mean  {np.mean(accs):>10.3f} {np.mean(precs):>10.3f} "
              f"{np.mean(recs):>10.3f} {np.mean(f1s):>10.3f}")
        print(f"  Std   {np.std(accs):>10.3f} {np.std(precs):>10.3f} "
              f"{np.std(recs):>10.3f} {np.std(f1s):>10.3f}")

        print(f"\n  Classification report (aggregated across all folds):")
        print(classification_report(
            res["all_true"], res["all_pred"],
            labels=present_labels,       # ← only labels present in this feature type
            target_names=present_names,  # ← matching names for those labels
            zero_division=0,
        ))

        summary_rows.append({
            "feature_type":     name,
            "mean_accuracy":    round(np.mean(accs),  3),
            "std_accuracy":     round(np.std(accs),   3),
            "mean_precision":   round(np.mean(precs), 3),
            "std_precision":    round(np.std(precs),  3),
            "mean_recall":      round(np.mean(recs),  3),
            "std_recall":       round(np.std(recs),   3),
            "mean_f1":          round(np.mean(f1s),   3),
            "std_f1":           round(np.std(f1s),    3),
        })

    # Print summary table
    summary = pd.DataFrame(summary_rows).set_index("feature_type")
    print(f"\n{'='*65}")
    print("  Summary — ExtraTrees leave-one-trial-out CV")
    print(f"{'='*65}")
    print(summary.to_string())

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fold_df    = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    cv_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
    with open(cv_path, "w") as f:
        f.write("# Per-fold metrics\n")
        fold_df.to_csv(f, index=False)
        f.write("\n# Summary metrics\n")
        summary_df.to_csv(f, index=False)

    print(f"\n  CV results saved to {cv_path}")

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    latex = build_latex_tables(results, le)
    print(f"\n{'='*65}")
    print("  LATEX TABLES")
    print(f"{'='*65}")
    print(latex)

    latex_path = os.path.join(OUTPUT_DIR, "cv_results_tables.tex")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX saved to {latex_path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    frames, le = load_all()
    results    = run_cv(frames, le)
    print_and_save_results(results, le)
    train_and_save(frames, le)