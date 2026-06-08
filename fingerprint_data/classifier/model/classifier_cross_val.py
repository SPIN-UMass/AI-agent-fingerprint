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
)

# ── configuration ──────────────────────────────────────────────────────────────
TEMPORAL_PATH   = "temporal_features.csv"
HTTP_PATH       = "http_features.csv"
TLS_PATH        = "tls_features.csv"
BEHAVIORAL_PATH = "behavioral_features.csv"

INPUT_DIR  = "/mnt/user-data/uploads"
OUTPUT_DIR = "/mnt/user-data/outputs"

N_ESTIMATORS = 200
RANDOM_SEED  = 42


# ── load & align ───────────────────────────────────────────────────────────────

def load_all() -> tuple[dict[str, pd.DataFrame], LabelEncoder]:
    """
    Load, preprocess, and inner-join all four feature types on (agent, trial).
    Returns a dict of aligned DataFrames and a fitted LabelEncoder.

    Each DataFrame has columns:
        agent | trial | agent_label | <feature columns ...>
    """
    temporal   = preprocess_temporal(os.path.join(INPUT_DIR, TEMPORAL_PATH))
    http       = preprocess_http(os.path.join(INPUT_DIR, HTTP_PATH))
    tls        = preprocess_tls(os.path.join(INPUT_DIR, TLS_PATH))
    behavioral = preprocess_behavioral(os.path.join(INPUT_DIR, BEHAVIORAL_PATH))

    # Inner join aligns rows; drops any trials absent from behavioral
    combined = (
        temporal
        .merge(http,       on=["agent", "trial"], how="inner", suffixes=("", "_http"))
        .merge(tls,        on=["agent", "trial"], how="inner", suffixes=("", "_tls"))
        .merge(behavioral, on=["agent", "trial"], how="inner", suffixes=("", "_beh"))
    )
    common_keys = combined[["agent", "trial"]]

    temporal   = common_keys.merge(temporal,   on=["agent", "trial"], how="left")
    http       = common_keys.merge(http,       on=["agent", "trial"], how="left")
    tls        = common_keys.merge(tls,        on=["agent", "trial"], how="left")
    behavioral = common_keys.merge(behavioral, on=["agent", "trial"], how="left")

    # Encode target on the combined index (shared across all frames)
    le = LabelEncoder()
    labels = pd.Series(
        le.fit_transform(combined["agent"]),
        index=combined.index,
        name="agent_label",
    )

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, labels], axis=1)

    frames = {
        "temporal":   _attach(temporal),
        "http":       _attach(http),
        "tls":        _attach(tls),
        "behavioral": _attach(behavioral),
        "combined":   _attach(combined),
    }

    print(f"\nAligned rows : {len(combined)}")
    print(f"Classes      : {list(le.classes_)}")
    print("Feature counts:")
    for name, df in frames.items():
        n_feat = df.shape[1] - 3   # exclude agent, trial, agent_label
        print(f"  {name:12s}  {n_feat} features")

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
    """
    Leave-one-trial-out cross-validation for each feature type.

    For each of the 30 trial numbers, rows matching that trial (across all
    agents) are held out as the test fold. The model trains on the remaining
    trials. This ensures no trial's data appears in both train and test.

    Returns
    -------
    results : dict
        Per feature type:
            fold_accuracies, fold_precisions, fold_recalls, fold_f1s,
            all_true, all_pred
    """
    ref       = frames["combined"]
    fold_ids  = ref["trial"].map(trial_number)
    folds     = sorted(fold_ids.unique())

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
        test_mask  = fold_ids == fold
        train_mask = ~test_mask

        for name, df in frames.items():
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

def print_and_save_results(
    results: dict[str, dict],
    le: LabelEncoder,
) -> None:
    """
    Print per-fold metrics, per-class classification report, and summary
    table. Also saves cv_results.csv to OUTPUT_DIR.
    """
    summary_rows  = []
    fold_rows     = []

    for name, res in results.items():
        accs  = res["fold_accuracies"]
        precs = res["fold_precisions"]
        recs  = res["fold_recalls"]
        f1s   = res["fold_f1s"]

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
            target_names=le.classes_,
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


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    frames, le = load_all()
    results    = run_cv(frames, le)
    print_and_save_results(results, le)
    train_and_save(frames, le)