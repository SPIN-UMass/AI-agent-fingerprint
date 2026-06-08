"""
evaluate_trials.py
------------------
Runs inference across all 180 trials (30 × 6 agents) using the saved
ExtraTrees models and plots how accuracy and F1 improve as more HTTP
requests are seen within each trial.

Directory structure expected
----------------------------
  <data-dir>/
    autogen_websurfer/
      trial-001/
        requests.jsonl
        interactions.jsonl
      trial-002/
        ...
    browser_use/
      trial-001/
        ...
    claude_computer_use/  gemini_computer_use/  operator/  skyvern/

For each trial, inference is run at each cumulative request count
(1 request seen, 2 requests seen, …, all requests seen). At each
checkpoint the five models predict the agent; the prediction is compared
to the ground truth agent label.

Metrics computed
----------------
  - Accuracy        : fraction of trials correctly predicted at each N
  - Macro F1        : averaged over all 6 agent classes
  - Per-agent F1    : one curve per agent
  - Per-model curves: separate line per feature type (combined shown by default)

Output files
------------
  results_per_checkpoint.csv   raw per-trial, per-request-count predictions
  metrics_over_requests.csv    accuracy + F1 per request count (all models)
  accuracy_f1_curve.png        accuracy and macro-F1 vs requests seen
  per_agent_f1_curve.png       per-agent F1 vs requests seen (combined model)
  per_model_accuracy_curve.png per-model accuracy vs requests seen

Usage
-----
  python evaluate_trials.py \\
      --data-dir   ./data \\
      --model-dir  ./models \\
      --extractor-dir ./extractors \\
      --output-dir ./eval_results \\
      --max-requests 30        # truncate x-axis (optional)
      --n-jobs 1               # parallel trials (optional)
"""

import sys
import os
import re
import json
import argparse
import warnings
import tempfile
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
FEATURE_TYPES = ["temporal", "http", "tls", "behavioral", "combined"]
AGENTS = [
    "autogen_websurfer", "browser_use", "claude_computer_use",
    "gemini_computer_use", "operator", "skyvern",
]
AGENT_SHORT = {
    "autogen_websurfer":   "autogen",
    "browser_use":         "browser_use",
    "claude_computer_use": "claude",
    "gemini_computer_use": "gemini",
    "operator":            "operator",
    "skyvern":             "skyvern",
}
AGENT_COLORS = {
    "autogen_websurfer":   "#4C72B0",
    "browser_use":         "#DD8452",
    "claude_computer_use": "#55A868",
    "gemini_computer_use": "#C44E52",
    "operator":            "#8172B3",
    "skyvern":             "#937860",
}
MODEL_COLORS = {
    "temporal":   "#4C72B0",
    "http":       "#DD8452",
    "tls":        "#55A868",
    "behavioral": "#C44E52",
    "combined":   "#000000",
}

# Label-encoder maps (must match preprocess.py)
SF_SITE_LE   = {"none": 0, "same-origin": 1, "unknown": 2}
UA_FAMILY_LE = {
    "chrome": 0, "curl": 1, "edge": 2, "firefox": 3, "go": 4,
    "headless-chrome": 5, "okhttp": 6, "other": 7, "python": 8, "safari": 9,
}
UA_OS_LE = {
    "android": 0, "ios": 1, "linux": 2, "macos": 3, "other": 4, "windows": 5,
}
NS_HDR_COLS = [
    "ns_hdr_X-Forwarded-Proto", "ns_hdr_Cf-Ray",    "ns_hdr_From",
    "ns_hdr_X-Request-Id",      "ns_hdr_Signature-Input",
    "ns_hdr_X-Requested-With",  "ns_hdr_Keep-Alive", "ns_hdr_Accept-Charset",
    "ns_hdr_Cdn-Loop",          "ns_hdr_Pragma",
    "ns_hdr_X-Envoy-Expected-Rq-Timeout-Ms", "ns_hdr_Signature-Agent",
    "ns_hdr_Cf-Ew-Via",         "ns_hdr_Cf-Worker",  "ns_hdr_Signature",
    "ns_hdr_Amp-Cache-Transform","ns_hdr_Sec-Gpc",   "ns_hdr_Cf-Visitor",
]


# ── timestamp helper ───────────────────────────────────────────────────────────

def parse_ts(ts: str):
    from datetime import datetime
    ts = re.sub(r"(\.\d{6})\d+", r"\1", ts).replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


# ── discover trials ────────────────────────────────────────────────────────────

def discover_trials(data_dir: Path) -> list[dict]:
    """
    Walk data_dir and return a list of dicts:
        {agent, trial, requests_path, interactions_path}
    Expects structure: data_dir/<agent>/<trial>/requests.jsonl
    """
    trials = []
    for agent_dir in sorted(data_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        if agent not in AGENTS:
            continue
        for trial_dir in sorted(agent_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            req_path = trial_dir / "requests.jsonl"
            int_path = trial_dir / "interactions.jsonl"
            if req_path.exists() and int_path.exists():
                trials.append({
                    "agent":             agent,
                    "trial":             trial_dir.name,
                    "requests_path":     req_path,
                    "interactions_path": int_path,
                })
    return trials


# ── slice helpers ──────────────────────────────────────────────────────────────

def load_sorted_requests(requests_path: Path) -> list[dict]:
    """Load all request records sorted by timestamp."""
    records = []
    with open(requests_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r.get("timestamp", ""))
    return records


def write_tmp_requests(records: list[dict]) -> str:
    """Write records to a temp JSONL file; return path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="ev_req_")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


def write_tmp_interactions(interactions_path: Path,
                           cutoff_ts) -> str:
    """
    Write interaction events whose timestamp ≤ cutoff_ts to a temp file.
    Preserves the per-page batch structure the extractors expect.
    """
    from collections import defaultdict
    batch_events: dict[str, list] = defaultdict(list)
    batch_meta:   dict[str, dict] = {}

    with open(interactions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("page", "") + "|" + rec.get("session", "")
            batch_meta[key] = rec
            for ev in rec.get("batch", []):
                if "t" not in ev:
                    continue
                try:
                    ev_ts = parse_ts(ev["t"])
                except Exception:
                    continue
                if ev_ts <= cutoff_ts:
                    batch_events[key].append(ev)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="ev_int_")
    for key, ev_list in batch_events.items():
        if not ev_list:
            continue
        base = {k: v for k, v in batch_meta[key].items() if k != "batch"}
        base["batch"]      = ev_list
        base["eventCount"] = len(ev_list)
        tmp.write(json.dumps(base) + "\n")
    tmp.close()
    return tmp.name


# ── feature extraction ─────────────────────────────────────────────────────────

def extract_all(req_path: str, int_path: str) -> dict[str, dict]:
    import extract_request_feature    as req_ext
    import extract_http_feature       as http_ext
    import extract_tls_feature        as tls_ext
    import extract_behavioral_feature as beh_ext

    rp  = Path(req_path)
    ip  = Path(int_path)
    raw = {}

    try:
        df = req_ext.load_requests(rp)
        f  = req_ext.extract_features(df)
        f["req_rate_hz_median"] = f.get("req_rate_hz", 0.0)
        f["req_rate_hz_cv"]     = 0.0
        raw["temporal"] = f
    except Exception:
        raw["temporal"] = {}

    try:
        df = http_ext.load_trial(rp, agent="rt", trial="rt", do_filter=True)
        f  = http_ext.build_trial_features(df).to_dict("records")[0]
        raw["http"] = f
    except Exception:
        raw["http"] = {}

    try:
        df = tls_ext.load_trial("rt", "rt", rp, do_filter=True)
        f  = (tls_ext.aggregate_trial(df).to_dict()
              if df is not None and not df.empty else {})
        raw["tls"] = f
    except Exception:
        raw["tls"] = {}

    try:
        flushes, evs = beh_ext.load_jsonl(ip)
        raw["behavioral"] = beh_ext.extract_features(flushes, evs)
    except Exception:
        raw["behavioral"] = {}

    return raw


# ── feature alignment ──────────────────────────────────────────────────────────

def _encode_http(raw_http: dict, feature_cols: list) -> dict:
    row = {}
    for col in feature_cols:
        if col == "sf_site_mode":
            row[col] = SF_SITE_LE.get(
                str(raw_http.get("sf_site_mode") or "unknown"), 2)
        elif col == "ua_family_mode":
            row[col] = UA_FAMILY_LE.get(
                str(raw_http.get("ua_family_mode") or "other"), 7)
        elif col == "ua_os_mode":
            row[col] = UA_OS_LE.get(
                str(raw_http.get("ua_os_mode") or "other"), 4)
        elif col == "ua_version_mode":
            try:
                row[col] = int(raw_http.get("ua_version_mode"))
            except (TypeError, ValueError):
                row[col] = np.nan
        elif col in ("priority_correct_rate", "priority_urgency_mode"):
            row[col] = raw_http.get(col) or 0.0
        elif col.startswith("ns_hdr_"):
            ns_raw  = str(raw_http.get("nonstandard_names") or "")
            ns_seen = set(ns_raw.split("|")) if ns_raw else set()
            row[col] = 1 if col[len("ns_hdr_"):] in ns_seen else 0
        else:
            row[col] = raw_http.get(col, np.nan)
    return row


def align_features(raw: dict, feat_type: str,
                   feature_cols: list) -> pd.DataFrame:
    if feat_type == "temporal":
        row = {c: raw["temporal"].get(c, np.nan) for c in feature_cols}
    elif feat_type == "http":
        row = _encode_http(raw["http"], feature_cols)
    elif feat_type == "tls":
        row = {c: raw["tls"].get(c, np.nan) for c in feature_cols}
    elif feat_type == "behavioral":
        row = {c: raw["behavioral"].get(c, np.nan) for c in feature_cols}
    else:   # combined
        merged = {}
        merged.update(raw["temporal"])
        merged.update(raw["tls"])
        merged.update(raw["behavioral"])
        http_enc = _encode_http(raw["http"], feature_cols)
        for col in feature_cols:
            if col not in merged:
                merged[col] = http_enc.get(col, np.nan)
        merged.setdefault("req_rate_hz_median",
                          merged.get("req_rate_hz", 0.0))
        merged.setdefault("req_rate_hz_cv", 0.0)
        row = {c: merged.get(c, np.nan) for c in feature_cols}
    return pd.DataFrame([row])[feature_cols].fillna(0)


# ── inference ──────────────────────────────────────────────────────────────────

def infer(raw: dict, models: dict) -> dict[str, str]:
    """Return {feat_type: predicted_agent} for all five models."""
    preds = {}
    for ft, (clf, le, feature_cols) in models.items():
        try:
            X    = align_features(raw, ft, feature_cols)
            pred = clf.predict(X)[0]
            preds[ft] = le.inverse_transform([pred])[0]
        except Exception:
            preds[ft] = "error"
    return preds


# ── per-trial evaluation ───────────────────────────────────────────────────────

def evaluate_trial(trial: dict, models: dict,
                   max_requests: int) -> list[dict]:
    """
    Run inference at each cumulative request count for one trial.
    Returns list of row dicts for results_per_checkpoint.csv.
    """
    req_records = load_sorted_requests(trial["requests_path"])
    n_total     = min(len(req_records), max_requests)
    rows        = []

    for n_req in range(1, n_total + 1):
        slice_recs = req_records[:n_req]
        cutoff_ts  = parse_ts(slice_recs[-1]["timestamp"])

        req_tmp = write_tmp_requests(slice_recs)
        int_tmp = write_tmp_interactions(
            trial["interactions_path"], cutoff_ts)

        try:
            raw   = extract_all(req_tmp, int_tmp)
            preds = infer(raw, models)
        except Exception:
            preds = {ft: "error" for ft in FEATURE_TYPES}
        finally:
            for p in [req_tmp, int_tmp]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        row = {
            "agent":   trial["agent"],
            "trial":   trial["trial"],
            "n_req":   n_req,
        }
        for ft in FEATURE_TYPES:
            row[f"pred_{ft}"] = preds.get(ft, "error")
        rows.append(row)

    return rows


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(raw_df: pd.DataFrame,
                    max_requests: int) -> pd.DataFrame:
    """
    For each n_req value, compute accuracy and macro-F1 per model,
    and per-agent F1 for the combined model.

    Returns a DataFrame with one row per n_req.
    """
    rows = []
    for n_req in range(1, max_requests + 1):
        # Use only trials that have at least n_req requests
        df_n = raw_df[raw_df["n_req"] == n_req]
        if df_n.empty:
            continue

        y_true = df_n["agent"].tolist()
        row    = {"n_req": n_req, "n_trials": len(df_n)}

        for ft in FEATURE_TYPES:
            y_pred = df_n[f"pred_{ft}"].tolist()
            acc    = accuracy_score(y_true, y_pred)
            f1_mac = f1_score(y_true, y_pred, average="macro",
                              labels=AGENTS, zero_division=0)
            row[f"accuracy_{ft}"]  = round(acc,    4)
            row[f"f1_macro_{ft}"]  = round(f1_mac, 4)

        # Per-agent F1 for combined model
        y_pred_comb = df_n["pred_combined"].tolist()
        for agent in AGENTS:
            f1_ag = f1_score(
                [1 if a == agent else 0 for a in y_true],
                [1 if p == agent else 0 for p in y_pred_comb],
                zero_division=0,
            )
            row[f"f1_{AGENT_SHORT[agent]}"] = round(f1_ag, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ── plotting ───────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(left=1)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_accuracy_f1(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy and macro-F1 for the combined model on one plot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Agent Classification vs. Requests Seen\n(ExtraTrees, 180 trials)",
        fontsize=13, fontweight="bold")

    x = metrics["n_req"]

    # Left: accuracy per model
    ax = axes[0]
    for ft in FEATURE_TYPES:
        lw = 2.5 if ft == "combined" else 1.2
        ls = "-"  if ft == "combined" else "--"
        ax.plot(x, metrics[f"accuracy_{ft}"],
                label=ft, color=MODEL_COLORS[ft], lw=lw, ls=ls)
    _style_ax(ax, "Requests seen", "Accuracy",
              "Accuracy vs. Requests Seen")

    # Right: macro-F1 per model
    ax = axes[1]
    for ft in FEATURE_TYPES:
        lw = 2.5 if ft == "combined" else 1.2
        ls = "-"  if ft == "combined" else "--"
        ax.plot(x, metrics[f"f1_macro_{ft}"],
                label=ft, color=MODEL_COLORS[ft], lw=lw, ls=ls)
    _style_ax(ax, "Requests seen", "Macro F1",
              "Macro F1 vs. Requests Seen")

    plt.tight_layout()
    path = output_dir / "accuracy_f1_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_per_agent_f1(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Per-agent F1 (combined model) over requests seen."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = metrics["n_req"]

    for agent in AGENTS:
        col   = f"f1_{AGENT_SHORT[agent]}"
        label = AGENT_SHORT[agent]
        ax.plot(x, metrics[col],
                label=label, color=AGENT_COLORS[agent], lw=2.0, marker="o",
                markersize=3)

    _style_ax(ax, "Requests seen", "F1 (binary, one-vs-rest)",
              "Per-Agent F1 vs. Requests Seen\n(combined model)")
    path = output_dir / "per_agent_f1_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_per_model_accuracy(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Per-model accuracy curves on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = metrics["n_req"]

    for ft in FEATURE_TYPES:
        lw = 2.5 if ft == "combined" else 1.5
        ax.plot(x, metrics[f"accuracy_{ft}"],
                label=ft, color=MODEL_COLORS[ft], lw=lw,
                ls="-" if ft == "combined" else "--")

    _style_ax(ax, "Requests seen", "Accuracy",
              "Per-Model Accuracy vs. Requests Seen")
    path = output_dir / "per_model_accuracy_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def run(data_dir:      Path,
        model_dir:     Path,
        extractor_dir: str | None,
        output_dir:    Path,
        max_requests:  int) -> None:

    # Add extractor scripts to path
    for d in [extractor_dir, str(data_dir), str(Path(__file__).parent)]:
        if d and d not in sys.path:
            sys.path.insert(0, d)

    os.makedirs(output_dir, exist_ok=True)

    # ── discover trials ───────────────────────────────────────────────────────
    trials = discover_trials(data_dir)
    if not trials:
        sys.exit(f"[error] No trials found under {data_dir}")
    print(f"Found {len(trials)} trials across "
          f"{len(set(t['agent'] for t in trials))} agents")

    # ── load models once ──────────────────────────────────────────────────────
    print("Loading models …")
    models = {}
    for ft in FEATURE_TYPES:
        mp = model_dir / f"extratrees_{ft}.joblib"
        if not mp.exists():
            print(f"  [warn] missing: {mp}")
            continue
        clf, le, feature_cols = joblib.load(mp)
        models[ft] = (clf, le, feature_cols)
    print(f"  Loaded {len(models)} models")

    # ── evaluate all trials ───────────────────────────────────────────────────
    all_rows = []
    n = len(trials)
    for i, trial in enumerate(trials, 1):
        print(f"  [{i:3d}/{n}]  {trial['agent']:<26}  {trial['trial']}", end="\r")
        rows = evaluate_trial(trial, models, max_requests)
        all_rows.extend(rows)
    print()  # newline after \r progress

    # ── save raw results ──────────────────────────────────────────────────────
    raw_df   = pd.DataFrame(all_rows)
    raw_path = output_dir / "results_per_checkpoint.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  Saved raw results → {raw_path}  "
          f"({len(raw_df)} rows)")

    # ── compute metrics ───────────────────────────────────────────────────────
    print("Computing metrics …")
    metrics      = compute_metrics(raw_df, max_requests)
    metrics_path = output_dir / "metrics_over_requests.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"  Saved metrics     → {metrics_path}")

    # Print summary at key request counts
    key_ns = [1, 2, 3, 5, 8, 10, 15, max_requests]
    key_ns = sorted(set(n for n in key_ns if n <= max_requests))
    print(f"\n── Accuracy (combined model) at key request counts ─────────────")
    print(f"  {'n_req':>5}  {'n_trials':>8}  {'accuracy':>9}  {'macro_F1':>9}")
    print(f"  {'─'*40}")
    for _, row in metrics[metrics["n_req"].isin(key_ns)].iterrows():
        print(f"  {int(row['n_req']):>5}  {int(row['n_trials']):>8}  "
              f"{row['accuracy_combined']:>9.3f}  {row['f1_macro_combined']:>9.3f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_accuracy_f1(metrics, output_dir)
    plot_per_agent_f1(metrics, output_dir)
    plot_per_model_accuracy(metrics, output_dir)
    print("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate agent classification across all trials vs requests seen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",       required=True,
                   help="Root directory containing <agent>/<trial>/ subdirs")
    p.add_argument("--model-dir",      required=True,
                   help="Directory containing extratrees_*.joblib files")
    p.add_argument("--extractor-dir",  default=None,
                   help="Directory containing extract_*_feature.py scripts")
    p.add_argument("--output-dir",     default="./eval_results",
                   help="Where to save CSVs and plots (default: ./eval_results)")
    p.add_argument("--max-requests",   type=int, default=30,
                   help="Maximum request count on the x-axis (default: 30)")
    args = p.parse_args()

    run(
        data_dir      = Path(args.data_dir),
        model_dir     = Path(args.model_dir),
        extractor_dir = args.extractor_dir,
        output_dir    = Path(args.output_dir),
        max_requests  = args.max_requests,
    )