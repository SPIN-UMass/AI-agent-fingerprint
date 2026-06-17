"""
plot_eval_results.py
--------------------
Reads the metrics_over_requests.csv produced by evaluate_trials.py
and generates three publication-quality plots.

Output files  (saved to --output-dir)
--------------------------------------
  request_eval_f1_curve.pdf      macro-F1 per feature type vs requests seen
  per_agent_f1_curve.pdf         per-agent F1 (combined model) vs requests seen
  per_model_accuracy_curve.pdf   accuracy per feature type vs requests seen

Usage
-----
  python plot_eval_results.py \
      --metrics-path ./eval_results/metrics_over_requests.csv \
      --output-dir   ./eval_results
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
FEATURE_TYPES = ["temporal", "http", "tls", "behavioral", "combined"]

AGENTS = [
    "autogen_websurfer", "browser_use", "claude_computer_use",
    "gemini_computer_use", "heritrix", "human",
    "nutch", "operator", "scrapy", "skyvern",
]
AGENT_SHORT = {
    "autogen_websurfer":   "AutoGen",
    "browser_use":         "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "heritrix":            "Heritrix",
    "human":               "Human",
    "nutch":               "Nutch",
    "operator":            "Operator",
    "scrapy":              "Scrapy",
    "skyvern":             "Skyvern",
}
AGENT_COLORS = {
    "autogen_websurfer":   "#4C72B0",
    "browser_use":         "#DD8452",
    "claude_computer_use": "#55A868",
    "gemini_computer_use": "#C44E52",
    "heritrix":            "#8172B3",
    "human":               "#937860",
    "nutch":               "#DA8BC3",
    "operator":            "#8C8C8C",
    "scrapy":              "#CCB974",
    "skyvern":             "#64B5CD",
}
MODEL_COLORS = {
    "temporal":   "#4C72B0",
    "http":       "#DD8452",
    "tls":        "#55A868",
    "behavioral": "#C44E52",
    "combined":   "#000000",
}
MODEL_LABELS = {
    "temporal":   "Temporal",
    "http":       "HTTP",
    "tls":        "TLS",
    "behavioral": "Behavioral",
    "combined":   "Combined",
}

# Agents that have behavioral / combined predictions
AGENTS_WITH_BEHAVIORAL = {
    "autogen_websurfer", "browser_use", "claude_computer_use",
    "gemini_computer_use", "operator", "skyvern",
}


# ── shared axis styling ────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update({
        "font.size":    20,
        "legend.fontsize": 16,
    })


def _style_ax(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=1)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{100*y:.0f}")
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_f1_by_model(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Macro-F1 per feature type vs requests seen."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = metrics["n_req"]

    for ft in FEATURE_TYPES:
        col = f"f1_macro_{ft}"
        if col not in metrics.columns or metrics[col].isna().all():
            continue
        lw = 3 if ft == "combined" else 2
        ls = "-"  if ft == "combined" else "--"
        ax.plot(x, metrics[col],
                label=MODEL_LABELS[ft], color=MODEL_COLORS[ft], lw=lw, ls=ls)

    _style_ax(ax, "Requests seen", "Macro F1 (%)")
    plt.tight_layout()
    path = output_dir / "request_eval_f1_curve.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_f1_per_agent(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Per-agent F1 (combined model) vs requests seen."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = metrics["n_req"]

    for agent in AGENTS:
        if agent not in AGENTS_WITH_BEHAVIORAL:
            continue
        col = f"f1_{AGENT_SHORT[agent]}"
        if col not in metrics.columns or metrics[col].isna().all():
            continue
        ax.plot(x, metrics[col],
                label=AGENT_SHORT[agent],
                color=AGENT_COLORS[agent], lw=2.0, marker="o", markersize=3)

    _style_ax(ax, "Requests seen", "F1 (%)")
    plt.tight_layout()
    path = output_dir / "per_agent_f1_curve.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_accuracy_by_model(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy per feature type vs requests seen."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = metrics["n_req"]

    for ft in FEATURE_TYPES:
        col = f"accuracy_{ft}"
        if col not in metrics.columns or metrics[col].isna().all():
            continue
        lw = 3 if ft == "combined" else 2
        ls = "-"  if ft == "combined" else "--"
        ax.plot(x, metrics[col],
                label=MODEL_LABELS[ft], color=MODEL_COLORS[ft], lw=lw, ls=ls)

    _style_ax(ax, "Requests seen", "Accuracy (%)")
    plt.tight_layout()
    path = output_dir / "per_model_accuracy_curve.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def run(metrics_path: Path, output_dir: Path) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Run evaluate_trials.py first to generate it."
        )

    print(f"Loading metrics from {metrics_path} ...")
    metrics = pd.read_csv(metrics_path)
    print(f"  {len(metrics)} rows, {len(metrics.columns)} columns")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots ...")
    plot_f1_by_model(metrics, output_dir)
    plot_f1_per_agent(metrics, output_dir)
    plot_accuracy_by_model(metrics, output_dir)
    print("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot evaluation metrics from evaluate_trials.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-path",
        default="./eval_results/metrics_over_requests.csv",
        help="Path to metrics_over_requests.csv "
             "(default: ./eval_results/metrics_over_requests.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="./eval_results",
        help="Directory to save plots (default: ./eval_results)",
    )
    args = p.parse_args()

    run(
        metrics_path = Path(args.metrics_path),
        output_dir   = Path(args.output_dir),
    )