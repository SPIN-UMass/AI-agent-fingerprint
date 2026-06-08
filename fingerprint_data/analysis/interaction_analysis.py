"""
Interaction Event Log Analyzer
================================
Analyzes logger.js sendBeacon payloads (interactions.jsonl) captured from
web component UX tests.  Designed for cross-system / cross-setting comparison
across multiple traces that share the same directory layout as requests.jsonl:

    traces/
      trial-001/interactions.jsonl
      trial-002/interactions.jsonl
      ...

Each interactions.jsonl contains one JSON object per flush (sendBeacon call).
Each flush has a `batch` array of raw events (mousemove, click, keydown, …).

Analysis dimensions
-------------------
1. Session-level      — page count, total events, session duration, flush count
2. Page-level         — dwell time, event density, first-interaction latency
3. Event-type profile — distribution of event types per page and per trace
4. Interaction timing — inter-event intervals, reaction times (focus→input,
                        mousedown→mouseup, input→submit)
5. Mouse behaviour    — trajectory length, speed, click coordinates
6. Keyboard behaviour — key frequencies, modifier usage
7. Scroll behaviour   — scroll depth, scroll event count
8. App-event funnel   — custom logEvent/addLog events per page
9. Cross-trace        — mean ± std of all metrics; comparison-ready DataFrame

Usage
-----
  # Standard layout
  python interaction_analysis.py --traces ./traces

  # Explicit files
  python interaction_analysis.py --files trial-001/interactions.jsonl trial-002/interactions.jsonl

  # Custom filename inside each trial subdirectory
  python interaction_analysis.py --traces ./traces --filename interactions.jsonl

  # Save plots and CSVs to a specific directory
  python interaction_analysis.py --traces ./traces --out ./results
"""

import json
import re
import sys
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── PAGE LABEL NORMALISATION ─────────────────────────────────────────────────

# Map verbose page titles → short labels for plots
PAGE_LABEL = {
    "Subscribe · V1 Subscription Button": "S1-subscribe-v1",
    "Subscribe · V2 Subscription Button": "S2-subscribe-v2",
    "Subscribe · V3 Subscription Button": "S3-subscribe-v3",
    "S2 · Scroll Gate — UX Behavior Test Suite": "S4-scroll-gate",
    "S3 · Hover Reveal — UX Behavior Test Suite": "S5-hover-reveal",
    "S4 · DOM Mismatch — UX Behavior Test Suite": "S6-dom-mismatch",
}

def short_page(title: str) -> str:
    return PAGE_LABEL.get(title, title.split("·")[-1].strip()[:30])

# ─── PARSING ─────────────────────────────────────────────────────────────────

def parse_ms(val) -> float:
    """Return epoch-ms as float, handling None gracefully."""
    return float(val) if val is not None else float("nan")


def load_interactions(filepath: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load one interactions.jsonl file.

    Returns
    -------
    events_df : one row per event, with trace/flush metadata joined in
    flushes   : raw list of flush-level dicts (for flush-level analysis)
    """
    flushes = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                flushes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows = []
    for fi, flush in enumerate(flushes):
        session     = flush.get("session", "")
        page_title  = flush.get("page", "")
        page_short  = short_page(page_title)
        flush_seq   = flush.get("flushSeq", fi)
        flush_reason= flush.get("flushReason", "")
        user_agent  = flush.get("userAgent", "")

        for ev in flush.get("batch", []):
            rows.append({
                # flush metadata
                "session":      session,
                "page_title":   page_title,
                "page":         page_short,
                "flush_seq":    flush_seq,
                "flush_reason": flush_reason,
                "user_agent":   user_agent,
                # event fields
                "ms":           parse_ms(ev.get("ms")),
                "type":         ev.get("type", ""),
                "target":       ev.get("target", ""),
                "x":            ev.get("x"),
                "y":            ev.get("y"),
                "px":           ev.get("px"),
                "py":           ev.get("py"),
                "button":       ev.get("button"),
                "key":          ev.get("key"),
                "code":         ev.get("code"),
                "ctrl":         ev.get("ctrl"),
                "shift":        ev.get("shift"),
                "alt":          ev.get("alt"),
                "meta":         ev.get("meta"),
                "value":        ev.get("value"),
                "scroll_x":     ev.get("scrollX"),
                "scroll_y":     ev.get("scrollY"),
                "el_scroll_top":ev.get("elScrollTop"),
                "action":       ev.get("action"),        # page load/unload/pagehide
                "vis_state":    ev.get("state"),         # visibility
                "log_id":       ev.get("logId"),
                "app_event":    ev.get("event"),
                "app_detail":   ev.get("detail"),
                "win_w":        ev.get("w"),
                "win_h":        ev.get("h"),
                "referrer":     ev.get("referrer"),
                "url":          ev.get("url"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, flushes

    df = df.sort_values("ms").reset_index(drop=True)
    df["ms_rel"] = df["ms"] - df["ms"].iloc[0]      # ms since session start
    df["iei_ms"] = df["ms"].diff().fillna(0)         # inter-event interval

    return df, flushes


def label_for(filepath: Path) -> str:
    """trial-001/interactions.jsonl → 'trial-001'"""
    label = filepath.parent.name
    return label if label else filepath.stem


def collect_filepaths(traces_dir: str | Path, filename: str = "interactions.jsonl") -> list[Path]:
    root  = Path(traces_dir)
    found = sorted(root.glob(f"*/{filename}"))
    return found if found else sorted(root.glob("*.jsonl"))


def load_all(filepaths: list[Path], filename: str) -> dict[str, pd.DataFrame]:
    traces = {}
    for fp in filepaths:
        df, _ = load_interactions(fp)
        if df.empty:
            print(f"  [warn] {fp} – empty or unparseable, skipped")
            continue
        traces[label_for(fp)] = df
    return traces

# ─── SESSION-LEVEL STATS ─────────────────────────────────────────────────────

def session_stats(df: pd.DataFrame) -> dict:
    duration_s = (df["ms"].max() - df["ms"].min()) / 1000
    return {
        "n_events":        len(df),
        "n_pages":         df["page"].nunique(),
        "duration_s":      duration_s,
        "event_rate_hz":   len(df) / duration_s if duration_s > 0 else 0,
        "n_flushes":       df["flush_seq"].nunique(),
        "n_clicks":        (df["type"] == "click").sum(),
        "n_keydowns":      (df["type"] == "keydown").sum(),
        "n_mousemoves":    (df["type"] == "mousemove").sum(),
        "n_scrolls":       (df["type"] == "scroll").sum(),
        "n_app_events":    (df["type"] == "app_event").sum(),
        "iei_mean_ms":     df["iei_ms"].iloc[1:].mean(),
        "iei_median_ms":   df["iei_ms"].iloc[1:].median(),
        "iei_std_ms":      df["iei_ms"].iloc[1:].std(),
    }

# ─── PAGE-LEVEL STATS ────────────────────────────────────────────────────────

def page_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-page breakdown within one trace."""
    rows = []
    pages = df["page"].unique()
    for pg in pages:
        sub = df[df["page"] == pg].sort_values("ms")
        if sub.empty:
            continue

        duration_s = (sub["ms"].max() - sub["ms"].min()) / 1000

        # First-interaction latency: ms from page load to first non-page event
        load_ev = sub[sub["type"] == "page"]
        load_ms = load_ev["ms"].min() if not load_ev.empty else sub["ms"].min()
        interact = sub[~sub["type"].isin(["page", "visibility"])]
        first_interact_ms = (interact["ms"].min() - load_ms) if not interact.empty else float("nan")

        # Reaction times: focus → first input on same target
        reaction_times = []
        focuses = sub[sub["type"] == "focus"]
        for _, frow in focuses.iterrows():
            tgt = frow["target"]
            inp = sub[(sub["type"] == "input") & (sub["target"].str.startswith(tgt.split("[")[0])) & (sub["ms"] > frow["ms"])]
            if not inp.empty:
                reaction_times.append(inp["ms"].iloc[0] - frow["ms"])

        # mousedown → mouseup duration (click hold time)
        hold_times = []
        mdowns = sub[sub["type"] == "mousedown"]
        for _, mrow in mdowns.iterrows():
            mups = sub[(sub["type"] == "mouseup") & (sub["ms"] > mrow["ms"])]
            if not mups.empty:
                hold_times.append(mups["ms"].iloc[0] - mrow["ms"])

        # Mouse trajectory
        mm = sub[sub["type"] == "mousemove"].dropna(subset=["x", "y"])
        traj_len = 0.0
        if len(mm) > 1:
            dx = mm["x"].diff().dropna()
            dy = mm["y"].diff().dropna()
            traj_len = float(np.sqrt(dx**2 + dy**2).sum())

        # Max scroll depth
        scroll_depth = sub["scroll_y"].dropna().max() if not sub["scroll_y"].dropna().empty else 0

        rows.append({
            "page":                 pg,
            "n_events":             len(sub),
            "duration_s":           duration_s,
            "event_density":        len(sub) / duration_s if duration_s > 0 else 0,
            "first_interact_ms":    first_interact_ms,
            "n_clicks":             (sub["type"] == "click").sum(),
            "n_keydowns":           (sub["type"] == "keydown").sum(),
            "n_mousemoves":         (sub["type"] == "mousemove").sum(),
            "n_scrolls":            (sub["type"] == "scroll").sum(),
            "n_app_events":         (sub["type"] == "app_event").sum(),
            "mouse_traj_px":        traj_len,
            "click_hold_mean_ms":   np.mean(hold_times) if hold_times else float("nan"),
            "focus_react_mean_ms":  np.mean(reaction_times) if reaction_times else float("nan"),
            "scroll_depth_px":      scroll_depth,
            "iei_mean_ms":          sub["iei_ms"].iloc[1:].mean(),
        })
    return pd.DataFrame(rows)


# ─── EVENT TYPE PROFILE ───────────────────────────────────────────────────────

def event_type_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion of each event type, per page."""
    counts = df.groupby(["page", "type"]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions


# ─── APP EVENT FUNNEL ────────────────────────────────────────────────────────

def app_event_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Extract custom app_event rows for funnel analysis."""
    aev = df[df["type"] == "app_event"].copy()
    if aev.empty:
        return aev
    return aev[["page", "log_id", "app_event", "app_detail", "ms_rel"]].reset_index(drop=True)


# ─── CROSS-TRACE AGGREGATION ─────────────────────────────────────────────────

def cross_trace_session(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in traces.items():
        s = session_stats(df)
        s["trace"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("trace")


def cross_trace_page(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-page metrics aggregated (mean ± std) across all traces."""
    all_pages = pd.concat(
        [page_stats(df).assign(trace=name) for name, df in traces.items()],
        ignore_index=True
    )
    numeric_cols = all_pages.select_dtypes(include=np.number).columns.tolist()
    agg = (
        all_pages.groupby("page")[numeric_cols]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def cross_trace_event_profile(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean event-type proportions across traces, per page."""
    profiles = []
    for name, df in traces.items():
        p = event_type_profile(df)   # MultiIndex: (page, type) after unstack
        p["trace"] = name
        profiles.append(p)
    all_p = pd.concat(profiles)      # page is still the index
    type_cols = [c for c in all_p.columns if c != "trace"]
    return all_p.groupby(level=0)[type_cols].mean()


# ─── PLOTTING ────────────────────────────────────────────────────────────────

C = {
    "blue":   "#2563EB",
    "teal":   "#0D9488",
    "amber":  "#D97706",
    "red":    "#DC2626",
    "gray":   "#6B7280",
    "purple": "#7C3AED",
    "green":  "#16A34A",
}

def _ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_trace(df: pd.DataFrame, trace_name: str, out_path=None):
    """8-panel single-trace interaction detail."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Interaction analysis — {trace_name}", fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    pages   = df["page"].unique().tolist()
    palette = [C["blue"], C["teal"], C["amber"], C["purple"], C["green"], C["red"]]
    pg_color= {pg: palette[i % len(palette)] for i, pg in enumerate(pages)}

    # 1. Event timeline coloured by page
    ax1 = fig.add_subplot(gs[0, :2])
    for pg in pages:
        sub = df[df["page"] == pg]
        ax1.scatter(sub["ms_rel"] / 1000, [pg]*len(sub),
                    s=6, color=pg_color[pg], alpha=0.5)
    _ax(ax1, "Event timeline by page", "elapsed time (s)", "")
    ax1.tick_params(axis="y", labelsize=6)

    # 2. Event type counts
    ax2 = fig.add_subplot(gs[0, 2])
    type_counts = df["type"].value_counts()
    ax2.barh(type_counts.index[::-1], type_counts.values[::-1], color=C["teal"])
    _ax(ax2, "Event type counts", "count", "")

    # 3. Inter-event interval distribution
    ax3 = fig.add_subplot(gs[1, 0])
    iei = df["iei_ms"].iloc[1:].clip(upper=df["iei_ms"].quantile(0.98))
    ax3.hist(iei, bins=40, color=C["blue"], edgecolor="white", linewidth=0.3)
    ax3.axvline(iei.mean(), color=C["red"], lw=1.5, linestyle="--",
                label=f"mean {iei.mean():.0f} ms")
    _ax(ax3, "Inter-event interval (IEI)", "ms", "count")
    ax3.legend(fontsize=7)

    # 4. Event density per page (events / second)
    ax4 = fig.add_subplot(gs[1, 1])
    ps = page_stats(df)
    bars = ax4.bar(ps["page"], ps["event_density"], color=C["purple"])
    ax4.set_xticks(range(len(ps)))
    ax4.set_xticklabels(ps["page"], rotation=35, ha="right", fontsize=6)
    _ax(ax4, "Event density per page", "page", "events/s")

    # 5. Mouse trajectory heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    mm = df[df["type"] == "mousemove"].dropna(subset=["x", "y"])
    if not mm.empty:
        w = df["win_w"].dropna().iloc[0] if not df["win_w"].dropna().empty else 1440
        h = df["win_h"].dropna().iloc[0] if not df["win_h"].dropna().empty else 900
        ax5.hist2d(mm["x"], mm["y"], bins=30,
                   range=[[0, w], [0, h]], cmap="Blues")
        ax5.invert_yaxis()
        ax5.set_aspect("equal", adjustable="box")
    _ax(ax5, "Mouse position heatmap", "x (px)", "y (px)")

    # 6. Click coordinates scatter
    ax6 = fig.add_subplot(gs[2, 0])
    cl = df[df["type"] == "click"].dropna(subset=["x", "y"])
    if not cl.empty:
        for pg in pages:
            sub = cl[cl["page"] == pg]
            ax6.scatter(sub["x"], sub["y"], s=40, color=pg_color[pg],
                        label=pg, alpha=0.8, edgecolors="white", lw=0.5)
        ax6.invert_yaxis()
    _ax(ax6, "Click coordinates by page", "x (px)", "y (px)")
    ax6.legend(fontsize=5, loc="upper right")

    # 7. First-interaction latency per page
    ax7 = fig.add_subplot(gs[2, 1])
    ps_sorted = ps.dropna(subset=["first_interact_ms"]).sort_values("first_interact_ms")
    ax7.barh(ps_sorted["page"], ps_sorted["first_interact_ms"] / 1000, color=C["amber"])
    _ax(ax7, "First-interaction latency", "seconds", "")
    ax7.tick_params(axis="y", labelsize=6)

    # 8. App-event funnel timeline
    ax8 = fig.add_subplot(gs[2, 2])
    aev = app_event_funnel(df)
    if not aev.empty:
        for i, (pg, grp) in enumerate(aev.groupby("page")):
            ax8.scatter(grp["ms_rel"] / 1000, [i]*len(grp),
                        s=30, color=pg_color.get(pg, C["gray"]),
                        label=pg, zorder=3)
        ax8.set_yticks(range(aev["page"].nunique()))
        ax8.set_yticklabels(aev["page"].unique(), fontsize=5)
    _ax(ax8, "App-event (logEvent) timeline", "elapsed time (s)", "")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


def plot_cross_trace(traces: dict[str, pd.DataFrame],
                     session_summary: pd.DataFrame,
                     out_path=None):
    """8-panel cross-trace comparison."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Cross-trace interaction comparison — {len(traces)} traces",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    ct_page = cross_trace_page(traces)
    ct_prof = cross_trace_event_profile(traces)

    # 1. Session duration distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(session_summary["duration_s"], bins=min(15, len(session_summary)),
             color=C["blue"], edgecolor="white")
    ax1.axvline(session_summary["duration_s"].mean(), color=C["red"],
                lw=1.5, linestyle="--",
                label=f"mean {session_summary['duration_s'].mean():.1f}s")
    _ax(ax1, "Session duration across traces", "seconds", "count")
    ax1.legend(fontsize=7)

    # 2. Total events per trace
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(session_summary["n_events"], bins=min(15, len(session_summary)),
             color=C["teal"], edgecolor="white")
    ax2.axvline(session_summary["n_events"].mean(), color=C["red"],
                lw=1.5, linestyle="--",
                label=f"mean {session_summary['n_events'].mean():.0f}")
    _ax(ax2, "Total events per trace", "events", "count")
    ax2.legend(fontsize=7)

    # 3. Mean IEI distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(session_summary["iei_mean_ms"], bins=min(15, len(session_summary)),
             color=C["purple"], edgecolor="white")
    ax3.axvline(session_summary["iei_mean_ms"].mean(), color=C["red"],
                lw=1.5, linestyle="--",
                label=f"mean {session_summary['iei_mean_ms'].mean():.0f} ms")
    _ax(ax3, "Mean IEI across traces", "ms", "count")
    ax3.legend(fontsize=7)

    # 4. Per-page event density (mean ± std across traces)
    ax4 = fig.add_subplot(gs[1, 0])
    if "event_density_mean" in ct_page.columns:
        x = np.arange(len(ct_page))
        ax4.bar(x, ct_page["event_density_mean"], color=C["blue"], alpha=0.8)
        ax4.errorbar(x, ct_page["event_density_mean"],
                     yerr=ct_page["event_density_std"].fillna(0),
                     fmt="none", color="black", capsize=3, lw=1)
        ax4.set_xticks(x)
        ax4.set_xticklabels(ct_page["page"], rotation=35, ha="right", fontsize=6)
    _ax(ax4, "Event density per page (mean ± std)", "page", "events/s")

    # 5. Per-page first-interaction latency (mean ± std)
    ax5 = fig.add_subplot(gs[1, 1])
    if "first_interact_ms_mean" in ct_page.columns:
        sub = ct_page.dropna(subset=["first_interact_ms_mean"])
        x = np.arange(len(sub))
        ax5.bar(x, sub["first_interact_ms_mean"] / 1000, color=C["amber"], alpha=0.8)
        ax5.errorbar(x, sub["first_interact_ms_mean"] / 1000,
                     yerr=sub["first_interact_ms_std"].fillna(0) / 1000,
                     fmt="none", color="black", capsize=3, lw=1)
        ax5.set_xticks(x)
        ax5.set_xticklabels(sub["page"], rotation=35, ha="right", fontsize=6)
    _ax(ax5, "First-interaction latency (mean ± std)", "page", "seconds")

    # 6. Mouse trajectory length per page (mean ± std)
    ax6 = fig.add_subplot(gs[1, 2])
    if "mouse_traj_px_mean" in ct_page.columns:
        x = np.arange(len(ct_page))
        ax6.bar(x, ct_page["mouse_traj_px_mean"], color=C["teal"], alpha=0.8)
        ax6.errorbar(x, ct_page["mouse_traj_px_mean"],
                     yerr=ct_page["mouse_traj_px_std"].fillna(0),
                     fmt="none", color="black", capsize=3, lw=1)
        ax6.set_xticks(x)
        ax6.set_xticklabels(ct_page["page"], rotation=35, ha="right", fontsize=6)
    _ax(ax6, "Mouse trajectory length (mean ± std)", "page", "pixels")

    # 7. Event type profile heatmap across pages
    ax7 = fig.add_subplot(gs[2, :2])
    if not ct_prof.empty:
        im = ax7.imshow(ct_prof.values.T, aspect="auto", cmap="Blues",
                        vmin=0, vmax=ct_prof.values.max())
        ax7.set_xticks(range(len(ct_prof.index)))
        ax7.set_xticklabels(ct_prof.index, rotation=35, ha="right", fontsize=6)
        ax7.set_yticks(range(len(ct_prof.columns)))
        ax7.set_yticklabels(ct_prof.columns, fontsize=7)
        plt.colorbar(im, ax=ax7, fraction=0.02, label="proportion")
    _ax(ax7, "Event type profile per page (mean across traces)", "page", "event type")

    # 8. IEI spaghetti: per-trace IEI over session time
    ax8 = fig.add_subplot(gs[2, 2])
    for name, df in traces.items():
        iei = df["iei_ms"].clip(upper=df["iei_ms"].quantile(0.98))
        ax8.plot(df["ms_rel"] / 1000, iei, lw=0.3, color=C["blue"], alpha=0.3)
    _ax(ax8, "IEI over session (all traces)", "elapsed time (s)", "IEI (ms)")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── CONSOLE OUTPUT ───────────────────────────────────────────────────────────

def _sec(title):
    print(f"\n{'─'*65}\n  {title}\n{'─'*65}")


def print_session_summary(summary: pd.DataFrame):
    _sec("Session-level summary (mean ± std across traces)")
    metrics = ["n_events", "duration_s", "event_rate_hz", "n_clicks",
               "n_keydowns", "n_mousemoves", "n_scrolls", "n_app_events",
               "iei_mean_ms", "iei_std_ms"]
    for m in metrics:
        if m in summary.columns:
            print(f"  {m:<28} {summary[m].mean():>10.3f}  ±  {summary[m].std():.3f}")


def print_page_summary(traces: dict[str, pd.DataFrame]):
    _sec("Per-page metrics (mean across traces)")
    ct = cross_trace_page(traces)
    cols = ["page", "duration_s_mean", "event_density_mean",
            "first_interact_ms_mean", "mouse_traj_px_mean",
            "n_clicks_mean", "n_scrolls_mean", "scroll_depth_px_mean"]
    cols = [c for c in cols if c in ct.columns]
    print(ct[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def print_event_profile(traces: dict[str, pd.DataFrame]):
    _sec("Event type profile per page (mean proportion across traces)")
    ct = cross_trace_event_profile(traces)
    print(ct.round(3).to_string())


def print_app_funnel(traces: dict[str, pd.DataFrame]):
    _sec("App-event funnel (all traces)")
    rows = []
    for name, df in traces.items():
        aev = app_event_funnel(df)
        if aev.empty:
            continue
        for _, r in aev.iterrows():
            rows.append({
                "trace":      name,
                "page":       r["page"],
                "log_id":     r["log_id"],
                "app_event":  r["app_event"],
                "detail":     str(r["app_detail"])[:40],
                "at_s":       f"{r['ms_rel']/1000:.2f}",
            })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Web interaction event log analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python interaction_analysis.py --traces ./traces
  python interaction_analysis.py --traces ./traces --filename interactions.jsonl
  python interaction_analysis.py --files trial-001/interactions.jsonl trial-002/interactions.jsonl
  python interaction_analysis.py --traces ./traces --out ./results
""")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--traces", metavar="DIR",
                     help="root traces directory containing trial-NNN subdirectories")
    grp.add_argument("--files",  nargs="+",
                     help="explicit list of .jsonl files")
    parser.add_argument("--filename", default="interactions.jsonl",
                        help="filename inside each trial subdirectory (default: interactions.jsonl)")
    parser.add_argument("--out", default=".",
                        help="output directory for plots and CSVs (default: .)")
    args = parser.parse_args()

    # ── Discover files ───────────────────────────────────────────────────────
    if args.traces:
        filepaths = collect_filepaths(args.traces, filename=args.filename)
    else:
        filepaths = sorted(set(
            Path(p) for pat in args.files for p in glob.glob(pat, recursive=True)
        ))

    if not filepaths:
        print("No interaction files found. Exiting.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {len(filepaths)} interaction file(s)…")
    traces = {}
    for fp in filepaths:
        df, _ = load_interactions(fp)
        label = label_for(fp)
        if df.empty:
            print(f"  [warn] {fp} – empty or unparseable, skipped")
            continue
        traces[label] = df
        print(f"  {label}: {len(df)} events across {df['page'].nunique()} page(s)")

    if not traces:
        print("No valid traces loaded. Exiting.")
        sys.exit(1)

    # ── Analysis ─────────────────────────────────────────────────────────────
    session_summary = cross_trace_session(traces)
    session_summary.to_csv(out_dir / "interaction_session_summary.csv")
    print(f"\n  Saved: {out_dir / 'interaction_session_summary.csv'}")

    page_summary = cross_trace_page(traces)
    page_summary.to_csv(out_dir / "interaction_page_summary.csv", index=False)
    print(f"  Saved: {out_dir / 'interaction_page_summary.csv'}")

    event_profile = cross_trace_event_profile(traces)
    event_profile.to_csv(out_dir / "interaction_event_profile.csv")
    print(f"  Saved: {out_dir / 'interaction_event_profile.csv'}")

    print_session_summary(session_summary)
    print_page_summary(traces)
    print_event_profile(traces)
    print_app_funnel(traces)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(traces) == 1:
        name, df = next(iter(traces.items()))
        plot_single_trace(df, name,
                          out_path=str(out_dir / f"interaction_{name}.png"))
    else:
        # First trace detail
        name0, df0 = next(iter(traces.items()))
        plot_single_trace(df0, name0,
                          out_path=str(out_dir / f"interaction_{name0}.png"))
        # Cross-trace comparison
        plot_cross_trace(traces, session_summary,
                         out_path=str(out_dir / "interaction_cross_trace.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()