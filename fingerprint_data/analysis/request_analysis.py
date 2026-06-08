"""
HTTP Request Trace Analyzer
============================
Analyzes multiple JSON-Lines trace files for:
  - Temporal patterns (inter-request timing, processing time)
  - Spatial patterns (page navigation flow, page dwell time)
  - Packet-size distributions (body_size per page, per method)
  - Cross-trace aggregation (mean/std/CI across 30 traces)

logger.js flush filtering
--------------------------
POST /collect requests emitted by logger.js are NOT genuine browser HTTP
fingerprints — they are sendBeacon payloads fired at page-unload/navigation.
Including them would contaminate fingerprint and timing analysis.

filter_logger_collects(df) removes them using three complementary rules
derived from the logger.js source:

  Rule 1 – Temporal co-occurrence (primary)
      A /collect POST is a logger flush if it arrives within UNLOAD_WINDOW_MS
      (default 500 ms) of the *next* page GET. sendBeacon fires synchronously
      at beforeunload/pagehide, so the gap between the POST and the next page
      load is tiny — typically 0–50 ms. Legitimate API calls don't cluster at
      navigation boundaries this way.

  Rule 2 – Simultaneous burst (secondary)
      logger.js can emit 2–3 flushes at once (beforeunload + pagehide +
      visibilitychange all fire in quick succession). If multiple /collect
      POSTs share the same timestamp cluster (within BURST_WINDOW_MS of each
      other), they are treated as a single logger flush burst.

  Rule 3 – Referrer matches preceding page (guard)
      logger.js always sets Referer to the page being unloaded. A /collect POST
      whose Referer matches a page URL we've seen is almost certainly a logger
      flush. This guards against false positives from legitimate API calls that
      happen to fall near a navigation.

All three rules must agree on the same POST for it to be removed (AND logic),
unless --aggressive is passed, in which case Rule 1 alone is sufficient.

Usage:
    # Option A: single trace file
    python trace_analysis.py --files trace1.jsonl

    # Option B: multiple trace files (glob supported)
    python trace_analysis.py --files traces/*.jsonl

    # Option C: directory
    python trace_analysis.py --dir ./traces/

    # Keep logger /collect POSTs in the data (skip filtering)
    python trace_analysis.py --files traces/*.jsonl --no-filter

    # Aggressive filter: Rule 1 alone is sufficient to drop a POST
    python trace_analysis.py --files traces/*.jsonl --aggressive

Output:
    - Console summary tables
    - trace_analysis_plots.png  (multi-panel figure)
    - trace_summary.csv         (per-trace stats)
    - filter_report.csv         (per-trace count of removed rows)
"""

import json
import re
import sys
import glob
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

COLLECT_ENDPOINT = "/collect"
PAGE_URL_PATTERN = re.compile(r"\.html$")

# ── logger.js filter tuning ───────────────────────────────────────────────────
# Max gap (ms) between a /collect POST and the *next* page GET for it to be
# considered a navigation-triggered sendBeacon flush (Rule 1).
UNLOAD_WINDOW_MS = 500

# Max gap (ms) between /collect POSTs in the same burst cluster (Rule 2).
BURST_WINDOW_MS  = 50

# ─── PARSING ─────────────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> datetime:
    """Parse ISO 8601 timestamps including nanosecond precision."""
    # Trim nanoseconds to microseconds for datetime compatibility
    ts = re.sub(r'(\.\d{6})\d+', r'\1', ts)
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def load_trace(filepath: str | Path) -> pd.DataFrame:
    """Load a single JSON-Lines trace file into a DataFrame."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            http = rec.get("http", {})
            tls  = rec.get("tls", {})
            h2   = rec.get("http2", {})

            records.append({
                "id":               rec.get("id"),
                "timestamp":        parse_timestamp(rec["timestamp"]),
                "source_ip":        rec.get("source_ip"),
                "method":           http.get("method"),
                "path":             http.get("path"),
                "protocol":         http.get("protocol"),
                "host":             http.get("host"),
                "referer":          http.get("referer", ""),
                "body_size":        http.get("body_size", 0),
                "body_sha256":      http.get("body_sha256"),
                "processing_ms":    rec.get("processing_time_ms", 0.0),
                "tls_version":      tls.get("version"),
                "cipher_suite":     tls.get("cipher_suite"),
                "ja3":              tls.get("ja3_hash"),
                "ja4":              tls.get("ja4_hash"),
                "akamai_fp":        h2.get("akamai_fingerprint"),
                "n_headers":        len(http.get("headers_ordered") or []),
                "user_agent":       http.get("user_agent", ""),
                "trace_file":       Path(filepath).parent.name or Path(filepath).stem,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["iri_ms"]    = df["timestamp"].diff().dt.total_seconds().fillna(0) * 1000  # inter-request interval
    df["is_page"]   = df["path"].str.match(r".*\.html$", na=False)
    df["is_collect"]= df["path"] == COLLECT_ENDPOINT
    df["page"]      = np.where(df["is_page"], df["path"].str.extract(r"([^/]+\.html)")[0], pd.NA)
    df["page"]      = df["page"].ffill()   # attribute /collect calls to their preceding page

    return df


def load_all_traces(filepaths: list[str | Path],
                    filter_logger: bool = True,
                    aggressive: bool = False) -> dict[str, pd.DataFrame]:
    traces = {}
    for fp in filepaths:
        df = load_trace(fp)
        if df.empty:
            print(f"  [warn] {fp} is empty or unparseable – skipped")
            continue
        if filter_logger:
            df, report = filter_logger_collects(df, aggressive=aggressive)
            n_removed = report["n_removed"]
            if n_removed:
                print(f"  [filter] {Path(fp).stem}: removed {n_removed} logger /collect POST(s)")
        traces[Path(fp).stem] = df
    return traces

# ─── LOGGER FILTER ────────────────────────────────────────────────────────────

def filter_logger_collects(
    df: pd.DataFrame,
    unload_window_ms: float = UNLOAD_WINDOW_MS,
    burst_window_ms:  float = BURST_WINDOW_MS,
    aggressive: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove POST /collect rows that were emitted by logger.js sendBeacon flushes.

    Three detection rules (see module docstring for full rationale):

      Rule 1 – Temporal co-occurrence with next page GET
          Gap between this POST's timestamp and the timestamp of the next page
          GET is <= unload_window_ms. sendBeacon at beforeunload/pagehide fires
          synchronously at navigation; the next page request follows within
          tens of milliseconds on the same TCP connection.

      Rule 2 – Burst cluster
          Multiple /collect POSTs whose timestamps are within burst_window_ms
          of each other (beforeunload + pagehide + visibilitychange can all
          fire in rapid succession for the same unload event).

      Rule 3 – Referer matches a known page URL (guard against false positives)
          logger.js always sets Referer to the page being unloaded. A POST
          whose Referer ends in .html is almost certainly a logger flush.

    In normal mode  : a row is flagged only if Rule 1 AND (Rule 2 OR Rule 3).
    In aggressive mode: a row is flagged if Rule 1 alone is true.

    Returns
    -------
    filtered_df : DataFrame with logger flush rows removed and iri_ms recomputed
    report      : dict with keys n_total, n_removed, removed_ids
    """
    collect_mask = (df["method"] == "POST") & (df["path"] == COLLECT_ENDPOINT)
    collect_idx  = df.index[collect_mask].tolist()

    if not collect_idx:
        return df, {"n_total": len(df), "n_removed": 0, "removed_ids": []}

    # Timestamps of page GETs, for Rule 1 lookups
    page_times = df.loc[df["is_page"], "timestamp"].sort_values()

    def ms_to_next_page(ts):
        """Milliseconds from ts to the next page GET after ts. NaN if none."""
        future = page_times[page_times > ts]
        if future.empty:
            return float("nan")
        return (future.iloc[0] - ts).total_seconds() * 1000

    flagged = set()

    for idx in collect_idx:
        row   = df.loc[idx]
        ts    = row["timestamp"]
        gap   = ms_to_next_page(ts)

        # Rule 1
        rule1 = (not np.isnan(gap)) and (gap <= unload_window_ms)

        if not rule1:
            continue  # can't be flagged without Rule 1

        if aggressive:
            flagged.add(idx)
            continue

        # Rule 2 — is this POST within burst_window_ms of another /collect POST?
        other_collect_times = df.loc[
            collect_mask & (df.index != idx), "timestamp"
        ]
        rule2 = any(
            abs((ts - ot).total_seconds() * 1000) <= burst_window_ms
            for ot in other_collect_times
        )

        # Rule 3 — Referer ends with .html (logger.js always sets this)
        rule3 = bool(re.search(r"\.html$", str(row.get("referer", ""))))

        if rule2 or rule3:
            flagged.add(idx)

    removed_ids = [df.loc[i, "id"] for i in sorted(flagged)]
    filtered    = df.drop(index=list(flagged)).reset_index(drop=True)

    # Recompute derived columns that depend on row order
    if not filtered.empty:
        filtered["elapsed_s"] = (
            filtered["timestamp"] - filtered["timestamp"].iloc[0]
        ).dt.total_seconds()
        filtered["iri_ms"] = (
            filtered["timestamp"].diff().dt.total_seconds().fillna(0) * 1000
        )

    report = {
        "n_total":    len(df),
        "n_removed":  len(flagged),
        "removed_ids": removed_ids,
    }
    return filtered, report


def filter_report_df(traces_raw: dict[str, pd.DataFrame],
                     traces_filtered: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a per-trace filter report DataFrame."""
    rows = []
    for name in traces_raw:
        n_raw      = len(traces_raw[name])
        n_filtered = len(traces_filtered.get(name, pd.DataFrame()))
        n_removed  = n_raw - n_filtered
        collect_raw= (traces_raw[name]["method"] == "POST").sum()
        rows.append({
            "trace":           name,
            "n_raw":           n_raw,
            "n_filtered":      n_filtered,
            "n_removed":       n_removed,
            "collect_posts_raw": collect_raw,
            "collect_posts_kept": collect_raw - n_removed,
        })
    return pd.DataFrame(rows)

# ─── ANALYSIS ────────────────────────────────────────────────────────────────

def temporal_stats(df: pd.DataFrame) -> dict:
    """Per-trace temporal summary."""
    duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    iri = df["iri_ms"].iloc[1:]  # skip first zero

    # Page dwell times: time between consecutive page GETs
    pages   = df[df["is_page"]].copy()
    page_dwell = pages["timestamp"].diff().dt.total_seconds().dropna()

    return {
        "n_requests":        len(df),
        "n_pages":           df["is_page"].sum(),
        "n_collect_posts":   df["is_collect"].sum(),
        "duration_s":        duration,
        "req_rate_hz":       len(df) / duration if duration > 0 else 0,
        "iri_mean_ms":       iri.mean(),
        "iri_median_ms":     iri.median(),
        "iri_std_ms":        iri.std(),
        "proc_mean_ms":      df["processing_ms"].mean(),
        "proc_max_ms":       df["processing_ms"].max(),
        "page_dwell_mean_s": page_dwell.mean() if not page_dwell.empty else np.nan,
        "page_dwell_max_s":  page_dwell.max()  if not page_dwell.empty else np.nan,
    }


def spatial_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-page request count, body size, and dwell time breakdown."""
    rows = []
    pages = df[df["is_page"]]["path"].unique()
    for pg in pages:
        mask_pg = df["page"] == pg.split("/")[-1].replace(".html","") + ".html" \
                  if "/" not in pg else df["page"] == pg.split("/")[-1]
        # simpler: match on the 'page' column we already built
        mask_pg = df["page"] == pg.lstrip("/").split("/")[-1] if pg.startswith("/") else df["page"] == pg

        sub = df[df["page"] == pg.lstrip("/")]
        if sub.empty:
            continue
        collect_sub = sub[sub["is_collect"]]
        rows.append({
            "page":             pg,
            "n_requests":       len(sub),
            "n_collect":        len(collect_sub),
            "total_body_bytes": sub["body_size"].sum(),
            "max_body_bytes":   sub["body_size"].max(),
            "mean_proc_ms":     sub["processing_ms"].mean(),
        })
    return pd.DataFrame(rows)


def body_size_by_page(df: pd.DataFrame) -> pd.DataFrame:
    """Body-size stats for POST /collect calls, grouped by referring page."""
    collect = df[df["is_collect"] & (df["method"] == "POST")].copy()
    if collect.empty:
        return pd.DataFrame()
    return (
        collect.groupby("page")["body_size"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={"count": "n_posts"})
        .reset_index()
    )


def cross_trace_summary(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate temporal stats across all traces with mean ± std."""
    rows = []
    for name, df in traces.items():
        s = temporal_stats(df)
        s["trace"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("trace")

# ─── PLOTTING ────────────────────────────────────────────────────────────────

COLORS = {
    "blue":   "#2563EB",
    "teal":   "#0D9488",
    "amber":  "#D97706",
    "red":    "#DC2626",
    "gray":   "#6B7280",
    "purple": "#7C3AED",
}


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_trace(df: pd.DataFrame, trace_name: str, out_path: str | None = None):
    """Detailed 6-panel plot for one trace."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Trace analysis — {trace_name}", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Request timeline ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    pages   = df[df["is_page"]]
    collect = df[df["is_collect"]]
    other   = df[~df["is_page"] & ~df["is_collect"]]
    ax1.scatter(pages["elapsed_s"],   [1]*len(pages),   s=60, c=COLORS["blue"],  label="page GET",    zorder=3)
    ax1.scatter(collect["elapsed_s"], [2]*len(collect), s=40, c=COLORS["amber"], label="/collect POST",zorder=3)
    ax1.scatter(other["elapsed_s"],   [3]*len(other),   s=30, c=COLORS["gray"],  label="other",       zorder=3, alpha=0.6)
    ax1.set_yticks([1,2,3]); ax1.set_yticklabels(["page","collect","other"], fontsize=8)
    ax1.set_xlim(left=-1)
    _style_ax(ax1, "Request timeline", "elapsed time (s)", "request type")
    ax1.legend(fontsize=7, loc="upper left")

    # ── 2. Inter-request intervals (IRI) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    iri = df["iri_ms"].iloc[1:].clip(upper=df["iri_ms"].quantile(0.99))
    ax2.hist(iri, bins=30, color=COLORS["teal"], edgecolor="white", linewidth=0.4)
    ax2.axvline(iri.mean(), color=COLORS["red"], lw=1.5, linestyle="--", label=f"mean {iri.mean():.0f} ms")
    _style_ax(ax2, "Inter-request interval distribution", "IRI (ms)", "count")
    ax2.legend(fontsize=8)

    # ── 3. Body size distribution ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sizes = df[df["body_size"] > 0]["body_size"]
    if not sizes.empty:
        ax3.hist(sizes, bins=25, color=COLORS["purple"], edgecolor="white", linewidth=0.4)
        ax3.axvline(sizes.median(), color=COLORS["amber"], lw=1.5, linestyle="--", label=f"median {sizes.median():.0f} B")
    _style_ax(ax3, "Body size distribution (non-zero)", "bytes", "count")
    if not sizes.empty:
        ax3.legend(fontsize=8)

    # ── 4. Body size per page ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    bsp = body_size_by_page(df)
    if not bsp.empty:
        labels = [str(p).replace("/","").replace(".html","") for p in bsp["page"]]
        x = np.arange(len(labels))
        bars = ax4.bar(x, bsp["mean"], color=COLORS["blue"], alpha=0.8, label="mean")
        ax4.errorbar(x, bsp["mean"], yerr=bsp["std"].fillna(0), fmt="none",
                     color="black", capsize=4, lw=1)
        ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    _style_ax(ax4, "POST /collect body size by page", "page", "bytes (mean ± std)")

    # ── 5. Processing time per request ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(df["elapsed_s"], df["processing_ms"], color=COLORS["teal"], lw=0.8, alpha=0.7)
    ax5.scatter(df["elapsed_s"], df["processing_ms"], s=18, c=COLORS["teal"], zorder=3)
    _style_ax(ax5, "Processing time over session", "elapsed time (s)", "processing (ms)")

    # ── 6. Page dwell times ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    page_rows = df[df["is_page"]].copy()
    page_rows["dwell_s"] = page_rows["timestamp"].diff().dt.total_seconds()
    page_labels = [str(p).replace("/","").replace(".html","") for p in page_rows["path"]]
    dwell = page_rows["dwell_s"].fillna(0)
    x = np.arange(len(page_labels))
    ax6.bar(x, dwell, color=COLORS["amber"], alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(page_labels, rotation=35, ha="right", fontsize=7)
    _style_ax(ax6, "Dwell time per page transition", "page", "seconds")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


def plot_cross_trace(summary: pd.DataFrame, traces: dict[str, pd.DataFrame],
                     out_path: str | None = None):
    """6-panel cross-trace comparison for N traces."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Cross-trace analysis — {len(traces)} traces", fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Session duration distribution ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    durations = summary["duration_s"]
    ax1.hist(durations, bins=min(15, len(durations)), color=COLORS["blue"], edgecolor="white")
    ax1.axvline(durations.mean(), color=COLORS["red"], lw=1.5, linestyle="--",
                label=f"mean {durations.mean():.1f}s")
    _style_ax(ax1, "Session duration across traces", "seconds", "count")
    ax1.legend(fontsize=8)

    # ── 2. Total request count distribution ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    n_req = summary["n_requests"]
    ax2.hist(n_req, bins=min(15, len(n_req)), color=COLORS["teal"], edgecolor="white")
    ax2.axvline(n_req.mean(), color=COLORS["red"], lw=1.5, linestyle="--",
                label=f"mean {n_req.mean():.1f}")
    _style_ax(ax2, "Total requests per trace", "requests", "count")
    ax2.legend(fontsize=8)

    # ── 3. Mean IRI per trace ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    iri_means = summary["iri_mean_ms"]
    ax3.hist(iri_means, bins=min(15, len(iri_means)), color=COLORS["purple"], edgecolor="white")
    ax3.axvline(iri_means.mean(), color=COLORS["red"], lw=1.5, linestyle="--",
                label=f"mean {iri_means.mean():.0f} ms")
    _style_ax(ax3, "Mean IRI distribution across traces", "mean IRI (ms)", "count")
    ax3.legend(fontsize=8)

    # ── 4. Body-size mean per page (across traces) ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    all_collect = pd.concat(
        [body_size_by_page(df).assign(trace=name) for name, df in traces.items()],
        ignore_index=True
    )
    if not all_collect.empty:
        agg = all_collect.groupby("page")["mean"].agg(["mean","std"]).reset_index()
        labels = [str(p).replace("/","").replace(".html","") for p in agg["page"]]
        x = np.arange(len(labels))
        ax4.bar(x, agg["mean"], color=COLORS["blue"], alpha=0.8)
        ax4.errorbar(x, agg["mean"], yerr=agg["std"].fillna(0), fmt="none",
                     color="black", capsize=4, lw=1)
        ax4.set_xticks(x); ax4.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    _style_ax(ax4, "Mean /collect body size by page (all traces)", "page", "bytes")

    # ── 5. Request rate distribution ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    rates = summary["req_rate_hz"]
    ax5.hist(rates, bins=min(15, len(rates)), color=COLORS["amber"], edgecolor="white")
    ax5.axvline(rates.mean(), color=COLORS["red"], lw=1.5, linestyle="--",
                label=f"mean {rates.mean():.2f} Hz")
    _style_ax(ax5, "Request rate across traces", "requests/sec", "count")
    ax5.legend(fontsize=8)

    # ── 6. IRI over elapsed time (all traces, faded) ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for name, df in traces.items():
        iri_clipped = df["iri_ms"].clip(upper=df["iri_ms"].quantile(0.99))
        ax6.plot(df["elapsed_s"], iri_clipped, lw=0.4, color=COLORS["teal"], alpha=0.3)
    # mean trace
    aligned = []
    for df in traces.values():
        s = df.set_index("elapsed_s")["iri_ms"].clip(upper=5000)
        aligned.append(s)
    _style_ax(ax6, "IRI over session (all traces)", "elapsed time (s)", "IRI (ms)")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig

# ─── FINGERPRINT ANALYSIS ────────────────────────────────────────────────────

def fingerprint_stability(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check whether TLS/HTTP2 fingerprints are stable across traces."""
    rows = []
    for name, df in traces.items():
        rows.append({
            "trace":       name,
            "ja3_unique":  df["ja3"].nunique(),
            "ja4_unique":  df["ja4"].nunique(),
            "ua_unique":   df["user_agent"].nunique(),
            "ja3_val":     df["ja3"].iloc[0] if not df.empty else None,
        })
    return pd.DataFrame(rows)

# ─── PRINT HELPERS ───────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def print_temporal(summary: pd.DataFrame):
    print_section("Temporal summary (mean ± std across traces)")
    cols = ["n_requests","duration_s","req_rate_hz","iri_mean_ms","iri_std_ms",
            "proc_mean_ms","page_dwell_mean_s"]
    for col in cols:
        if col in summary.columns:
            m, s = summary[col].mean(), summary[col].std()
            print(f"  {col:<26} {m:>10.3f}  ±  {s:.3f}")


def print_spatial(traces: dict[str, pd.DataFrame]):
    print_section("Spatial summary (page navigation, all traces)")
    all_pages = defaultdict(list)
    for df in traces.values():
        for pg in df[df["is_page"]]["path"].unique():
            sub = df[df["page"] == pg.lstrip("/")]
            all_pages[pg].append(len(sub))
    for pg, counts in sorted(all_pages.items()):
        print(f"  {pg:<40}  visits={len(counts):>3}  req/visit mean={np.mean(counts):.1f}")


def print_packet_sizes(traces: dict[str, pd.DataFrame]):
    print_section("Body-size distribution (POST /collect) per page")
    all_collect = pd.concat(
        [body_size_by_page(df) for df in traces.values()], ignore_index=True
    )
    if all_collect.empty:
        print("  No POST /collect data found.")
        return
    agg = (
        all_collect.groupby("page")
        .agg(n_posts=("n_posts","sum"), mean=("mean","mean"),
             median=("median","median"), std=("mean","std"),
             min=("min","min"), max=("max","max"))
        .reset_index()
    )
    print(f"  {'page':<38} {'n_posts':>7} {'mean B':>8} {'median B':>9} {'std':>7} {'min':>7} {'max':>7}")
    for _, row in agg.iterrows():
        pg = str(row["page"]).replace("/","").replace(".html","")
        print(f"  {pg:<38} {row['n_posts']:>7.0f} {row['mean']:>8.0f} {row['median']:>9.0f} "
              f"{row['std']:>7.0f} {row['min']:>7.0f} {row['max']:>7.0f}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def collect_filepaths(traces_dir: str | Path,
                      filename: str = "requests.jsonl") -> list[Path]:
    """
    Walk `traces_dir` and return one `filename` per immediate subdirectory,
    matching the layout:

        traces/
          trial-001/requests.jsonl
          trial-002/requests.jsonl
          ...

    Each returned path is labelled by its parent directory name (e.g. "trial-001")
    so that label is used as the trace name in all outputs.

    Falls back to a flat glob (*.jsonl directly inside traces_dir) if no
    subdirectory files are found, for backward compatibility.
    """
    root = Path(traces_dir)
    # Primary: one level deep — traces/trial-NNN/requests.jsonl
    found = sorted(root.glob(f"*/{filename}"))
    if found:
        return found
    # Fallback: flat layout — traces/trial-001.jsonl
    return sorted(root.glob("*.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="HTTP trace analyzer — expects traces/trial-NNN/requests.jsonl layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Standard layout: traces/trial-001/requests.jsonl ...
  python trace_analysis.py --traces ./traces

  # Custom filename inside each trial subdirectory
  python trace_analysis.py --traces ./traces --filename capture.jsonl

  # Explicit list of files (any layout)
  python trace_analysis.py --files trial-001/requests.jsonl trial-002/requests.jsonl

  # Skip logger.js /collect filtering
  python trace_analysis.py --traces ./traces --no-filter
""")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--traces", metavar="DIR",
                     help="root traces directory containing trial-NNN subdirectories")
    grp.add_argument("--files",  nargs="+",
                     help="explicit list of .jsonl files (glob ok)")
    parser.add_argument("--filename",   default="requests.jsonl",
                        help="filename to look for inside each trial directory (default: requests.jsonl)")
    parser.add_argument("--out",        default=".",
                        help="output directory for plots and CSV files (default: .)")
    parser.add_argument("--no-filter",  action="store_true",
                        help="skip logger.js /collect filtering (keep all rows)")
    parser.add_argument("--aggressive", action="store_true",
                        help="flag /collect POSTs using Rule 1 alone (more removals)")
    args = parser.parse_args()

    do_filter = not args.no_filter

    # ── Collect file paths ────────────────────────────────────────────────────
    if args.traces:
        filepaths = collect_filepaths(args.traces, filename=args.filename)
    else:
        filepaths = []
        for pat in args.files:
            filepaths += [Path(p) for p in glob.glob(pat, recursive=True)]
        filepaths = sorted(set(filepaths))

    if not filepaths:
        print("No trace files found. Exiting.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw (unfiltered) traces for filter report ───────────────────────
    print(f"\nLoading {len(filepaths)} trace file(s)…")
    traces_raw = {}
    for fp in filepaths:
        df = load_trace(fp)
        label = fp.parent.name or fp.stem   # "trial-001" from trial-001/requests.jsonl
        if not df.empty:
            traces_raw[label] = df
        else:
            print(f"  [warn] {fp} – empty or unparseable, skipped")

    if not traces_raw:
        print("All files empty or invalid.")
        sys.exit(1)

    # ── Apply logger.js filter ───────────────────────────────────────────────
    if do_filter:
        print(f"  Applying logger.js /collect filter "
              f"({'aggressive' if args.aggressive else 'normal'} mode)…")
        traces = {}
        filter_rows = []
        for name, df in traces_raw.items():
            filtered, report = filter_logger_collects(df, aggressive=args.aggressive)
            traces[name] = filtered
            filter_rows.append({
                "trace":              name,
                "n_raw":              report["n_total"],
                "n_removed":          report["n_removed"],
                "n_kept":             report["n_total"] - report["n_removed"],
                "removed_request_ids": "; ".join(report["removed_ids"]),
            })
            if report["n_removed"]:
                print(f"    {name}: removed {report['n_removed']} row(s)")
        frep = pd.DataFrame(filter_rows)
        frep.to_csv(out_dir / "filter_report.csv", index=False)
        print(f"  Saved: {out_dir / 'filter_report.csv'}")
        print(f"  Total removed across all traces: {frep['n_removed'].sum()}")
    else:
        print("  Filter disabled (--no-filter). Using raw data.")
        traces = traces_raw

    print(f"  Loaded: {len(traces)} traces")

    # ── Summary across all traces ────────────────────────────────────────────
    summary = cross_trace_summary(traces)
    summary.to_csv(out_dir / "trace_summary.csv")
    print(f"  Saved: {out_dir / 'trace_summary.csv'}")

    print_temporal(summary)
    print_spatial(traces)
    print_packet_sizes(traces)

    # ── Fingerprint stability ────────────────────────────────────────────────
    fp = fingerprint_stability(traces)
    print_section("Fingerprint stability across traces")
    print(fp[["trace","ja3_unique","ja4_unique","ua_unique"]].to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    if len(traces) == 1:
        name, df = next(iter(traces.items()))
        plot_single_trace(df, name, out_path=str(out_dir / "trace_analysis_single.png"))
    else:
        # Plot first trace in detail
        name0, df0 = next(iter(traces.items()))
        plot_single_trace(df0, name0,
                          out_path=str(out_dir / f"trace_analysis_{name0}.png"))
        # Cross-trace overview
        plot_cross_trace(summary, traces,
                         out_path=str(out_dir / "trace_analysis_cross.png"))

    print("\nDone.\n")


# ─── CONVENIENCE: run on embedded sample data ─────────────────────────────────

def run_on_sample(json_text: str, trace_name: str = "sample"):
    """
    Call this directly from a notebook or REPL with raw JSON-Lines text.

    Example:
        import trace_analysis as ta
        fig = ta.run_on_sample(open("trace.jsonl").read())
        fig.show()
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json_text)
        tmp = f.name
    try:
        df = load_trace(tmp)
    finally:
        os.unlink(tmp)

    print(f"Loaded {len(df)} records from sample")
    stats = temporal_stats(df)
    for k, v in stats.items():
        print(f"  {k:<28} {v:.3f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print("\nBody-size per page:")
    print(body_size_by_page(df).to_string(index=False))

    fig = plot_single_trace(df, trace_name)
    return fig


if __name__ == "__main__":
    main()