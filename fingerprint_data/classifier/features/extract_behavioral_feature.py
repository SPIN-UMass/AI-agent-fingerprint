"""
extract_features.py
================================================================================
Extracts ML-ready features from interactions.jsonl files for agent classification.

Each trial's interactions.jsonl is collapsed into a single fixed-width feature
vector of purely behavioral signals. One row per trial.

Features are derived from user-generated browser events inside the batch[]
arrays. The boundary rule is:
  INCLUDED: any event the browser fires in response to a user action
            (click, keydown, scroll, unload, pagehide, visibilitychange)
  EXCLUDED: events/metadata the browser generates autonomously
            (page load, sendBeacon flush counts, batch sizes)

Directory layout expected:
    <root>/
      autogen/
        trial-001/interactions.jsonl
        trial-002/interactions.jsonl
        ...
      skyvern/
        trial-001/interactions.jsonl
        ...
      claude/
        ...

Usage
-----
  python extract_features.py --root ./data --out features.csv
  python extract_features.py --root ./data/autogen --out autogen.csv --no-label
  python extract_features.py --root ./data --preview

Dependencies
------------
  pip install pandas numpy
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# -- PAGE LABEL NORMALISATION --------------------------------------------------
# Maps verbose HTML <title> strings to short consistent identifiers
PAGE_LABEL = {
    "Subscribe \u00b7 V1 Subscription Button":           "S1-subscribe-v1",
    "Subscribe \u00b7 V2 Subscription Button":           "S2-subscribe-v2",
    "Subscribe \u00b7 V3 Subscription Button":           "S3-subscribe-v3",
    "S2 \u00b7 Scroll Gate \u2014 UX Behavior Test Suite":    "S4-scroll-gate",
    "S3 \u00b7 Hover Reveal \u2014 UX Behavior Test Suite":   "S5-hover-reveal",
    "S4 \u00b7 DOM Mismatch \u2014 UX Behavior Test Suite":   "S6-dom-mismatch",
    "S5 \u00b7 Delayed Feedback \u2014 UX Behavior Test Suite": "S7-delayed-feedback",
}

# Pages that have text-input form components
SUBSCRIBE_PAGES = {"S1-subscribe-v1", "S2-subscribe-v2", "S3-subscribe-v3"}
# Pages without forms — keyboard activity here is unusual (User 5 fingerprint)
NON_FORM_PAGES  = {"S4-scroll-gate", "S5-hover-reveal", "S6-dom-mismatch",
                   "S7-delayed-feedback"}
# Ordered list for bitmask encoding
PAGE_BITS = [
    "S1-subscribe-v1", "S2-subscribe-v2", "S3-subscribe-v3",
    "S4-scroll-gate",  "S5-hover-reveal", "S6-dom-mismatch",
    "S7-delayed-feedback",
]


def short_page(title: str) -> str:
    return PAGE_LABEL.get(title, title.split("\u00b7")[-1].strip()[:30])


# -- PARSING -------------------------------------------------------------------

def load_jsonl(filepath: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse one interactions.jsonl file.

    Returns
    -------
    flushes : list of flush-level dicts  (top-level JSON objects per sendBeacon call)
    events  : flat list of event dicts, each annotated with flush metadata
    """
    flushes, events = [], []
    with open(filepath, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                flush = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"    [warn] {filepath}:{line_no} - JSON parse error: {e}")
                continue

            flush["_flush_idx"] = len(flushes)
            flushes.append(flush)

            page_title  = flush.get("page", "")
            page_short  = short_page(page_title)
            flush_idx   = flush["_flush_idx"]
            received_at = flush.get("receivedAt", "")

            for ev in flush.get("batch", []):
                ev = dict(ev)
                ev["_page_short"]  = page_short
                ev["_flush_idx"]   = flush_idx
                ev["_received_at"] = received_at
                events.append(ev)

    return flushes, events


def events_to_df(events: list[dict]) -> pd.DataFrame:
    """Flat list of event dicts -> sorted DataFrame with IEI column."""
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["ms"] = pd.to_numeric(df.get("ms"), errors="coerce")
    df = df.sort_values("ms").reset_index(drop=True)
    df["iei_ms"] = df["ms"].diff().fillna(0).clip(lower=0)
    return df


# -- HELPERS -------------------------------------------------------------------

def _mean(seq):
    return float(np.mean(seq)) if len(seq) > 0 else float("nan")

def _std(seq):
    return float(np.std(seq, ddof=1)) if len(seq) > 1 else float("nan")

def _median(seq):
    return float(np.median(seq)) if len(seq) > 0 else float("nan")

def nan0(v):
    """Return v if finite, else 0.0 -- safe ratio denominator fallback."""
    return v if (v is not None and np.isfinite(v)) else 0.0


# -- FEATURE EXTRACTION --------------------------------------------------------

def extract_features(flushes: list[dict], events: list[dict]) -> dict:
    """
    Collapse one trial's events into a single feature dict.

    Feature groups
    --------------
    A. Session-level counts & rates      (from interaction_analysis session_stats)
    B. Inter-event interval (IEI)        (discriminates passive vs active agents)
    C. Event-type ratios                 (key discriminators found in analysis)
    D. Timing / reaction times           (focus->input, mousedown->mouseup)
    E. Mouse behaviour                   (trajectory, click coordinates, targets)
    F. Keyboard behaviour                (Enter ratio, modifier ratio, key diversity)
    G. Scroll behaviour                  (depth, count)
    H. Page-level structural             (dwell time, pages visited)
    I. Task completion / app-events      (correct answers, submits, decoy clicks)
    J. Navigation style                  (tab clicks to move between pages)

    NOTE: flushes argument is accepted for API compatibility but intentionally
    unused -- sendBeacon/network metadata (flush counts, batch sizes) is
    excluded. Browser-generated events triggered by user actions (unload,
    pagehide, visibilitychange) are retained as behavioral signals.

    Excluded as non-user-triggered:
      - page(load): browser fires automatically when DOM is ready
      - n_flushes / events_per_flush: sendBeacon network metadata
    Retained as user-triggered:
      - page(unload/pagehide): user navigated away or closed tab
      - visibility(hidden): user switched tabs or minimised
      - scroll, resize: direct user physical actions
    """
    f: dict = {}
    df = events_to_df(events)

    for col in ["x", "y"]:
        if col not in df.columns:
            df[col] = np.nan

    if df.empty:
        return f

    # -- convenience boolean masks --------------------------------------------
    is_mm   = df["type"] == "mousemove"
    is_kd   = df["type"] == "keydown"
    is_cl   = df["type"] == "click"
    is_sc   = df["type"] == "scroll"
    is_inp  = df["type"] == "input"
    is_aev  = df["type"] == "app_event"
    is_foc  = df["type"] == "focus"
    is_md   = df["type"] == "mousedown"
    is_mu   = df["type"] == "mouseup"

    n_total = len(df)

    # -- A. Session-level counts & rates --------------------------------------
    # Exclude page/visibility lifecycle events from timing so duration reflects
    # only the span of user-generated events.
    # Exclude only page(load) from timing — it fires before any user action.
    # page(unload/pagehide) and visibility are user-initiated (navigation,
    # tab switching) so they ARE included in duration and IEI.
    page_load_mask = (df["type"] == "page") & (df.get("action", pd.Series(dtype=str)) == "load")
    user_events = df[~page_load_mask]
    if user_events.empty:
        return f

    t_start    = user_events["ms"].min()
    t_end      = user_events["ms"].max()
    duration_s = max((t_end - t_start) / 1000.0, 1e-6)

    f["n_events"]      = int((~page_load_mask).sum())  # all events except page(load)
    f["duration_s"]    = duration_s
    f["event_rate_hz"] = f["n_events"] / duration_s
    f["n_pages"]       = int(df["_page_short"].nunique())
    f["n_clicks"]      = int(is_cl.sum())
    f["n_keydowns"]    = int(is_kd.sum())
    f["n_mousemoves"]  = int(is_mm.sum())
    f["n_scrolls"]     = int(is_sc.sum())
    f["n_app_events"]  = int(is_aev.sum())
    f["n_inputs"]      = int(is_inp.sum())
    f["n_focus"]       = int(is_foc.sum())
    # visibility=hidden fires when user switches tabs / minimises window
    is_vis = (df["type"] == "visibility") & (df.get("state", pd.Series(dtype=str)) == "hidden")
    f["n_tab_switches"] = int(is_vis.sum())

    # -- B. Inter-event interval (IEI) ----------------------------------------
    # Computed only over user-generated events to avoid large gaps from
    # page load/unload events inflating the distribution.
    # IEI over all user events including unload/pagehide/visibility
    # (those are user-initiated). Only page(load) excluded.
    iei = user_events["iei_ms"].iloc[1:].dropna()
    iei = iei[iei > 0]
    f["iei_mean_ms"]    = _mean(iei)
    f["iei_median_ms"]  = _median(iei)
    f["iei_std_ms"]     = _std(iei)
    f["iei_p95_ms"]     = float(np.percentile(iei, 95)) if len(iei) > 0 else float("nan")
    # Burst ratio: fraction of IEIs < 50 ms (machine-speed action bursts)
    f["iei_burst_ratio"] = float((iei < 50).sum()) / len(iei) if len(iei) > 0 else 0.0

    # -- C. Event-type ratios -------------------------------------------------
    # These are the primary axes of the scatter plot and heatmap.
    f["mousemove_ratio"] = nan0(f["n_mousemoves"] / n_total)  # User 6 fingerprint
    f["keydown_ratio"]   = nan0(f["n_keydowns"]   / n_total)  # Users 2-4 vs 1,5,6
    f["scroll_ratio"]    = nan0(f["n_scrolls"]    / n_total)  # User 5 fingerprint
    f["input_ratio"]     = nan0(f["n_inputs"]     / n_total)
    f["click_ratio"]     = nan0(f["n_clicks"]     / n_total)
    f["app_event_ratio"] = nan0(f["n_app_events"] / n_total)

    # Keyboard-dominant vs mouse-dominant signed index [-1, +1]
    f["kbd_vs_mouse"] = (
        (f["n_keydowns"] - f["n_mousemoves"]) /
        max(f["n_keydowns"] + f["n_mousemoves"], 1)
    )

    # Per-page event-type proportions (mirrors the heatmap's page x event cells)
    for page_short in PAGE_BITS:
        page_df = df[df["_page_short"] == page_short]
        n_pg    = len(page_df)
        prefix  = page_short.replace("-", "_")
        for etype in ("mousemove", "keydown", "input", "scroll", "click"):
            f[f"{prefix}__{etype}_prop"] = (
                float((page_df["type"] == etype).sum()) / n_pg if n_pg > 0 else 0.0
            )
        f[f"{prefix}__visited"] = int(n_pg > 0)

    # Keydown on non-form pages -- User 5's unique fingerprint
    nonform_df = df[df["_page_short"].isin(NON_FORM_PAGES)]
    n_nf       = len(nonform_df)
    f["nonform_keydown_count"] = int((nonform_df["type"] == "keydown").sum())
    f["nonform_keydown_ratio"] = (
        f["nonform_keydown_count"] / n_nf if n_nf > 0 else 0.0
    )

    # -- D. Timing / reaction times -------------------------------------------

    # focus -> first input on same target (typing reaction latency)
    focus_rows = df[is_foc].copy()
    input_rows = df[is_inp].copy()
    react_times = []
    for _, frow in focus_rows.iterrows():
        tgt_prefix = str(frow.get("target", "")).split("[")[0]
        later = input_rows[
            (input_rows["ms"] > frow["ms"]) &
            (input_rows.get("target", pd.Series(dtype=str))
             .str.startswith(tgt_prefix, na=False))
        ]
        if not later.empty:
            react_times.append(later["ms"].iloc[0] - frow["ms"])
    f["focus_to_input_latency_mean_ms"] = _mean(react_times)
    f["focus_to_input_latency_std_ms"]  = _std(react_times)

    # mousedown -> mouseup hold time (click duration)
    md_rows = df[is_md][["ms"]].reset_index(drop=True)
    mu_rows = df[is_mu][["ms"]].reset_index(drop=True)
    hold_times = []
    for _, mrow in md_rows.iterrows():
        later_ups = mu_rows[mu_rows["ms"] > mrow["ms"]]
        if not later_ups.empty:
            hold_times.append(later_ups["ms"].iloc[0] - mrow["ms"])
    f["click_hold_mean_ms"] = _mean(hold_times)
    f["click_hold_std_ms"]  = _std(hold_times)

    # -- E. Mouse behaviour ---------------------------------------------------

    # Total trajectory length (sum of Euclidean distances between mousemove points)
    mm_df = df[is_mm].dropna(subset=["x", "y"])
    # print("\n=== DEBUG extract_features ===")
    # print("df type:", type(df))
    # print("df shape:", df.shape)
    # print("df columns:", df.columns.tolist())
    # print("is_mm count:", is_mm.sum())
    if len(mm_df) > 1:
        dx = mm_df["x"].diff().dropna()
        dy = mm_df["y"].diff().dropna()
        f["mouse_traj_total_px"] = float(np.sqrt(dx**2 + dy**2).sum())
    else:
        f["mouse_traj_total_px"] = 0.0
    f["mouse_traj_mean_px_per_page"] = f["mouse_traj_total_px"] / max(f["n_pages"], 1)

    # Click coordinate statistics (spatial precision / stereotypy)
    cl_df = df[is_cl].dropna(subset=["x", "y"])
    if not cl_df.empty:
        f["click_x_mean"] = float(cl_df["x"].mean())
        f["click_y_mean"] = float(cl_df["y"].mean())
        f["click_x_std"]  = float(cl_df["x"].std(ddof=1)) if len(cl_df) > 1 else 0.0
        f["click_y_std"]  = float(cl_df["y"].std(ddof=1)) if len(cl_df) > 1 else 0.0
    else:
        f["click_x_mean"] = f["click_y_mean"] = float("nan")
        f["click_x_std"]  = f["click_y_std"]  = float("nan")

    # Number of distinct click targets (breadth of interaction)
    f["n_unique_click_targets"] = int(
        df[is_cl]["target"].dropna().nunique()
    ) if "target" in df.columns else 0

    # -- F. Keyboard behaviour ------------------------------------------------
    kd_df = df[is_kd]
    if not kd_df.empty and "key" in kd_df.columns:
        keys = kd_df["key"].fillna("").str.strip()
        n_kd = len(keys)
        f["enter_key_count"]    = int((keys == "Enter").sum())
        f["enter_key_ratio"]    = f["enter_key_count"] / n_kd if n_kd > 0 else 0.0
        f["n_unique_keys"]      = int(keys.nunique())
        # Modifier usage
        mod_count = 0
        for mod in ("ctrl", "shift", "alt", "meta"):
            if mod in kd_df.columns:
                mod_count += int(kd_df[mod].fillna(False).astype(bool).sum())
        f["modifier_key_count"] = mod_count
        f["modifier_key_ratio"] = mod_count / n_kd if n_kd > 0 else 0.0
        # Key diversity: Shannon entropy of key distribution
        key_probs = keys.value_counts(normalize=True)
        f["key_entropy"] = float(-np.sum(key_probs * np.log2(key_probs + 1e-12)))
    else:
        for feat in ("enter_key_count", "enter_key_ratio", "n_unique_keys",
                     "modifier_key_count", "modifier_key_ratio", "key_entropy"):
            f[feat] = 0.0

    # -- G. Scroll behaviour --------------------------------------------------
    sc_df = df[is_sc]
    if not sc_df.empty:
        # scrollY may appear as "scrollY" or "scroll_y" depending on row origin
        sy = sc_df["scrollY"].dropna() if "scrollY" in sc_df.columns else pd.Series(dtype=float)
        if sy.empty and "scroll_y" in sc_df.columns:
            sy = sc_df["scroll_y"].dropna()
        f["max_scroll_depth_px"]  = float(sy.max())  if not sy.empty else 0.0
        f["scroll_depth_mean_px"] = float(sy.mean()) if not sy.empty else 0.0
    else:
        f["max_scroll_depth_px"]  = 0.0
        f["scroll_depth_mean_px"] = 0.0

    # -- H. Page-level structural ---------------------------------------------
    f["n_unique_pages"] = int(df["_page_short"].nunique())

    # Bitmask encoding of which pages were visited (7-bit integer)
    f["pages_visited_bitmask"] = sum(
        (1 << i)
        for i, pg in enumerate(PAGE_BITS)
        if pg in df["_page_short"].values
    )

    # Per-page dwell time (duration of user activity on that page)
    dwell_times = []
    for pg in df["_page_short"].unique():
        # Exclude page(load) from dwell timing; unload/pagehide/visibility
        # are user actions and bound the dwell window correctly.
        pg_load_mask = (
            (df["type"] == "page") &
            (df.get("action", pd.Series(dtype=str)) == "load")
        )
        pg_user = df[
            (df["_page_short"] == pg) &
            (~pg_load_mask)
        ]
        if len(pg_user) > 1:
            dwell = (pg_user["ms"].max() - pg_user["ms"].min()) / 1000.0
            dwell_times.append(dwell)
    f["page_dwell_mean_s"] = _mean(dwell_times)
    f["page_dwell_std_s"]  = _std(dwell_times)
    f["page_dwell_min_s"]  = float(min(dwell_times)) if dwell_times else float("nan")
    f["page_dwell_max_s"]  = float(max(dwell_times)) if dwell_times else float("nan")

    # -- I. Task completion / app-events --------------------------------------
    aev_df = df[is_aev].copy()
    if not aev_df.empty and "event" in aev_df.columns:
        ev_col  = aev_df["event"].fillna("")
        det_col = aev_df["detail"].fillna("")

        f["n_successful_submits"]    = int(det_col.str.contains("Subscribed|checkmark", regex=True).sum())
        f["n_blocked_submits"]       = int(det_col.str.contains("BLOCKED", na=False).sum())
        f["hover_reveal_correct"]    = int(det_col.str.contains("CORRECT|Hidden button", regex=True).sum() > 0)
        f["dom_mismatch_correct"]    = int(
            aev_df[ev_col == "select"]["detail"].fillna("")
            .str.contains("CORRECT|mars", regex=True).sum() > 0
        )
        f["dom_mismatch_wrong"]      = int(
            aev_df[ev_col == "select"]["detail"].fillna("")
            .str.contains("INCORRECT", na=False).sum() > 0
        )
        f["false_cue_clicks"]        = int(det_col.str.contains("NO EFFECT|False cue", regex=True).sum())
        f["n_hover_events"]          = int(ev_col.str.contains("hover", case=False, na=False).sum())
        f["n_select_events"]         = int((ev_col == "select").sum())
        f["delayed_feedback_retry"]  = int(det_col.str.contains("retry|Retry", regex=True).sum() > 0)
        f["delayed_feedback_completed"] = int(
            det_col.str.contains("completed|Completed|Done", regex=True).sum() > 0
        )
    else:
        for feat in ("n_successful_submits", "n_blocked_submits", "hover_reveal_correct",
                     "dom_mismatch_correct", "dom_mismatch_wrong", "false_cue_clicks",
                     "n_hover_events", "n_select_events",
                     "delayed_feedback_retry", "delayed_feedback_completed"):
            f[feat] = 0

    # -- J. Navigation style --------------------------------------------------

    # Inter-page gap: time between page(unload/pagehide) and next user event
    # on a new page. Both unload triggers are user-initiated (clicking a link,
    # closing a tab), so this gap reflects navigation decision speed.
    if "action" in df.columns:
        unloads = df[
            (df["type"] == "page") &
            (df["action"].isin(["unload", "pagehide"]))
        ]
        gaps = []
        for _, urow in unloads.iterrows():
            next_ev = user_events[user_events["ms"] > urow["ms"]]
            if not next_ev.empty:
                gaps.append(next_ev["ms"].iloc[0] - urow["ms"])
        f["inter_page_gap_mean_ms"] = _mean(gaps)
        f["inter_page_gap_std_ms"]  = _std(gaps)
    else:
        f["inter_page_gap_mean_ms"] = float("nan")
        f["inter_page_gap_std_ms"]  = float("nan")

    # load_to_first_interact: how quickly the user acted after page appeared.
    # page(load) fires automatically, but the TIME TO FIRST USER ACTION is a
    # behavioral signal — agents that plan before acting have higher latency.
    if "action" in df.columns:
        load_rows = df[(df["type"] == "page") & (df["action"] == "load")]
        latencies = []
        for _, lrow in load_rows.iterrows():
            pg = lrow.get("_page_short", "")
            interact = user_events[
                (user_events["_page_short"] == pg) &
                (user_events["ms"] > lrow["ms"])
            ]
            if not interact.empty:
                latencies.append(interact["ms"].iloc[0] - lrow["ms"])
        f["load_to_first_interact_mean_ms"] = _mean(latencies)
        f["load_to_first_interact_std_ms"]  = _std(latencies)
    else:
        f["load_to_first_interact_mean_ms"] = float("nan")
        f["load_to_first_interact_std_ms"]  = float("nan")

    # -- J. Navigation style --------------------------------------------------
    # nav-tab clicks are user-initiated page transitions (clicking the nav bar)
    f["nav_via_tab_clicks"] = int(
        df[is_cl]["target"].fillna("").str.contains("nav-tab", na=False).sum()
    ) if "target" in df.columns else 0

    return f


# -- PER-TRIAL ENTRY POINT ----------------------------------------------------

def process_trial(jsonl_path: Path) -> dict | None:
    """Load one trial file and return its feature dict, or None on failure."""
    try:
        flushes, events = load_jsonl(jsonl_path)
    except Exception as e:
        print(f"    [error] {jsonl_path}: {e}")
        return None

    if not events:
        print(f"    [warn] {jsonl_path}: no events found, skipping")
        return None

    feats = extract_features(flushes, events)
    feats["agent"] = jsonl_path.parent.parent.name
    feats["trial"] = jsonl_path.parent.name
    return feats


# -- DIRECTORY DISCOVERY ------------------------------------------------------

def discover_trials(root: Path, filename: str = "interactions.jsonl") -> list[tuple[str, Path]]:
    """
    Walk <root> and find all trial interaction files.

    Supports two layouts:
      Layout A:  <root>/<agent>/<trial>/interactions.jsonl
      Layout B:  <root>/<trial>/interactions.jsonl  (no agent subfolder)

    Returns list of (agent_label, filepath) tuples sorted by agent then trial.
    """
    results = []
    for fp in sorted(root.rglob(filename)):
        rel_parts = fp.relative_to(root).parts
        if len(rel_parts) == 3:      # agent / trial / filename
            agent = rel_parts[0]
        elif len(rel_parts) == 2:    # trial / filename (no agent folder)
            agent = root.name
        else:
            agent = "unknown"
        results.append((agent, fp))
    return results


# -- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract ML features from interactions.jsonl for agent classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Full dataset with multiple agents
  python extract_features.py --root ./data --out features.csv

  # Single agent folder, no label column
  python extract_features.py --root ./data/autogen --out autogen.csv --no-label

  # Preview feature names and shape only
  python extract_features.py --root ./data --preview
""")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--root",  metavar="DIR",
                     help="Root directory containing agent subfolders")
    grp.add_argument("--files", nargs="+",
                     help="Explicit list of .jsonl files")
    parser.add_argument("--filename", default="interactions.jsonl",
                        help="Filename inside each trial dir (default: interactions.jsonl)")
    parser.add_argument("--out",      default="behavioral_features.csv",
                        help="Output CSV path (default: features.csv)")
    parser.add_argument("--no-label", action="store_true",
                        help="Do not add 'agent' label column")
    parser.add_argument("--preview",  action="store_true",
                        help="Print feature names and counts; do not save")
    args = parser.parse_args()

    root = Path(args.root) if args.root else None

    if root:
        trials = discover_trials(root, filename=args.filename)
    else:
        import glob
        trials = [("unknown", Path(p))
                  for pat in args.files
                  for p in glob.glob(pat, recursive=True)]

    if not trials:
        print(f"[error] No '{args.filename}' files found.")
        raise SystemExit(1)

    print(f"\nFound {len(trials)} trial file(s)")

    rows = []
    for agent, fp in trials:
        rel = fp.relative_to(root) if root else fp
        print(f"  Processing [{agent}] {rel} ...", end=" ", flush=True)
        feats = process_trial(fp)
        if feats is None:
            print("skipped")
            continue
        if not args.no_label:
            feats["agent"] = agent
        rows.append(feats)
        print(f"OK  ({feats['n_events']} events, {feats['n_pages']} pages)")

    if not rows:
        print("\n[error] No valid trials processed.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)

    # Column order: identifiers first, label last
    label_cols = ["agent"] if not args.no_label else []
    id_cols    = [c for c in ("agent", "trial") if c in df.columns]
    feat_cols  = [c for c in df.columns if c not in id_cols + label_cols]
    df = df[id_cols + feat_cols + label_cols]

    if args.preview:
        print(f"\nFeature matrix shape: {df.shape}")
        groups = {
            "A. Session counts & rates":  [c for c in feat_cols if c in (
                "n_events","n_pages","n_clicks","n_keydowns","n_mousemoves",
                "n_scrolls","n_app_events","n_inputs","n_focus",
                "duration_s","event_rate_hz")],
            "B. IEI":                     [c for c in feat_cols if "iei" in c],
            "C. Event-type ratios":        [c for c in feat_cols if c.endswith("_ratio") and "iei" not in c and "nonform" not in c],
            "D. Timing / reaction":        [c for c in feat_cols if any(x in c for x in ("latency","hold","kbd_vs"))],
            "E. Mouse":                    [c for c in feat_cols if any(x in c for x in ("mouse","click_x","click_y","traj","unique_click"))],
            "F. Keyboard":                 [c for c in feat_cols if any(x in c for x in ("key","enter","modifier"))],
            "G. Scroll":                   [c for c in feat_cols if "scroll" in c],
            "H. Page structure":           [c for c in feat_cols if any(x in c for x in ("page_dwell","bitmask","unique_pages","n_pages"))],
            "I. Task completion":          [c for c in feat_cols if any(x in c for x in ("submit","hover_reveal","dom_mismatch","false_cue","select","delayed","hover_event"))],
            "J. Navigation":               [c for c in feat_cols if any(x in c for x in ("nav","inter_page","load_to_first","tab_switch"))],
            "L. Per-page x event props":   [c for c in feat_cols if "__" in c],
        }
        for grp_name, cols in groups.items():
            if cols:
                print(f"  {grp_name}: {len(cols)} features")
        print(f"\n  Total features: {len(feat_cols)}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Label  : {'agent' if not args.no_label else '(none)'}")

    groups = {
        "A. Session counts & rates":  [c for c in feat_cols if c in (
            "n_events","n_pages","n_clicks","n_keydowns","n_mousemoves",
            "n_scrolls","n_app_events","n_inputs","n_focus",
            "duration_s","event_rate_hz")],
        "B. IEI":                     [c for c in feat_cols if "iei" in c],
        "C. Event-type ratios":        [c for c in feat_cols if c.endswith("_ratio") and "iei" not in c and "nonform" not in c],
        "D. Timing / reaction":        [c for c in feat_cols if any(x in c for x in ("latency","hold","kbd_vs"))],
        "E. Mouse":                    [c for c in feat_cols if any(x in c for x in ("mouse","click_x","click_y","traj","unique_click"))],
        "F. Keyboard":                 [c for c in feat_cols if any(x in c for x in ("key","enter","modifier"))],
        "G. Scroll":                   [c for c in feat_cols if "scroll" in c],
        "H. Page structure":           [c for c in feat_cols if any(x in c for x in ("page_dwell","bitmask","unique_pages"))],
        "I. Task completion":          [c for c in feat_cols if any(x in c for x in ("submit","hover_reveal","dom_mismatch","false_cue","select","delayed","hover_event"))],
        "J. Navigation":               [c for c in feat_cols if any(x in c for x in ("nav","inter_page","load_to_first","tab_switch"))],
        "L. Per-page x event props":   [c for c in feat_cols if "__" in c],
    }
    print(f"  Features ({len(feat_cols)}):")
    for grp_name, cols in groups.items():
        if cols:
            print(f"    {grp_name}: {len(cols)}")
    print()


if __name__ == "__main__":
    main()