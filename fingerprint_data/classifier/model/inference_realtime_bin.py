"""
realtime_inference.py
---------------------
Real-time agent classification evaluated as a time series.

requests.jsonl and interactions.jsonl are merged into a single
millisecond-resolution event stream. Inference runs at two granularities:

  Page checkpoints   — after every GET *.html navigation
  Event checkpoints  -- after every N interaction events (default N=25)

Both checkpoint types share the same pipeline:
  1. Slice  — all events up to this moment
  2. Extract — call each feature extractor on the slice
  3. Align  — apply training-time encoding transforms
  4. Predict — ExtraTrees models (temporal / http / tls / behavioral / combined)
  5. Save   — append to the two output CSVs

Output files
------------
  page_checkpoints.csv
      One row per page navigation. Columns per model:
        <ft>_prediction, <ft>_confidence, <ft>_proba_json (full distribution)

  event_checkpoints.csv
      One row per N-event sub-page checkpoint. Same schema as above, plus
        checkpoint_type  ("page" | "event")
        event_index      (cumulative interaction events seen)

Usage
-----
  python realtime_inference.py \\
      --requests     requests.jsonl \\
      --interactions interactions.jsonl \\
      --model-dir    /path/to/models \\
      --extractor-dir /path/to/extractors \\
      --event-interval 25 \\
      --save-dir     ./results
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
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
FEATURE_TYPES = ["temporal", "http", "tls", "behavioral", "combined"]
AGENTS        = [
    "autogen_websurfer", "browser_use", "claude_computer_use",
    "gemini_computer_use", "operator", "skyvern",
]
AGENT_SHORT = {
    "autogen_websurfer":   "autogen",
    "browser_use":         "browser",
    "claude_computer_use": "claude",
    "gemini_computer_use": "gemini",
    "operator":            "operator",
    "skyvern":             "skyvern",
}

# Label-encoder maps — must match preprocess.py training-time encodings
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


# ── timestamp helpers ──────────────────────────────────────────────────────────

def parse_ts(ts: str) -> datetime:
    ts = re.sub(r"(\.\d{6})\d+", r"\1", ts).replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


# ── unified event stream ───────────────────────────────────────────────────────

class Event:
    __slots__ = ("ts", "kind", "source_record", "page")
    def __init__(self, ts, kind, source_record, page=""):
        self.ts            = ts
        self.kind          = kind       # "request" | "interaction"
        self.source_record = source_record
        self.page          = page


def build_event_stream(requests_path: Path,
                       interactions_path: Path) -> list[Event]:
    """
    Merge both JSONL files into a single timestamp-sorted Event list.
    Each HTTP request → one Event.
    Each individual browser event inside an interactions batch → one Event
    (carrying its parent batch record for later reconstruction).
    """
    events: list[Event] = []

    with open(requests_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(Event(parse_ts(rec["timestamp"]), "request", rec))

    with open(interactions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batch_rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            page = batch_rec.get("page", "")
            for ev in batch_rec.get("batch", []):
                if "t" not in ev:
                    continue
                try:
                    ts = parse_ts(ev["t"])
                except Exception:
                    continue
                events.append(Event(ts, "interaction", batch_rec, page))

    events.sort(key=lambda e: e.ts)
    return events


# ── checkpoint detection ───────────────────────────────────────────────────────

def find_checkpoints(events: list[Event],
                     event_interval: int) -> list[dict]:
    """
    Return a list of checkpoint dicts, sorted by stream position.

    Two kinds:
      {"type": "page",  "ev_idx": int, "page_path": str, "ts": datetime,
       "n_int_events": int}
      {"type": "event", "ev_idx": int, "page_path": str, "ts": datetime,
       "n_int_events": int, "event_index": int}

    Event checkpoints fire every `event_interval` cumulative interaction
    events. Page checkpoints fire at every GET *.html.
    Both kinds share the same ev_idx (index into `events`) so slicing is
    identical.
    """
    checkpoints = []
    n_int      = 0          # cumulative interaction events seen
    last_event_cp = 0       # last n_int at which an event-checkpoint fired
    current_page  = ""

    for i, ev in enumerate(events):

        if ev.kind == "request":
            http = ev.source_record.get("http", {})
            if http.get("method") == "GET" and http.get("path", "").endswith(".html"):
                current_page = http["path"]
                checkpoints.append({
                    "type":        "page",
                    "ev_idx":      i,
                    "page_path":   current_page,
                    "ts":          ev.ts,
                    "n_int_events": n_int,
                })
                # Reset event-checkpoint counter at each new page
                last_event_cp = n_int

        elif ev.kind == "interaction":
            n_int += 1
            if (n_int - last_event_cp) >= event_interval:
                checkpoints.append({
                    "type":        "event",
                    "ev_idx":      i,
                    "page_path":   current_page,
                    "ts":          ev.ts,
                    "n_int_events": n_int,
                    "event_index": n_int,
                })
                last_event_cp = n_int

    return checkpoints


# ── slice reconstruction ───────────────────────────────────────────────────────

def slice_to_tmpfiles(events: list[Event],
                      end_idx: int) -> tuple[str, str]:
    """
    Write all events[0:end_idx+1] into two temporary JSONL files
    (requests, interactions) and return their paths.

    Interaction events from the same source batch are regrouped; only
    events whose timestamp falls within the window are included, so the
    behavioral extractor sees a realistic partial batch.
    """
    window = events[: end_idx + 1]

    req_records = [ev.source_record
                   for ev in window if ev.kind == "request"]

    # Regroup interaction events by parent batch record
    batch_groups: dict[int, tuple[dict, list]] = {}
    for ev in window:
        if ev.kind != "interaction":
            continue
        key = id(ev.source_record)
        if key not in batch_groups:
            batch_groups[key] = (ev.source_record, [])
        # Match the individual event inside the batch by timestamp
        for orig_ev in ev.source_record.get("batch", []):
            if "t" not in orig_ev:
                continue
            try:
                orig_ts = parse_ts(orig_ev["t"])
            except Exception:
                continue
            if abs((orig_ts - ev.ts).total_seconds()) < 0.001:
                batch_groups[key][1].append(orig_ev)
                break

    int_records = []
    for batch_rec, ev_list in batch_groups.values():
        if not ev_list:
            continue
        new_rec = {k: v for k, v in batch_rec.items() if k != "batch"}
        new_rec["batch"]      = ev_list
        new_rec["eventCount"] = len(ev_list)
        int_records.append(new_rec)

    tmp_req = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="rt_req_")
    tmp_int = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="rt_int_")
    for rec in req_records:
        tmp_req.write(json.dumps(rec) + "\n")
    for rec in int_records:
        tmp_int.write(json.dumps(rec) + "\n")
    tmp_req.close()
    tmp_int.close()
    return tmp_req.name, tmp_int.name


# ── feature extraction ─────────────────────────────────────────────────────────

def extract_all(req_path: str, int_path: str) -> dict[str, dict]:
    """Run all four extractors; return raw feature dicts keyed by type."""
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
        flushes, ev_list = beh_ext.load_jsonl(ip)
        raw["behavioral"] = beh_ext.extract_features(flushes, ev_list)
    except Exception:
        raw["behavioral"] = {}

    return raw


# ── feature alignment ──────────────────────────────────────────────────────────

def _encode_http(raw_http: dict, feature_cols: list[str]) -> dict:
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


def align_features(raw: dict[str, dict],
                   feat_type: str,
                   feature_cols: list[str]) -> pd.DataFrame:
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

def infer(raw: dict[str, dict],
          model_dir: Path) -> dict[str, dict]:
    """Run all five models. Returns {feat_type: result_dict}."""
    results = {}
    for ft in FEATURE_TYPES:
        mp = model_dir / f"extratrees_{ft}.joblib"
        if not mp.exists():
            continue
        try:
            clf, le, feature_cols = joblib.load(mp)
            X     = align_features(raw, ft, feature_cols)
            pred  = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]
            top2  = sorted(zip(le.classes_, proba),
                           key=lambda x: -x[1])[:2]
            results[ft] = {
                "predicted_agent": le.inverse_transform([pred])[0],
                "confidence":      round(float(proba[pred]), 3),
                "top2":            [(a, round(float(p), 3)) for a, p in top2],
                "proba_all":       {a: round(float(p), 4)
                                    for a, p in zip(le.classes_, proba)},
            }
        except Exception as e:
            results[ft] = {"predicted_agent": "error", "confidence": 0.0,
                           "top2": [], "proba_all": {}}
    return results


# ── CSV row builder ────────────────────────────────────────────────────────────

def build_csv_row(cp: dict, results: dict[str, dict],
                  start_ts: datetime, n_req: int) -> dict:
    """
    Build a flat dict suitable for one CSV row.

    Columns
    -------
    Metadata
        checkpoint_type, page_index (cumulative pages seen),
        event_index (cumulative interaction events seen),
        page_path, elapsed_s, wall_time, n_requests, n_int_events

    Per model (×5)
        <ft>_prediction   : predicted agent name
        <ft>_confidence   : probability of predicted class (float)
        <ft>_proba_json   : JSON string of full {agent: prob} distribution
        <ft>_prob_<agent> : individual probability column for each of 6 agents
    """
    elapsed = (cp["ts"] - start_ts).total_seconds()
    row = {
        "checkpoint_type": cp["type"],
        "page_index":      cp.get("page_index", ""),
        "event_index":     cp.get("event_index", ""),
        "page_path":       cp["page_path"],
        "elapsed_s":       round(elapsed, 3),
        "wall_time":       cp["ts"].strftime("%H:%M:%S.%f")[:-3],
        "n_requests":      n_req,
        "n_int_events":    cp["n_int_events"],
    }
    for ft in FEATURE_TYPES:
        res = results.get(ft, {})
        row[f"{ft}_prediction"] = res.get("predicted_agent", "")
        row[f"{ft}_confidence"] = res.get("confidence", "")
        row[f"{ft}_proba_json"] = json.dumps(res.get("proba_all", {}))
        for agent in AGENTS:
            col = f"{ft}_prob_{AGENT_SHORT[agent]}"
            row[col] = res.get("proba_all", {}).get(agent, "")
    return row


# ── console output ─────────────────────────────────────────────────────────────

def _bar(p: float, w: int = 10) -> str:
    return "█" * round(p * w) + "░" * (w - round(p * w))


def print_checkpoint(cp: dict, results: dict[str, dict],
                     start_ts: datetime, n_req: int) -> None:
    elapsed = (cp["ts"] - start_ts).total_seconds()
    kind    = cp["type"].upper()
    page    = cp["page_path"].lstrip("/")
    ts_str  = cp["ts"].strftime("%H:%M:%S.%f")[:-3]
    n_ev    = cp["n_int_events"]

    print(f"\n{'─'*70}")
    if cp["type"] == "page":
        print(f"  [{kind}] Page {cp.get('page_index','')}  |  {page}")
    else:
        print(f"  [{kind}] {page}  |  event #{cp.get('event_index','')}")
    print(f"  {ts_str}  (+{elapsed:.1f}s)  "
          f"|  {n_req} requests  |  {n_ev} interaction events")
    print(f"{'─'*70}")

    print(f"\n  {'Model':<12}  {'Prediction':<24}  {'Conf':>5}  "
          f"Probability distribution")
    print(f"  {'─'*65}")

    votes = []
    for ft in FEATURE_TYPES:
        res   = results.get(ft, {})
        pred  = res.get("predicted_agent", "n/a")
        conf  = res.get("confidence", 0.0)
        proba = res.get("proba_all", {})
        bars  = "  ".join(
            f"{AGENT_SHORT[a]}: {_bar(proba.get(a, 0))}"
            for a in AGENTS
        )
        print(f"  {ft:<12}  {pred:<24}  {conf:>5.3f}  {bars}")
        if pred not in ("n/a", "error"):
            votes.append(pred)

    if votes:
        vc  = Counter(votes)
        con, n = vc.most_common(1)[0]
        print(f"\n  Consensus: {con}  [{n}/{len(votes)} models]  "
              + "  ".join(f"{AGENT_SHORT.get(a,a)}×{c}"
                          for a, c in vc.most_common()))


def print_summary(page_rows: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  TIME-SERIES SUMMARY  (combined model — page checkpoints only)")
    print(f"{'='*70}")
    hdr = (f"  {'Page':<30} {'Time':>8}  {'Req':>3}  {'Ev':>4}  "
           f"{'Prediction':<24} {'Conf':>5}")
    print(hdr)
    print(f"  {'─'*65}")
    for r in page_rows:
        label = r["page_path"].lstrip("/")[:29]
        res   = {}
        # Decode the combined proba_json back for display
        try:
            proba = json.loads(r.get("combined_proba_json", "{}"))
        except Exception:
            proba = {}
        pred = r.get("combined_prediction", "")
        conf = r.get("combined_confidence", "")
        print(f"  {label:<30} {r['wall_time']:>8}  "
              f"{r['n_requests']:>3}  {r['n_int_events']:>4}  "
              f"{pred:<24} {conf:>5}")


# ── main pipeline ──────────────────────────────────────────────────────────────

def run(requests_path:     Path,
        interactions_path: Path,
        model_dir:         Path,
        extractor_dir:     str | None,
        event_interval:    int,
        save_dir:          str | None) -> None:

    # Add extractor scripts to import path
    for d in [extractor_dir, str(requests_path.parent), str(SCRIPT_DIR)]:
        if d and d not in sys.path:
            sys.path.insert(0, d)

    # ── build unified stream ───────────────────────────────────────────────────
    print("Building unified event stream …")
    stream = build_event_stream(requests_path, interactions_path)
    n_req_total = sum(1 for e in stream if e.kind == "request")
    n_int_total = sum(1 for e in stream if e.kind == "interaction")
    print(f"  {len(stream)} events total  "
          f"({n_req_total} requests  +  {n_int_total} interaction events)")

    checkpoints = find_checkpoints(stream, event_interval)

    # Number page checkpoints sequentially
    page_counter = 0
    for cp in checkpoints:
        if cp["type"] == "page":
            page_counter += 1
            cp["page_index"] = page_counter

    n_page_cps  = sum(1 for cp in checkpoints if cp["type"] == "page")
    n_event_cps = sum(1 for cp in checkpoints if cp["type"] == "event")
    print(f"  {n_page_cps} page checkpoints  +  "
          f"{n_event_cps} event checkpoints  "
          f"(every {event_interval} interaction events)\n")

    start_ts   = stream[0].ts
    page_rows  = []    # for summary table
    all_rows   = []    # all checkpoints (both types)

    # ── process each checkpoint ────────────────────────────────────────────────
    for cp in checkpoints:
        req_tmp, int_tmp = slice_to_tmpfiles(stream, cp["ev_idx"])
        try:
            n_req = sum(1 for e in stream[:cp["ev_idx"]+1]
                        if e.kind == "request")
            raw     = extract_all(req_tmp, int_tmp)
            results = infer(raw, model_dir)

            # Console
            print_checkpoint(cp, results, start_ts, n_req)

            # Build CSV row
            row = build_csv_row(cp, results, start_ts, n_req)
            all_rows.append(row)
            if cp["type"] == "page":
                page_rows.append(row)

        finally:
            for p in [req_tmp, int_tmp]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ── summary table ──────────────────────────────────────────────────────────
    print_summary(page_rows)

    # ── save CSVs ──────────────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # All checkpoints (both types) → one file
        all_df = pd.DataFrame(all_rows)
        all_path = os.path.join(save_dir, "all_checkpoints.csv")
        all_df.to_csv(all_path, index=False)
        print(f"\n  Saved all checkpoints  → {all_path}  "
              f"({len(all_df)} rows × {len(all_df.columns)} columns)")

        # Page-only checkpoints → separate file for quick analysis
        page_df = all_df[all_df["checkpoint_type"] == "page"].copy()
        page_path = os.path.join(save_dir, "page_checkpoints.csv")
        page_df.to_csv(page_path, index=False)
        print(f"  Saved page checkpoints → {page_path}  "
              f"({len(page_df)} rows)")

        # Event-only checkpoints → separate file
        event_df = all_df[all_df["checkpoint_type"] == "event"].copy()
        event_path = os.path.join(save_dir, "event_checkpoints.csv")
        event_df.to_csv(event_path, index=False)
        print(f"  Saved event checkpoints→ {event_path}  "
              f"({len(event_df)} rows)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Time-series agent inference from raw browsing trace files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--requests",        required=True,
                   help="Path to requests.jsonl")
    p.add_argument("--interactions",    required=True,
                   help="Path to interactions.jsonl")
    p.add_argument("--model-dir",       default=str(SCRIPT_DIR),
                   help="Directory containing extratrees_*.joblib files")
    p.add_argument("--extractor-dir",   default=None,
                   help="Directory containing extract_*_feature.py scripts")
    p.add_argument("--event-interval",  type=int, default=25,
                   help="Trigger an event checkpoint every N interaction "
                        "events within a page (default: 25)")
    p.add_argument("--save-dir",        default=None,
                   help="Directory to write output CSVs "
                        "(all_checkpoints.csv, page_checkpoints.csv, "
                        "event_checkpoints.csv)")
    args = p.parse_args()

    run(
        requests_path     = Path(args.requests),
        interactions_path = Path(args.interactions),
        model_dir         = Path(args.model_dir),
        extractor_dir     = args.extractor_dir,
        event_interval    = args.event_interval,
        save_dir          = args.save_dir,
    )