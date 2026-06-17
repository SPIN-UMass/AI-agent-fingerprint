"""
Agent Feature Extractor  —  agent_feature_extractor.py
========================================================
Derives 78 ML-ready features from requests.jsonl files for AI agent
identification. Extracts features from four layers:

  Layer 1 — TLS fingerprint     (JA4 A/B/C, cipher/curve/sig order,
                                  GREASE, PQ crypto, ALPN, versions)
  Layer 2 — HTTP/2 Akamai       (SETTINGS, WINDOW_UPDATE, pseudo-header
                                  order, priority tree growth)
  Layer 3 — HTTP headers        (header order per request type, Sec-Fetch
                                  consistency, Client Hints coherence,
                                  Priority header correctness)
  Layer 4 — Temporal & Spatial  (request rate, IRI stats, dwell times,
                                  subresource ratio, referer chain,
                                  connection reuse)

sendBeacon POST /collect filtering
-----------------------------------
POST /collect rows emitted by logger.js are instrumentation artifacts,
not genuine browsing traffic. They are removed using three rules:
  Rule 1: POST arrives within UNLOAD_WINDOW_MS of the next page GET
  Rule 2: Multiple /collect POSTs cluster within BURST_WINDOW_MS
  Rule 3: Referer ends in .html (logger.js always sets this)
Removed when Rule 1 AND (Rule 2 OR Rule 3).

Directory layout
----------------
The extractor reads a two-level tree where the agent name is the
parent directory and each trial is a subdirectory:

    autogen/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
    skyvern/
      trial-001/requests.jsonl
      trial-002/requests.jsonl

Output
------
  agent_features.csv   — one row per trial, all 78 features + agent + trial
                         columns. Direct input to LightGBM / sklearn.
  agent_features_summary.csv — per-agent mean ± std for every numeric feature.

Usage
-----
  # Root directory contains agent subdirectories
  python agent_feature_extractor.py --root ./data

  # Process specific agents only
  python agent_feature_extractor.py --root ./data --agents autogen skyvern

  # Custom requests filename inside each trial directory
  python agent_feature_extractor.py --root ./data --filename requests.jsonl

  # Skip sendBeacon filter (keep all traffic)
  python agent_feature_extractor.py --root ./data --no-filter

  # Output to a specific directory
  python agent_feature_extractor.py --root ./data --out ./features
"""

import re
import json
import glob
import hashlib
import argparse
import sys
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COLLECT_ENDPOINT = "/collect"
UNLOAD_WINDOW_MS = 500   # Rule 1: max ms gap between POST and next page GET
BURST_WINDOW_MS  = 50    # Rule 2: max ms gap between POSTs in same burst

# Lookup tables for human-readable names (used in string features)
CIPHER_NAMES = {
    4865: "TLS_AES_128_GCM_SHA256",      4866: "TLS_AES_256_GCM_SHA384",
    4867: "TLS_CHACHA20_POLY1305_SHA256", 49195:"ECDHE-ECDSA-AES128-GCM",
    49199:"ECDHE-RSA-AES128-GCM",        49196:"ECDHE-ECDSA-AES256-GCM",
    49200:"ECDHE-RSA-AES256-GCM",        52393:"ECDHE-ECDSA-CHACHA20",
    52392:"ECDHE-RSA-CHACHA20",          49171:"ECDHE-RSA-AES128-CBC",
    49172:"ECDHE-RSA-AES256-CBC",        156:  "RSA-AES128-GCM",
    157:  "RSA-AES256-GCM",              47:   "RSA-AES128-CBC",
    53:   "RSA-AES256-CBC",
}
CURVE_NAMES = {
    29: "x25519", 23: "secp256r1", 24: "secp384r1",
    25497: "x25519Kyber768", 4588: "X25519MLKEM768",
}
SIG_NAMES = {
    1027: "ecdsa_secp256r1_sha256", 2052: "rsa_pss_rsae_sha256",
    1025: "rsa_pkcs1_sha256",       1283: "ecdsa_secp384r1_sha384",
    2053: "rsa_pss_rsae_sha384",    1281: "rsa_pkcs1_sha384",
    2054: "rsa_pss_rsae_sha512",    1537: "rsa_pkcs1_sha512",
}
VER_NAMES = {772: "TLS1.3", 771: "TLS1.2", 770: "TLS1.1", 769: "TLS1.0"}

# Expected Priority values per request role
PRIORITY_EXPECTED = {
    "navigate":    {"u=0, i", "u=0,i"},
    "subresource": {"u=1", "u=2"},
    "fetch_xhr":   {"u=4, i", "u=4,i", "u=3, i", "u=3,i"},
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def is_grease(v: int) -> bool:
    """Return True if v is a GREASE value (RFC 8701: 0x?A?A pattern)."""
    return isinstance(v, int) and (v & 0x0F0F) == 0x0A0A


def sha16(s: str) -> str:
    """SHA-256 of string, first 16 hex chars — compact categorical fingerprint."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def modal(series: pd.Series):
    """Modal value of a Series; None if empty."""
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else None


def parse_ts(ts: str) -> datetime:
    """Parse ISO 8601 timestamp with nanosecond precision."""
    ts = re.sub(r"(\.\d{6})\d+", r"\1", ts)
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


# ─── SENBEACON FILTER ─────────────────────────────────────────────────────────

def filter_sendbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    collect_mask = (
        df["method"].eq("POST")
        & df["path"].eq("/collect")
    )

    n_removed = int(collect_mask.sum())

    return (
        df.loc[~collect_mask].reset_index(drop=True),
        n_removed,
    )


# EXPERIMENT_PAGES = {
#     "/", "/subscribe-v1.html", "/subscribe-v2.html", "/subscribe-v3.html",
#     "/s2-scroll-gate.html", "/s3-hover-reveal.html", "/s4-dom-mismatch.html",
#     "/s5-delayed-feedback.html",
# }

# def filter_sendbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
#     if df.empty:
#         return df, 0

#     # Resolve the content-type column name flexibly
#     ct_col = None
#     for candidate in ("content_type", "http_content_type", "Content-Type"):
#         if candidate in df.columns:
#             ct_col = candidate
#             break

#     def is_experiment_referer(referer) -> bool:
#         if not referer or not isinstance(referer, str):
#             return False
#         return urlparse(referer).path in EXPERIMENT_PAGES

#     def sec_fetch_consistent(row) -> bool:
#         mode = row.get("sec_fetch_mode", "") or ""
#         dest = row.get("sec_fetch_dest", "") or ""
#         site = row.get("sec_fetch_site", "") or ""
#         if mode and mode != "cors":
#             return False
#         if dest and dest != "empty":
#             return False
#         if site and site not in ("same-origin", "same-site"):
#             return False
#         return True

#     beacon_mask = (
#         df["method"].eq("POST")
#         & df["path"].eq(COLLECT_ENDPOINT)
#         & df["referer"].apply(is_experiment_referer)
#         & df.apply(sec_fetch_consistent, axis=1)
#     )

#     # Add content-type check only if the column was found
#     if ct_col:
#         beacon_mask &= df[ct_col].str.contains("application/json", na=False)

#     n_removed = int(beacon_mask.sum())
#     return df.loc[~beacon_mask].reset_index(drop=True), n_removed

# ─── RAW RECORD LOADER ────────────────────────────────────────────────────────

def load_requests(filepath: Path) -> pd.DataFrame:
    """
    Load one requests.jsonl into a flat DataFrame.
    Keeps all fields needed for feature extraction.
    Applies sendBeacon filter before returning.
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            http = rec.get("http") or {}
            tls  = rec.get("tls")  or {}
            h2   = rec.get("http2") or {}
            hdrs_ordered = http.get("headers_ordered") or []
            hdrs_dict    = http.get("headers") or {}

            def hval(name):
                v = hdrs_dict.get(name)
                return v[0] if v else ""

            rows.append({
                # ── identifiers ──────────────────────────────────────────
                "id":               rec.get("id"),
                "timestamp":        parse_ts(rec["timestamp"]),
                "source_port":      rec.get("source_port"),
                "proc_ms":          rec.get("processing_time_ms", 0.0),
                # ── request ──────────────────────────────────────────────
                "method":           http.get("method", ""),
                "path":             http.get("path", ""),
                "protocol":         http.get("protocol", ""),
                "referer":          http.get("referer", ""),
                "body_size":        http.get("body_size", 0),
                # ── TLS ───────────────────────────────────────────────────
                "tls_version":      tls.get("version", ""),
                "negotiated_proto": tls.get("negotiated_protocol", ""),
                "cipher_suite":     tls.get("cipher_suite", ""),
                "ja3":              tls.get("ja3_hash", ""),
                "ja4":              tls.get("ja4_hash", ""),
                "cipher_suites":    tls.get("cipher_suites_offered") or [],
                "elliptic_curves":  tls.get("elliptic_curves") or [],
                "sig_schemes":      tls.get("signature_schemes") or [],
                "supported_vers":   tls.get("supported_versions") or [],
                "alpn_protocols":   tls.get("alpn_protocols") or [],
                # ── HTTP/2 ────────────────────────────────────────────────
                "akamai_fp":        h2.get("akamai_fingerprint", ""),
                "pseudo_order":     h2.get("pseudo_header_order") or [],
                "h2_settings":      h2.get("settings") or {},
                "h2_window":        h2.get("window_update_size"),
                # ── HTTP header order ─────────────────────────────────────
                "hdr_order":        [h["name"] for h in hdrs_ordered],
                # ── Sec-Fetch ────────────────────────────────────────────
                "sf_mode":          http.get("sec_fetch_mode") or hval("Sec-Fetch-Mode"),
                "sf_dest":          http.get("sec_fetch_dest") or hval("Sec-Fetch-Dest"),
                "sf_site":          http.get("sec_fetch_site") or hval("Sec-Fetch-Site"),
                "sf_user":          http.get("sec_fetch_user") or hval("Sec-Fetch-User"),
                # ── Client Hints ─────────────────────────────────────────
                "ch_ua":            http.get("sec_ch_ua") or hval("Sec-Ch-Ua"),
                "ch_mobile":        http.get("sec_ch_ua_mobile") or hval("Sec-Ch-Ua-Mobile"),
                "ch_platform":      http.get("sec_ch_ua_platform") or hval("Sec-Ch-Ua-Platform"),
                "ua":               http.get("user_agent", ""),
                # ── Other headers ─────────────────────────────────────────
                "priority":         hval("Priority"),
                "accept":           http.get("accept", "") or hval("Accept"),
                "accept_encoding":  http.get("accept_encoding", "") or hval("Accept-Encoding"),
                "upgrade_insecure": hval("Upgrade-Insecure-Requests"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived columns needed for filter and feature extraction
    df["is_page"]    = df["path"].str.endswith(".html", na=False)
    df["is_collect"] = (df["method"] == "POST") & (df["path"] == COLLECT_ENDPOINT)
    df["is_subres"]  = (
        ~df["is_page"] & ~df["is_collect"] & (df["method"] == "GET")
    )
    df["is_fetch"]   = (
        (df["sf_mode"].isin(["cors", "same-origin"])) &
        (df["sf_dest"] == "empty") &
        (~df["is_collect"])
    )

    # Request type
    def req_type(row):
        if row["sf_mode"] == "navigate":
            return "navigate"
        if row["sf_mode"] == "no-cors":
            return f"subresource:{row['sf_dest'] or 'unknown'}"
        if row["is_fetch"]:
            return "fetch_xhr"
        if row["is_collect"]:
            return "collect"
        if row["path"].endswith(".html"):
            return "navigate"
        return "other"

    df["req_type"] = df.apply(req_type, axis=1)

    # Inter-request interval
    df["iri_ms"] = df["timestamp"].diff().dt.total_seconds().fillna(0) * 1000

    return df


# ─── JA4 PARSER ───────────────────────────────────────────────────────────────

def parse_ja4(ja4: str) -> dict:
    out = {k: None for k in [
        "ja4_a","ja4_b","ja4_c","ja4_protocol","ja4_version",
        "ja4_sni","ja4_cipher_count","ja4_ext_count","ja4_alpn_code",
    ]}
    if not ja4:
        return out
    parts = ja4.split("_")
    if len(parts) != 3:
        return out
    a, b, c = parts
    out.update({"ja4_a": a, "ja4_b": b, "ja4_c": c})
    if len(a) >= 10:
        out["ja4_protocol"]     = a[0]
        out["ja4_version"]      = a[1:3]
        out["ja4_sni"]          = a[3]
        out["ja4_cipher_count"] = int(a[4:6]) if a[4:6].isdigit() else None
        out["ja4_ext_count"]    = int(a[6:8]) if a[6:8].isdigit() else None
        out["ja4_alpn_code"]    = a[8:10]
    return out


# ─── AKAMAI PARSER ────────────────────────────────────────────────────────────

def parse_akamai(fp: str) -> dict:
    out = {k: None for k in [
        "ak_settings_str","ak_window_update","ak_pseudo_order",
        "ak_stream_count","ak_header_table","ak_enable_push",
        "ak_init_window","ak_max_header_list",
    ]}
    if not fp:
        return out
    secs = fp.split("|")
    out["ak_settings_str"]  = secs[0] if secs else None
    out["ak_window_update"] = int(secs[1]) if len(secs) > 1 and secs[1].isdigit() else None
    if len(secs) > 2 and secs[2]:
        out["ak_stream_count"] = len(secs[2].split(","))
    out["ak_pseudo_order"]  = secs[3] if len(secs) > 3 else None
    setting_map = {
        "1": "ak_header_table", "2": "ak_enable_push",
        "4": "ak_init_window",  "6": "ak_max_header_list",
    }
    for pair in (secs[0] if secs else "").split(";"):
        if ":" in pair:
            k, v = pair.split(":", 1)
            col = setting_map.get(k.strip())
            if col and v.strip().lstrip("-").isdigit():
                out[col] = int(v.strip())
    return out


# ─── SEQUENCE FEATURES ────────────────────────────────────────────────────────

def seq_features(seq: list, name: str, lookup: dict) -> dict:
    no_g   = [v for v in seq if not is_grease(v)]
    named  = [lookup.get(v, str(v)) for v in no_g]
    return {
        f"{name}_count":         len(no_g),
        f"{name}_has_grease":    any(is_grease(v) for v in seq),
        f"{name}_no_grease_str": "_".join(named),
        f"{name}_hash":          sha16("_".join(named)),
    }


# ─── CLIENT HINTS COHERENCE ───────────────────────────────────────────────────

CH_UA_RE = re.compile(r'"([^"]+)";v="(\d+)"')

def parse_ch_ua(s: str) -> list:
    return [(m.group(1), int(m.group(2))) for m in CH_UA_RE.finditer(s or "")]

def ua_family(ua: str) -> str:
    ua = ua or ""
    if "HeadlessChrome" in ua or "headless" in ua.lower(): return "headless-chrome"
    if "Edg/"  in ua: return "edge"
    if "Chrome/" in ua: return "chrome"
    if "Firefox/" in ua: return "firefox"
    if "Safari/" in ua and "Chrome" not in ua: return "safari"
    if "curl/" in ua: return "curl"
    if "python" in ua.lower(): return "python"
    if "Go-http-client" in ua: return "go"
    return "other"

def ua_os(ua: str) -> str:
    ua = ua or ""
    if "Windows" in ua: return "windows"
    if "Macintosh" in ua or "Mac OS" in ua: return "macos"
    if "Linux" in ua and "Android" not in ua: return "linux"
    if "Android" in ua: return "android"
    if "iPhone" in ua or "iPad" in ua: return "ios"
    return "other"

def ua_version(ua: str) -> str | None:
    for pat in [r"Chrome/(\d+)", r"Firefox/(\d+)"]:
        m = re.search(pat, ua or "")
        if m: return m.group(1)
    return None

def ch_incoherent(ch_ua: str, ch_mobile: str, ch_platform: str, ua: str) -> bool:
    """True if Sec-Ch-Ua and User-Agent are mutually inconsistent."""
    brands = [b.lower() for b, _ in parse_ch_ua(ch_ua)]
    ua = ua or ""
    # Headless in UA but not in CH-UA
    if "HeadlessChrome" in ua and not any("headless" in b for b in brands):
        return True
    # Mobile flag mismatch
    mobile_flag = (ch_mobile or "").strip() == "?1"
    ua_mobile   = any(kw in ua for kw in ["Mobile","Android","iPhone","iPad"])
    if mobile_flag != ua_mobile:
        return True
    return False


# ─── HEADER ORDER FEATURES ────────────────────────────────────────────────────

def header_order_hash(df: pd.DataFrame, req_type_val: str) -> str | None:
    """Modal header order hash for a given request type."""
    sub = df[df["req_type"] == req_type_val]["hdr_order"]
    if sub.empty:
        return None
    orders = sub.apply(lambda x: ",".join(x) if isinstance(x, list) else "")
    modal_order = orders.mode().iloc[0] if not orders.empty else ""
    return sha16(modal_order) if modal_order else None

def header_count_modal(df: pd.DataFrame, req_type_val: str) -> float | None:
    sub = df[df["req_type"] == req_type_val]["hdr_order"]
    if sub.empty:
        return None
    return float(sub.apply(len).median())


# ─── SEC-FETCH CONSISTENCY ────────────────────────────────────────────────────

def sf_combo_modal(df: pd.DataFrame, req_type_val: str) -> str | None:
    sub = df[df["req_type"] == req_type_val]
    if sub.empty:
        return None
    combo = (sub["sf_mode"] + "|" + sub["sf_dest"] + "|" + sub["sf_site"])
    return modal(combo)

def sf_has_violation(df: pd.DataFrame) -> bool:
    """True if any Sec-Fetch rule is violated across the session."""
    # navigate must have sf_user=?1
    nav = df[df["req_type"] == "navigate"]
    if not nav.empty and not (nav["sf_user"] == "?1").all():
        return True
    # non-navigate must NOT have sf_user
    non_nav = df[df["req_type"] != "navigate"]
    if not non_nav.empty and (non_nav["sf_user"] != "").any():
        return True
    # cors/fetch dest must be empty
    fetch = df[df["req_type"] == "fetch_xhr"]
    if not fetch.empty and not (fetch["sf_dest"] == "empty").all():
        return True
    return False

def sf_missing_rate(df: pd.DataFrame) -> float:
    """Fraction of non-collect requests with no Sec-Fetch-Mode."""
    non_col = df[~df["is_collect"]]
    if non_col.empty:
        return 0.0
    return float((non_col["sf_mode"] == "").mean())


# ─── PRIORITY CORRECTNESS ─────────────────────────────────────────────────────

def priority_modal(df: pd.DataFrame, req_type_val: str) -> str | None:
    sub = df[df["req_type"] == req_type_val]["priority"]
    return modal(sub)

def priority_correct_rate(df: pd.DataFrame) -> float:
    """Fraction of requests where Priority header matches expected value."""
    total, correct = 0, 0
    for rtype, expected in [
        ("navigate",    PRIORITY_EXPECTED["navigate"]),
        ("fetch_xhr",   PRIORITY_EXPECTED["fetch_xhr"]),
    ]:
        sub = df[df["req_type"] == rtype]["priority"]
        total   += len(sub)
        correct += int(sub.isin(expected).sum())
    return correct / total if total > 0 else 1.0


# ─── SPATIAL FEATURES ─────────────────────────────────────────────────────────

def subresource_ratio(df: pd.DataFrame) -> float:
    n_gets = (df["method"] == "GET").sum()
    if n_gets == 0:
        return 0.0
    return float(df["is_subres"].sum() / n_gets)

def referer_chain_intact(df: pd.DataFrame) -> bool:
    """True if each page GET has the correct Referer from the preceding page."""
    pages = df[df["is_page"]].reset_index(drop=True)
    if len(pages) < 2:
        return True
    for i in range(1, len(pages)):
        prev_path = pages.loc[i-1, "path"].lstrip("/")
        curr_ref  = pages.loc[i, "referer"] or ""
        if prev_path not in curr_ref:
            return False
    return True

def connection_reuse(df: pd.DataFrame) -> bool:
    """True if all requests share the same TCP source port."""
    ports = df["source_port"].dropna().unique()
    return len(ports) == 1

def page_sequence(df: pd.DataFrame) -> str:
    """Ordered sequence of page paths visited, as a string."""
    pages = df[df["is_page"]]["path"].tolist()
    return "->".join(p.lstrip("/") for p in pages)


# ─── TEMPORAL FEATURES ────────────────────────────────────────────────────────

def temporal_features(df: pd.DataFrame) -> dict:
    """
    Nine temporal features — all confirmed useful across analysis sessions.
    Excluded features and reasons:
      n_requests, n_pages    — controlled by experiment; identical across agents
      duration_s, iri_mean_ms— redundant with req_rate_hz (r≈-0.99)
      proc_mean_ms, proc_max — server-side; agent-independent
      page_dwell_max_s       — noisy; adds little over mean
      iri_cv                 — superseded by req_rate_hz_cv

    Per-trial features (7):
      req_rate_hz            requests / session_duration
      iri_median_ms          median inter-request interval — robust to
                             bimodal burst/wait IRI distributions
      iri_std_ms             std of IRI — captures timing variability;
                             NOT redundant with req_rate_hz
      page_dwell_mean_s      mean gap between consecutive page GETs —
                             directly reflects LLM processing latency
                             or human reading-time simulation
      page_dwell_std_s       std of page dwell — low=scripted fixed
                             timeouts, high=adaptive/human-like variance
      iri_page_to_subres_ms  mean gap from each page GET to its first
                             subresource fetch — reflects the browser
                             render pipeline latency; API crawlers that
                             skip subresources produce NaN here
      subresource_burst_ms   mean spread of subresource fetches per page
                             (max_ts - min_ts within each burst) —
                             tight bursts = parallel HTTP/2 fetching,
                             zero = no subresources (API crawler)

    Cross-trial features (2) — filled by add_cross_trial_temporal():
      req_rate_hz_median     median of req_rate_hz across all 30 trials;
                             robust to outlier sessions (e.g. near-zero
                             duration recording a 62.96 Hz spike)
      req_rate_hz_cv         std/mean of req_rate_hz across all trials;
                             claude_cu CV≈0.05 (mechanical pacing),
                             operator CV≈0.52 (task-dependent variability)
    """
    duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    iri = df["iri_ms"].iloc[1:]   # skip leading zero of first row

    req_rate = len(df) / duration if duration > 0 else 0.0
    iri_std  = float(iri.std())   if not iri.empty else 0.0

    # Page dwell: gap between consecutive page GETs only (excludes subresource bursts)
    page_ts = df[df["is_page"]]["timestamp"].sort_values()
    dwell   = page_ts.diff().dt.total_seconds().dropna()

    # iri_page_to_subres_ms: per page GET, find the first subresource fetch that
    # follows it and measure the gap. Captures browser render pipeline latency.
    subres_ts = df[df["is_subres"]]["timestamp"].sort_values()
    p2s_gaps  = []
    for pt in page_ts:
        after = subres_ts[subres_ts > pt]
        if not after.empty:
            p2s_gaps.append((after.iloc[0] - pt).total_seconds() * 1000)

    # subresource_burst_ms: for each page, collect all subresource timestamps
    # between this page GET and the next one, then measure max−min spread.
    pages_list = page_ts.tolist()
    burst_spreads = []
    for i, pt in enumerate(pages_list):
        next_pt = pages_list[i+1] if i+1 < len(pages_list) else df["timestamp"].iloc[-1]
        window  = subres_ts[(subres_ts > pt) & (subres_ts <= next_pt)]
        if len(window) >= 2:
            spread = (window.iloc[-1] - window.iloc[0]).total_seconds() * 1000
            burst_spreads.append(spread)

    return {
        # per-trial
        "req_rate_hz":            req_rate,
        "iri_median_ms":          float(iri.median())    if not iri.empty  else 0.0,
        "iri_std_ms":             iri_std,
        "page_dwell_mean_s":      float(dwell.mean())   if not dwell.empty else 0.0,
        "page_dwell_std_s":       float(dwell.std())    if not dwell.empty else 0.0,
        "iri_page_to_subres_ms":  float(np.mean(p2s_gaps))    if p2s_gaps     else float("nan"),
        "subresource_burst_ms":   float(np.mean(burst_spreads)) if burst_spreads else float("nan"),
        # cross-trial (filled later by add_cross_trial_temporal)
        "req_rate_hz_median":     None,
        "req_rate_hz_cv":         None,
    }


# ─── MAIN FEATURE EXTRACTOR ───────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> dict:
    """
    Extract all 78 features from a single trial's request DataFrame.
    Returns a flat dict suitable for one row in the output CSV.
    """
    feat = {}

    # ── Temporal ────────────────────────────────────────────────────
    temp = temporal_features(df)
    feat.update(temp)

    # ── Spatial ─────────────────────────────────────────────────────
    # feat["subresource_ratio"]    = subresource_ratio(df)
    # feat["referer_chain_intact"] = referer_chain_intact(df)
    # feat["n_unique_pages"]       = int(df[df["is_page"]]["path"].nunique())
    # feat["page_sequence_str"]    = page_sequence(df)
    # feat["connection_reuse"]     = connection_reuse(df)
    # feat["n_requests"]           = len(df)
    # feat["n_page_gets"]          = int(df["is_page"].sum())
    # feat["n_subresource_gets"]   = int(df["is_subres"].sum())

    return feat


# ─── DIRECTORY WALKER ─────────────────────────────────────────────────────────

def discover_trials(root: Path,
                    agents: list[str] | None,
                    filename: str) -> list[tuple[str, str, Path]]:
    """
    Walk root/agent/trial-NNN/filename and return (agent, trial, path) tuples.
    If agents is None, all immediate subdirectories of root are treated as agents.
    """
    trials = []
    agent_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and
         (agents is None or d.name in agents)]
    )
    for agent_dir in agent_dirs:
        for trial_dir in sorted(agent_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            fp = trial_dir / filename
            if fp.exists():
                trials.append((agent_dir.name, trial_dir.name, fp))
    return trials


# ─── CROSS-TRIAL TEMPORAL AGGREGATION ───────────────────────────────────────────

def add_cross_trial_temporal(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in req_rate_hz_median and req_rate_hz_cv for each row.

    These two features require all trials of the same agent to be
    collected first, then aggregated:

      req_rate_hz_median : median of req_rate_hz across all trials
                           of that agent. More robust than mean because
                           a single failed/short trial (like autogen
                           trial-15 at 62.96 Hz) inflates the mean
                           by 35× but barely moves the median.

      req_rate_hz_cv     : coefficient of variation = std / mean of
                           req_rate_hz across all trials of that agent.
                           Low CV (≈0.05) means mechanically consistent
                           pacing (e.g. claude_computer_use). High CV
                           (≈0.5) means variable, task-dependent timing
                           (e.g. operator). CV itself is a discriminating
                           feature independent of the absolute rate.
    """
    feat_df = feat_df.copy()
    for agent, grp in feat_df.groupby("agent"):
        rates  = grp["req_rate_hz"].dropna()
        median = float(rates.median()) if not rates.empty else 0.0
        cv     = float(rates.std() / rates.mean())                  if (not rates.empty and rates.mean() > 0) else 0.0
        feat_df.loc[feat_df["agent"] == agent, "req_rate_hz_median"] = median
        feat_df.loc[feat_df["agent"] == agent, "req_rate_hz_cv"]     = cv
    return feat_df


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract ML features from requests.jsonl for AI agent identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory layout expected:
  root/
    autogen/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
    skyvern/
      trial-001/requests.jsonl
      ...

Examples
--------
  python agent_feature_extractor.py --root ./data
  python agent_feature_extractor.py --root ./data --agents autogen skyvern
  python agent_feature_extractor.py --root ./data --out ./features
  python agent_feature_extractor.py --root ./data --no-filter
""")
    parser.add_argument("--root",      required=True,
                        help="root directory containing agent subdirectories")
    parser.add_argument("--agents",    nargs="+", default=None,
                        help="agent names to process (default: all subdirs)")
    parser.add_argument("--filename",  default="requests.jsonl",
                        help="requests filename inside each trial dir (default: requests.jsonl)")
    parser.add_argument("--out",       default=".",
                        help="output directory (default: .)")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip sendBeacon POST /collect filter")
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    do_filter = not args.no_filter

    # ── Discover all trial files ───────────────────────────────────────────────
    trials = discover_trials(root, args.agents, args.filename)
    if not trials:
        print(f"No {args.filename} files found under {root}. Exiting.")
        sys.exit(1)
    print(f"\nFound {len(trials)} trial(s) across "
          f"{len(set(a for a,_,_ in trials))} agent(s)\n")

    # ── Extract features per trial ────────────────────────────────────────────
    rows = []
    filter_log = []

    for agent, trial, fp in trials:
        df_raw = load_requests(fp)
        if df_raw.empty:
            print(f"  [warn] {agent}/{trial} — empty or unparseable, skipped")
            continue

        n_raw = len(df_raw)
        if do_filter:
            df, n_removed = filter_sendbeacon(df_raw)
            filter_log.append({
                "agent": agent, "trial": trial,
                "n_raw": n_raw, "n_removed": n_removed,
                "n_kept": n_raw - n_removed,
            })
            if n_removed:
                print(f"  {agent}/{trial}: {n_raw} → {n_raw-n_removed} reqs "
                      f"(removed {n_removed} sendBeacon POST(s))")
            else:
                print(f"  {agent}/{trial}: {n_raw} requests")
        else:
            df = df_raw
            print(f"  {agent}/{trial}: {n_raw} requests (no filter)")

        if df.empty:
            print(f"    [warn] {agent}/{trial} — empty after filtering, skipped")
            continue

        try:
            feat = extract_features(df)
        except Exception as e:
            print(f"    [error] {agent}/{trial} — feature extraction failed: {e}")
            continue

        feat["agent"] = agent
        feat["trial"] = trial
        rows.append(feat)

    if not rows:
        print("No features extracted. Exiting.")
        sys.exit(1)

    # ── Build feature DataFrame ────────────────────────────────────────────────
    feat_df = pd.DataFrame(rows)

    # Build the base DataFrame, then fill cross-trial temporal aggregates
    feat_df = pd.DataFrame(rows)
    feat_df = add_cross_trial_temporal(feat_df)

    # Move agent and trial to first two columns
    cols = ["agent", "trial"] + [c for c in feat_df.columns
                                  if c not in ("agent", "trial")]
    feat_df = feat_df[cols]

    # Sort by agent then trial
    feat_df = feat_df.sort_values(["agent", "trial"]).reset_index(drop=True)

    # ── Save filter report ────────────────────────────────────────────────────
    if do_filter and filter_log:
        pd.DataFrame(filter_log).to_csv(out_dir / "filter_report.csv", index=False)
        total_removed = sum(r["n_removed"] for r in filter_log)
        print(f"\n  Total sendBeacon rows removed: {total_removed}")
        print(f"  Saved: {out_dir / 'filter_report.csv'}")

    # ── Save features ─────────────────────────────────────────────────────────
    out_path = out_dir / "temporal_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  "
          f"({len(feat_df)} rows × {len(feat_df.columns)} columns)")

    # ── Per-agent summary (mean ± std for numeric features) ───────────────────
    numeric_cols = feat_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        summary = feat_df.groupby("agent")[numeric_cols].agg(["mean","std"]).round(4)
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.to_csv(out_dir / "temporal_features_summary.csv")
        print(f"  Saved: {out_dir / 'temporal_features_summary.csv'}")

    # ── Console: feature count and agent breakdown ────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Features extracted: {len(feat_df.columns) - 2} "
          f"(excluding agent, trial columns)")
    print(f"\n  Trials per agent:")
    for agent, grp in feat_df.groupby("agent"):
        print(f"    {agent:<30} {len(grp):>3} trial(s)")

    print(f"\n  Feature groups:")
    groups = {
        "Temporal":              [c for c in feat_df.columns if any(c.startswith(p)
                                   for p in ("req_rate","iri_","page_dwell",
                                             "subresource_burst"))],
        # "Spatial":               [c for c in feat_df.columns if any(c in
        #                            ("subresource_ratio","referer_chain_intact",
        #                             "n_unique_pages","page_sequence_str",
        #                             "connection_reuse","n_requests",
        #                             "n_page_gets","n_subresource_gets") for c in [c])],
    }
    for gname, gcols in groups.items():
        if gcols:
            print(f"    {gname:<25} {len(gcols):>3} features")

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()