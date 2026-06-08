"""
HTTP Header Fingerprint Analyzer  —  http_header_analysis.py
==============================================================
Analyzes HTTP header fingerprints from requests.jsonl files collected
across multiple trials, with the goal of distinguishing AI agents by
their HTTP client identity.

sendBeacon POST /collect traffic (logger.js instrumentation artifact) is
removed by default using the same three-rule filter as tls_analysis.py.

Directory layout (same as tls_analysis.py and trace_analysis.py):
    traces/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
      ...

What is analyzed
----------------
Module 1 — Header order fingerprint
    The ordered sequence of header names is the strongest HTTP-layer signal.
    Different TLS/HTTP clients (curl, python-requests, playwright, real Chrome,
    headless Chrome) produce the same headers in different canonical orders.
    We fingerprint this per request type (navigation / subresource / XHR/fetch).

    Outputs:
      • header_order_str       : canonical comma-separated header name sequence
      • header_order_hash      : SHA-256(order_str)[:16] — compact fingerprint
      • n_headers              : total header count
      • has_header_{name}      : boolean presence of key discriminating headers

Module 2 — Sec-Fetch-* suite consistency
    Chrome's Sec-Fetch-* headers encode the request's fetch context.
    The combination (Mode, Dest, Site, User) must follow strict rules
    depending on request type. Automation frameworks frequently get this wrong.

    Request type classification:
      navigate   : Mode=navigate, Dest=document, User=?1
      subresource: Mode=no-cors,  Dest=script|style|image|font
      cors_fetch : Mode=cors,     Dest=empty
      same_origin: Site=same-origin
      cross_origin: Site=cross-site

    Consistency rules checked:
      • navigate  → Dest must be document
      • navigate  → Site must be none or same-origin
      • navigate  → User must be ?1
      • cors/fetch → Dest must be empty
      • script    → Mode must be no-cors or cors
      • Sec-Fetch-User must not appear on non-navigation requests

Module 3 — Client Hints coherence
    Sec-Ch-Ua, Sec-Ch-Ua-Mobile, Sec-Ch-Ua-Platform must be mutually
    consistent with each other and with the User-Agent string.
    Inconsistencies reveal spoofed or synthetic UA strings.

    Checks:
      • Brand tokens in Sec-Ch-Ua parsed and compared to UA string
      • Version numbers cross-checked between Sec-Ch-Ua and UA
      • Headless Chrome detection (HeadlessChrome brand token)
      • Mobile flag vs UA mobile keyword consistency
      • Platform token vs UA OS keyword consistency

Module 4 — Priority header analysis
    Chrome's Priority header encodes fetch urgency and incrementality.
    The value must match the request's role in the page:
      u=0, i  → navigation (highest urgency, incremental)
      u=1     → render-blocking subresource
      u=2     → high-priority prefetch
      u=3     → medium
      u=4, i  → background async (XHR, beacon, fetch)

    Agents that emit wrong Priority values for a given request type
    are immediately identifiable.

Module 5 — User-Agent string analysis
    Parsed into: browser family, version, OS, rendering engine.
    Headless Chrome has a distinctive UA pattern.

Module 6 — Cross-trace stability & discriminative power
    For every extracted feature, compute:
      • stability   : fraction of traces where value == modal value (1=identical)
      • entropy     : Shannon entropy across traces (higher = more discriminative)
    Features are ranked by entropy to identify the best agent-distinguishing signals.

Module 7 — Request-type profile
    Distribution of request types per trace. Agents that browse differently
    (e.g. skip subresource loading, have unusual cors/navigate ratios) are
    detectable here.

Outputs
-------
  http_header_summary.csv         — per-request parsed features, all traces
  http_header_stability.csv       — per-feature stability & entropy
  http_header_fingerprint_vector.csv — one row per trace, feature vector
  http_consistency_violations.csv — per-request Sec-Fetch & CH coherence flags
  http_header_single_{name}.png   — single-trace detail (8 panels)
  http_header_cross_trace.png     — cross-trace comparison (8 panels)

Usage
-----
  python http_header_analysis.py --traces ./traces
  python http_header_analysis.py --traces ./traces --out ./results
  python http_header_analysis.py --files trial-001/requests.jsonl trial-002/requests.jsonl
  python http_header_analysis.py --traces ./traces --no-filter
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COLLECT_ENDPOINT = "/collect"
UNLOAD_WINDOW_MS = 500
BURST_WINDOW_MS  = 50

# Headers that are meaningful for fingerprinting (presence + position)
FINGERPRINT_HEADERS = [
    "Accept", "Accept-Encoding", "Accept-Language",
    "User-Agent",
    "Sec-Ch-Ua", "Sec-Ch-Ua-Mobile", "Sec-Ch-Ua-Platform",
    "Sec-Fetch-Mode", "Sec-Fetch-Dest", "Sec-Fetch-Site", "Sec-Fetch-User",
    "Priority",
    "Upgrade-Insecure-Requests",
    "Referer", "Origin",
    "Content-Type", "Content-Length",
    "Cache-Control", "Connection",
    "Te", "Dnt",
]

# Expected Sec-Fetch-* combinations per request type
# (mode, dest) → (valid_sites, user_required)
SEC_FETCH_RULES = {
    ("navigate",  "document"):  ({"none", "same-origin", "cross-site"}, True),
    ("navigate",  "iframe"):    ({"none", "same-origin", "cross-site"}, True),
    ("no-cors",   "script"):    ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",   "style"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",   "image"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",   "font"):      ({"same-origin", "cross-site", "same-site"}, False),
    ("cors",      "empty"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("cors",      "script"):    ({"same-origin", "cross-site", "same-site"}, False),
    ("same-origin","empty"):    ({"same-origin"}, False),
    ("same-origin","document"): ({"same-origin"}, False),
}

# Priority values by request role
PRIORITY_EXPECTED = {
    "navigate":    {"u=0, i", "u=0,i"},
    "subresource": {"u=1", "u=2"},
    "async":       {"u=4, i", "u=4,i", "u=3, i", "u=3,i"},
}

# ─── SENBEACON FILTER ─────────────────────────────────────────────────────────

def is_grease_version(val: int) -> bool:
    return isinstance(val, int) and (val & 0x0F0F) == 0x0A0A


def filter_senbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove POST /collect rows that are sendBeacon flushes from logger.js.
    Three-rule filter — same logic as tls_analysis.py and trace_analysis.py.

    Rule 1: POST arrives within UNLOAD_WINDOW_MS before the next page GET.
    Rule 2: Multiple /collect POSTs cluster within BURST_WINDOW_MS.
    Rule 3: Referer header ends in .html.

    Removed when Rule 1 AND (Rule 2 OR Rule 3).
    """
    if df.empty:
        return df, 0

    collect_mask = (df["method"] == "POST") & (df["path"] == COLLECT_ENDPOINT)
    collect_idx  = df.index[collect_mask].tolist()
    if not collect_idx:
        return df, 0

    page_ts = df.loc[
        df["path"].str.endswith(".html", na=False), "timestamp"
    ].sort_values()

    def ms_to_next_page(ts):
        future = page_ts[page_ts > ts]
        if future.empty:
            return float("nan")
        return (future.iloc[0] - ts).total_seconds() * 1000

    flagged = set()
    for idx in collect_idx:
        row  = df.loc[idx]
        ts   = row["timestamp"]
        gap  = ms_to_next_page(ts)
        rule1 = (not np.isnan(gap)) and (gap <= UNLOAD_WINDOW_MS)
        if not rule1:
            continue
        other_ts = df.loc[collect_mask & (df.index != idx), "timestamp"]
        rule2 = any(
            abs((ts - ot).total_seconds() * 1000) <= BURST_WINDOW_MS
            for ot in other_ts
        )
        rule3 = bool(re.search(r"\.html$", str(row.get("referer", "") or "")))
        if rule2 or rule3:
            flagged.add(idx)

    if not flagged:
        return df, 0

    filtered = df.drop(index=list(flagged)).reset_index(drop=True)
    return filtered, len(flagged)


# ─── REQUEST TYPE CLASSIFIER ──────────────────────────────────────────────────

def classify_request(method: str, path: str,
                     sf_mode: str, sf_dest: str, sf_site: str) -> str:
    """
    Classify a request into a canonical type used for per-type header
    order fingerprinting and consistency checking.
    """
    if sf_mode == "navigate":
        return "navigate"
    if sf_mode == "no-cors":
        return f"subresource:{sf_dest or 'unknown'}"
    if sf_mode in ("cors", "same-origin") and sf_dest == "empty":
        return "fetch_xhr"
    if sf_mode == "cors":
        return f"cors:{sf_dest or 'unknown'}"
    if method == "POST":
        return "post"
    if path.endswith(".js"):
        return "subresource:script"
    if path.endswith((".css",)):
        return "subresource:style"
    if path.endswith((".png",".jpg",".jpeg",".gif",".webp",".svg",".ico")):
        return "subresource:image"
    return "other"


# ─── CLIENT HINTS PARSER ──────────────────────────────────────────────────────

# Sec-Ch-Ua brand list: "Chrome";v="128", "Not;A=Brand";v="24", ...
CH_UA_RE = re.compile(r'"([^"]+)";v="(\d+)"')

def parse_ch_ua(ch_ua_str: str) -> list[tuple[str, int]]:
    """Return list of (brand, version) tuples from Sec-Ch-Ua value."""
    if not ch_ua_str:
        return []
    return [(m.group(1), int(m.group(2))) for m in CH_UA_RE.finditer(ch_ua_str)]


def is_headless(ch_ua_str: str) -> bool:
    brands = [b.lower() for b, _ in parse_ch_ua(ch_ua_str)]
    return any("headless" in b for b in brands)


def ua_browser_family(ua: str) -> str:
    """Extract coarse browser family from User-Agent string."""
    ua = ua or ""
    if "HeadlessChrome" in ua or "headless" in ua.lower():
        return "headless-chrome"
    if "Edg/" in ua:
        return "edge"
    if "Chrome/" in ua:
        return "chrome"
    if "Firefox/" in ua:
        return "firefox"
    if "Safari/" in ua and "Chrome" not in ua:
        return "safari"
    if "curl/" in ua:
        return "curl"
    if "python" in ua.lower():
        return "python"
    if "Go-http-client" in ua:
        return "go"
    if "okhttp" in ua.lower():
        return "okhttp"
    return "other"


def ua_os(ua: str) -> str:
    ua = ua or ""
    if "Windows" in ua:
        return "windows"
    if "Macintosh" in ua or "Mac OS" in ua:
        return "macos"
    if "Linux" in ua and "Android" not in ua:
        return "linux"
    if "Android" in ua:
        return "android"
    if "iPhone" in ua or "iPad" in ua:
        return "ios"
    return "other"


def ua_version(ua: str) -> str | None:
    """Extract Chrome/Firefox/Safari version from UA string."""
    for pattern in [r"Chrome/(\d+)", r"Firefox/(\d+)", r"Safari/(\d+)"]:
        m = re.search(pattern, ua or "")
        if m:
            return m.group(1)
    return None


# ─── SEC-FETCH CONSISTENCY ────────────────────────────────────────────────────

def check_sec_fetch(sf_mode: str, sf_dest: str,
                    sf_site: str, sf_user: str,
                    req_type: str) -> dict:
    """
    Return a dict of boolean consistency flags for the Sec-Fetch-* suite.
    All flags True = no violations detected.
    """
    sf_mode = sf_mode or ""
    sf_dest = sf_dest or ""
    sf_site = sf_site or ""
    sf_user = sf_user or ""

    violations = {}

    key = (sf_mode, sf_dest)
    rule = SEC_FETCH_RULES.get(key)

    # Rule: known (mode, dest) combination?
    violations["sf_known_combo"] = rule is not None

    if rule:
        valid_sites, user_required = rule
        violations["sf_site_valid"]    = (sf_site in valid_sites) if sf_site else True
        violations["sf_user_correct"]  = (sf_user == "?1") if user_required else (sf_user == "")
    else:
        violations["sf_site_valid"]   = True   # can't assess unknown combo
        violations["sf_user_correct"] = True

    # navigate must have Sec-Fetch-User = ?1
    if sf_mode == "navigate":
        violations["sf_nav_has_user"] = sf_user == "?1"
    else:
        violations["sf_nav_has_user"] = True  # N/A

    # non-navigate must NOT have Sec-Fetch-User
    if sf_mode != "navigate":
        violations["sf_no_user_on_non_nav"] = sf_user == ""
    else:
        violations["sf_no_user_on_non_nav"] = True  # N/A

    # cors mode → dest must be empty (for fetch/XHR)
    if sf_mode == "cors" and req_type == "fetch_xhr":
        violations["sf_cors_dest_empty"] = sf_dest == "empty"
    else:
        violations["sf_cors_dest_empty"] = True  # N/A

    violations["sf_any_violation"] = not all(violations.values())
    return violations


# ─── CLIENT HINTS COHERENCE ───────────────────────────────────────────────────

def check_client_hints(ch_ua: str, ch_mobile: str,
                       ch_platform: str, ua: str) -> dict:
    """
    Cross-check Sec-Ch-Ua, Sec-Ch-Ua-Mobile, Sec-Ch-Ua-Platform vs User-Agent.
    Returns dict of coherence flags (True = coherent).
    """
    ua = ua or ""
    brands = parse_ch_ua(ch_ua)
    brand_names_lower = [b.lower() for b, _ in brands]

    checks = {}

    # Headless detection
    checks["ch_is_headless"] = any("headless" in b for b in brand_names_lower)

    # If UA says Chrome, Sec-Ch-Ua should mention Chromium or Chrome
    ua_is_chrome = "Chrome/" in ua or "Chromium/" in ua
    if ua_is_chrome and brands:
        checks["ch_brand_matches_ua"] = any(
            "chromium" in b or "chrome" in b or "edge" in b or "brand" in b
            for b in brand_names_lower
        )
    else:
        checks["ch_brand_matches_ua"] = True  # can't assess non-Chrome

    # Version cross-check: Chrome/NNN in UA vs version in Sec-Ch-Ua
    ua_ver = ua_version(ua)
    if ua_ver and brands:
        ch_versions = [v for _, v in brands if v > 10]  # ignore "Not A Brand" v=24
        checks["ch_version_near_ua"] = any(
            abs(v - int(ua_ver)) <= 2 for v in ch_versions
        ) if ch_versions else True
    else:
        checks["ch_version_near_ua"] = True

    # Mobile flag coherence: ?1 → UA should mention Mobile/Android/iPhone
    mobile_flag = (ch_mobile or "").strip() == "?1"
    ua_is_mobile = any(kw in ua for kw in ["Mobile", "Android", "iPhone", "iPad"])
    checks["ch_mobile_matches_ua"] = (mobile_flag == ua_is_mobile)

    # Platform coherence
    platform = (ch_platform or "").strip().strip('"').lower()
    ua_os_str = ua_os(ua)
    platform_os_map = {
        "windows": "windows",
        "macos":   "macos",
        "linux":   "linux",
        "android": "android",
        "ios":     "ios",
    }
    expected_platform = platform_os_map.get(ua_os_str, "")
    if platform and expected_platform:
        checks["ch_platform_matches_ua"] = (platform == expected_platform)
    else:
        checks["ch_platform_matches_ua"] = True  # can't assess

    checks["ch_any_incoherence"] = not all(
        v for k, v in checks.items() if k != "ch_is_headless"
    )
    return checks


# ─── PRIORITY HEADER ─────────────────────────────────────────────────────────

def classify_priority(priority_val: str, req_type: str) -> dict:
    """Assess whether the Priority header value is appropriate for the request type."""
    p = (priority_val or "").strip()
    result = {
        "priority_raw":     p,
        "priority_urgency": None,
        "priority_incr":    None,
        "priority_expected":None,
    }
    if not p:
        return result

    m = re.match(r"u=(\d+)(?:,\s*i)?", p)
    if m:
        result["priority_urgency"] = int(m.group(1))
        result["priority_incr"]    = "i" in p

    # Check expectation
    if req_type == "navigate":
        result["priority_expected"] = p in PRIORITY_EXPECTED["navigate"]
    elif req_type.startswith("subresource"):
        result["priority_expected"] = p in PRIORITY_EXPECTED["subresource"]
    elif req_type == "fetch_xhr":
        result["priority_expected"] = p in PRIORITY_EXPECTED["async"]
    else:
        result["priority_expected"] = True  # can't assess

    return result


# ─── SINGLE-REQUEST PARSER ────────────────────────────────────────────────────

def parse_request(rec: dict, trace_label: str) -> dict:
    http    = rec.get("http") or {}
    headers_ordered = http.get("headers_ordered") or []
    headers         = http.get("headers") or {}

    def hval(name: str) -> str:
        """Get first value of a header (case-sensitive key in headers dict)."""
        vals = headers.get(name)
        return vals[0] if vals else ""

    method   = http.get("method", "")
    path     = http.get("path", "")
    protocol = http.get("protocol", "")

    sf_mode = http.get("sec_fetch_mode") or hval("Sec-Fetch-Mode")
    sf_dest = http.get("sec_fetch_dest") or hval("Sec-Fetch-Dest")
    sf_site = http.get("sec_fetch_site") or hval("Sec-Fetch-Site")
    sf_user = http.get("sec_fetch_user") or hval("Sec-Fetch-User")

    ch_ua       = http.get("sec_ch_ua")       or hval("Sec-Ch-Ua")
    ch_mobile   = http.get("sec_ch_ua_mobile") or hval("Sec-Ch-Ua-Mobile")
    ch_platform = http.get("sec_ch_ua_platform") or hval("Sec-Ch-Ua-Platform")
    ua          = http.get("user_agent")      or hval("User-Agent")
    accept      = http.get("accept")          or hval("Accept")
    accept_enc  = http.get("accept_encoding") or hval("Accept-Encoding")
    referer     = http.get("referer")         or hval("Referer")
    priority    = hval("Priority")
    origin      = hval("Origin")
    content_type= hval("Content-Type")
    upgrade_ins = hval("Upgrade-Insecure-Requests")

    req_type = classify_request(method, path, sf_mode, sf_dest, sf_site)

    # ── Header order fingerprint ───────────────────────────────────────────
    order_names = [h["name"] for h in headers_ordered]
    order_str   = ",".join(order_names)
    order_hash  = hashlib.sha256(order_str.encode()).hexdigest()[:16]

    row = {
        "trace":        trace_label,
        "request_id":   rec.get("id"),
        "timestamp":    rec.get("timestamp"),
        "method":       method,
        "path":         path,
        "protocol":     protocol,
        "req_type":     req_type,
        # ── header order ─────────────────────────────────────────────────
        "header_order_str":  order_str,
        "header_order_hash": order_hash,
        "n_headers":         len(order_names),
        # ── Sec-Fetch-* ───────────────────────────────────────────────────
        "sf_mode":      sf_mode,
        "sf_dest":      sf_dest,
        "sf_site":      sf_site,
        "sf_user":      sf_user,
        "sf_combo":     f"{sf_mode}|{sf_dest}|{sf_site}",
        # ── Client Hints ──────────────────────────────────────────────────
        "ch_ua":              ch_ua,
        "ch_mobile":          ch_mobile,
        "ch_platform":        ch_platform,
        "ch_is_headless":     is_headless(ch_ua),
        "ch_brand_count":     len(parse_ch_ua(ch_ua)),
        # ── User-Agent ────────────────────────────────────────────────────
        "ua":                 ua,
        "ua_family":          ua_browser_family(ua),
        "ua_os":              ua_os(ua),
        "ua_version":         ua_version(ua),
        # ── Priority ──────────────────────────────────────────────────────
        "priority":           priority,
        # ── Other headers ─────────────────────────────────────────────────
        "accept":             accept,
        "accept_encoding":    accept_enc,
        "referer":            referer,
        "origin":             origin,
        "content_type":       content_type,
        "upgrade_insecure":   upgrade_ins,
        # ── Header presence flags (key discriminating headers) ────────────
        **{f"has_{h.replace('-','_').lower()}": (h in [n for n in order_names])
           for h in FINGERPRINT_HEADERS},
    }

    # ── Sec-Fetch consistency ─────────────────────────────────────────────
    sf_checks = check_sec_fetch(sf_mode, sf_dest, sf_site, sf_user, req_type)
    row.update({f"sf_{k}": v for k, v in sf_checks.items()
                if k not in row})   # avoid overwriting sf_mode etc.

    # ── Client Hints coherence ────────────────────────────────────────────
    ch_checks = check_client_hints(ch_ua, ch_mobile, ch_platform, ua)
    row.update(ch_checks)

    # ── Priority assessment ────────────────────────────────────────────────
    prio = classify_priority(priority, req_type)
    row.update(prio)

    return row


# ─── TRACE LOADER ─────────────────────────────────────────────────────────────

def load_trace(filepath: Path, label: str) -> pd.DataFrame:
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
            rows.append(parse_request(rec, label))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def label_for(fp: Path) -> str:
    return fp.parent.name if fp.parent.name else fp.stem


def collect_filepaths(traces_dir: str | Path,
                      filename: str = "requests.jsonl") -> list[Path]:
    root  = Path(traces_dir)
    found = sorted(root.glob(f"*/{filename}"))
    return found if found else sorted(root.glob("*.jsonl"))


def load_all(filepaths: list[Path],
             do_filter: bool = True) -> dict[str, pd.DataFrame]:
    traces = {}
    for fp in filepaths:
        label = label_for(fp)
        df = load_trace(fp, label)
        if df.empty:
            print(f"  [warn] {fp} — empty or unparseable, skipped")
            continue
        if do_filter:
            df, n_removed = filter_senbeacon(df)
            suffix = f", removed {n_removed} sendBeacon POST(s)" if n_removed else ""
        else:
            suffix = " (no filter)"
        traces[label] = df
        print(f"  {label}: {len(df)} requests{suffix}")
    return traces


# ─── PER-TYPE HEADER ORDER FINGERPRINTING ────────────────────────────────────

def header_order_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each request type, find the modal header order and how stable it is.
    """
    rows = []
    for req_type, grp in df.groupby("req_type"):
        orders = grp["header_order_str"].dropna()
        if orders.empty:
            continue
        counts = orders.value_counts()
        modal  = counts.index[0]
        modal_f = counts.iloc[0] / len(orders)
        rows.append({
            "req_type":       req_type,
            "n_requests":     len(grp),
            "n_distinct_orders": counts.nunique(),
            "modal_order":    modal,
            "modal_order_hash": hashlib.sha256(modal.encode()).hexdigest()[:16],
            "order_stability": modal_f,
        })
    return pd.DataFrame(rows)


# ─── CROSS-TRACE STABILITY ────────────────────────────────────────────────────

STABILITY_FIELDS = [
    "header_order_hash", "n_headers",
    "sf_mode", "sf_dest", "sf_site", "sf_combo",
    "sf_known_combo", "sf_any_violation",
    "ch_is_headless", "ch_brand_count",
    "ch_brand_matches_ua", "ch_mobile_matches_ua", "ch_platform_matches_ua",
    "ch_any_incoherence",
    "ua_family", "ua_os", "ua_version",
    "priority_urgency", "priority_incr", "priority_expected",
    "accept", "accept_encoding", "upgrade_insecure",
    "has_priority", "has_sec_ch_ua", "has_sec_fetch_mode",
    "has_upgrade_insecure_requests",
]


def cross_trace_stability(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-field: modal value per trace, then stability + entropy across all traces.
    """
    rep = {}
    for name, df in traces.items():
        row = {}
        for field in STABILITY_FIELDS:
            if field in df.columns:
                vals = df[field].dropna()
                row[field] = str(vals.mode().iloc[0]) if not vals.empty else None
        rep[name] = row

    rep_df = pd.DataFrame(rep).T

    results = []
    for field in rep_df.columns:
        vals = rep_df[field].dropna()
        if vals.empty:
            continue
        counts = vals.value_counts()
        total  = len(vals)
        modal  = counts.index[0]
        modal_f = counts.iloc[0] / total
        probs  = np.array([c / total for c in counts.tolist()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        results.append({
            "field":       field,
            "n_distinct":  len(counts),
            "modal_value": str(modal),
            "stability":   modal_f,
            "entropy":     entropy,
        })
    return (pd.DataFrame(results)
            .sort_values("entropy", ascending=False)
            .reset_index(drop=True))


def build_feature_vector(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per trace — modal value of each STABILITY_FIELD."""
    rows = []
    for name, df in traces.items():
        row = {"trace": name}
        for field in STABILITY_FIELDS:
            if field in df.columns:
                vals = df[field].dropna()
                row[field] = vals.mode().iloc[0] if not vals.empty else None
            else:
                row[field] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("trace")


def consistency_violations(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All requests where any Sec-Fetch or Client Hints violation was detected."""
    parts = []
    for name, df in traces.items():
        viol_cols = [c for c in df.columns
                     if c in ("sf_any_violation", "ch_any_incoherence",
                               "priority_expected")]
        sub = df[
            df.get("sf_any_violation", pd.Series(False, index=df.index)) |
            df.get("ch_any_incoherence", pd.Series(False, index=df.index)) |
            (~df.get("priority_expected", pd.Series(True, index=df.index)).fillna(True))
        ].copy()
        if not sub.empty:
            sub["trace"] = name
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

C = {
    "blue":   "#2563EB", "teal":   "#0D9488",
    "amber":  "#D97706", "red":    "#DC2626",
    "gray":   "#6B7280", "purple": "#7C3AED",
    "green":  "#16A34A", "pink":   "#DB2777",
}


def _ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_trace(df: pd.DataFrame, trace_name: str, out_path=None):
    fig = plt.figure(figsize=(20, 14), layout="constrained")
    fig.suptitle(f"HTTP header fingerprint — {trace_name}",
                 fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.25)
    fig.tight_layout(pad=2.0)

    # 1. Header order per request type (heatmap of header presence × request)
    ax1 = fig.add_subplot(gs[0, :2])
    presence = df[[f"has_{h.replace('-','_').lower()}"
                   for h in FINGERPRINT_HEADERS
                   if f"has_{h.replace('-','_').lower()}" in df.columns]].astype(float)
    short_names = [h.replace("Sec-Fetch-","SF-").replace("Sec-Ch-Ua","CH-UA")
                   .replace("Accept-Encoding","Acc-Enc")
                   .replace("Upgrade-Insecure-Requests","UIR")
                   .replace("Content-Type","CT").replace("Content-Length","CL")
                   for h in FINGERPRINT_HEADERS
                   if f"has_{h.replace('-','_').lower()}" in df.columns]
    if not presence.empty:
        im = ax1.imshow(presence.values.T, aspect="auto",
                        cmap="Blues", vmin=0, vmax=1)
        ax1.set_yticks(range(len(short_names)))
        ax1.set_yticklabels(short_names, fontsize=5.5)
        ax1.set_xlabel("request index", fontsize=7)
        plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01, label="present")
    _ax(ax1, "Header presence per request (columns=requests, rows=headers)")

    # 2. Request type distribution
    ax2 = fig.add_subplot(gs[0, 2])
    rt_counts = df["req_type"].value_counts()
    ax2.barh(range(len(rt_counts)), rt_counts.values, color=C["teal"])
    ax2.set_yticks(range(len(rt_counts)))
    ax2.set_yticklabels(rt_counts.index, fontsize=6)
    _ax(ax2, "Request type distribution", "count")

    # 3. Header order fingerprint — unique hashes per request type
    ax3 = fig.add_subplot(gs[1, 0])
    hop = header_order_profiles(df)
    if not hop.empty:
        x = np.arange(len(hop))
        ax3.bar(x, hop["order_stability"], color=C["blue"], alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(hop["req_type"], rotation=30, ha="right", fontsize=6)
        ax3.set_ylim(0, 1.1)
        ax3.axhline(1, color=C["gray"], lw=0.8, linestyle="--")
    _ax(ax3, "Header order stability\nby request type", ylabel="stability (1=always same order)")

    # 4. Sec-Fetch-* combo distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sf_counts = df["sf_combo"].value_counts().head(10)
    ax4.barh(range(len(sf_counts)), sf_counts.values, color=C["purple"])
    ax4.set_yticks(range(len(sf_counts)))
    ax4.set_yticklabels(sf_counts.index, fontsize=5.5)
    _ax(ax4, "Sec-Fetch-* combos\n(Mode|Dest|Site)", "count")

    # 5. Client Hints coherence flags
    ax5 = fig.add_subplot(gs[1, 2])
    ch_flags = {
        "brand↔UA":    df.get("ch_brand_matches_ua", pd.Series(dtype=float)).mean(),
        "mobile↔UA":   df.get("ch_mobile_matches_ua", pd.Series(dtype=float)).mean(),
        "platform↔UA": df.get("ch_platform_matches_ua", pd.Series(dtype=float)).mean(),
        "version≈UA":  df.get("ch_version_near_ua", pd.Series(dtype=float)).mean(),
    }
    labels_ch = list(ch_flags.keys())
    vals_ch   = [v if not np.isnan(v) else 0 for v in ch_flags.values()]
    colors_ch = [C["green"] if v >= 0.95 else C["amber"] if v >= 0.5 else C["red"]
                 for v in vals_ch]
    ax5.bar(range(len(labels_ch)), vals_ch, color=colors_ch)
    ax5.set_xticks(range(len(labels_ch)))
    ax5.set_xticklabels(labels_ch, rotation=20, ha="right", fontsize=7)
    ax5.set_ylim(0, 1.1)
    ax5.axhline(1, color=C["gray"], lw=0.8, linestyle="--")
    _ax(ax5, "Client Hints coherence\n(1.0 = fully coherent)", ylabel="fraction coherent")

    # 6. Priority values by request type
    ax6 = fig.add_subplot(gs[2, 0])
    prio_data = df[df["priority"].notna() & (df["priority"] != "")]
    if not prio_data.empty:
        prio_counts = prio_data.groupby(["req_type", "priority"]).size().unstack(fill_value=0)
        prio_counts.plot(kind="bar", ax=ax6, legend=True,
                         colormap="tab10", width=0.7)
        ax6.set_xticklabels(prio_counts.index, rotation=30, ha="right", fontsize=6)
        ax6.legend(fontsize=5, loc="upper right")
    _ax(ax6, "Priority header values\nby request type", ylabel="count")

    # 7. Sec-Fetch violations (if any)
    ax7 = fig.add_subplot(gs[2, 1])
    viol_fields = ["sf_known_combo", "sf_site_valid", "sf_user_correct",
                   "sf_nav_has_user", "sf_no_user_on_non_nav"]
    viol_rates = {}
    for f in viol_fields:
        if f in df.columns:
            rate = 1 - df[f].fillna(True).astype(float).mean()
            viol_rates[f.replace("sf_","")] = rate
    if viol_rates:
        v_labels = list(viol_rates.keys())
        v_vals   = list(viol_rates.values())
        colors_v = [C["red"] if v > 0.05 else C["green"] for v in v_vals]
        ax7.bar(range(len(v_labels)), v_vals, color=colors_v)
        ax7.set_xticks(range(len(v_labels)))
        ax7.set_xticklabels(v_labels, rotation=30, ha="right", fontsize=6)
        ax7.set_ylim(0, max(max(v_vals) * 1.3, 0.1))
    _ax(ax7, "Sec-Fetch-* violation rate\n(0 = fully consistent)", ylabel="violation rate")

    # 8. UA family & headless detection
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    summary_data = [
        ("UA family",       df["ua_family"].mode().iloc[0] if not df["ua_family"].empty else "—"),
        ("UA OS",           df["ua_os"].mode().iloc[0] if not df["ua_os"].empty else "—"),
        ("UA version",      df["ua_version"].dropna().mode().iloc[0] if not df["ua_version"].dropna().empty else "—"),
        ("Headless",        str(df["ch_is_headless"].any())),
        ("Headless rate",   f"{df['ch_is_headless'].mean():.0%}"),
        ("Sec-Fetch violations", str(df.get("sf_any_violation", pd.Series([False])).sum())),
        ("CH incoherences", str(df.get("ch_any_incoherence", pd.Series([False])).sum())),
        ("Priority mismatches", str((~df.get("priority_expected",
                                             pd.Series([True]*len(df))).fillna(True)).sum())),
        ("Unique header orders", str(df["header_order_hash"].nunique())),
        ("Total requests",  str(len(df))),
    ]
    tbl = ax8.table(
        cellText=[[k, str(v)] for k, v in summary_data],
        colLabels=["Metric", "Value"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.2)
    _ax(ax8, "Session summary")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


def plot_cross_trace(traces: dict[str, pd.DataFrame],
                     stability_df: pd.DataFrame,
                     fv: pd.DataFrame,
                     out_path=None):
    fig = plt.figure(figsize=(20, 14), layout="constrained")
    # fig.suptitle(f"HTTP header cross-trace comparison — {len(traces)} traces",
    #              fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.25)
    fig.tight_layout(pad=2.0)

    # 1. Field stability ranked by entropy
    ax1 = fig.add_subplot(gs[0, :2])
    top = stability_df.head(20)
    colors_s = [C["green"] if s >= 0.95 else C["amber"] if s >= 0.5 else C["red"]
                for s in top["stability"]]
    ax1.barh(range(len(top)), top["stability"], color=colors_s[::-1])
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top["field"][::-1], fontsize=6)
    ax1.set_xlim(0, 1.15)
    ax1.axvline(1.0, color=C["gray"], lw=0.8, linestyle="--")
    for i, (_, row) in enumerate(top.iterrows()):
        ax1.text(min(row["stability"] + 0.01, 1.08), len(top) - 1 - i,
                 f"{row['stability']:.2f}", va="center", fontsize=5.5)
    _ax(ax1, "Feature stability across traces (sorted by entropy — most discriminative first)",
        "stability score")

    # 2. Entropy ranking
    ax2 = fig.add_subplot(gs[0, 2])
    top_ent = stability_df[stability_df["entropy"] > 0].head(12)
    ax2.barh(range(len(top_ent)), top_ent["entropy"], color=C["purple"])
    ax2.set_yticks(range(len(top_ent)))
    ax2.set_yticklabels(top_ent["field"], fontsize=6)
    _ax(ax2, "Discriminative power\n(Shannon entropy)", "entropy (bits)")

    # 3. Header order hash distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if "header_order_hash" in fv.columns:
        cnt = fv["header_order_hash"].value_counts()
        ax3.bar(range(len(cnt)), cnt.values, color=C["blue"])
        ax3.set_xticks(range(len(cnt)))
        ax3.set_xticklabels(cnt.index, rotation=45, ha="right", fontsize=5)
    _ax(ax3, "Header order hash distribution\nacross traces", ylabel="trace count")

    # 4. UA family distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if "ua_family" in fv.columns:
        cnt = fv["ua_family"].value_counts()
        ax4.bar(range(len(cnt)), cnt.values, color=C["teal"])
        ax4.set_xticks(range(len(cnt)))
        ax4.set_xticklabels(cnt.index, rotation=30, ha="right", fontsize=7)
    _ax(ax4, "UA family distribution\nacross traces", ylabel="trace count")

    # 5. Headless Chrome detection rate
    ax5 = fig.add_subplot(gs[1, 2])
    headless_rates = [
        df["ch_is_headless"].mean() for df in traces.values()
    ]
    ax5.hist(headless_rates, bins=min(10, len(headless_rates)),
             color=C["red"], edgecolor="white")
    ax5.axvline(np.mean(headless_rates), color=C["gray"],
                lw=1.5, linestyle="--",
                label=f"mean {np.mean(headless_rates):.0%}")
    ax5.legend(fontsize=7)
    _ax(ax5, "Headless Chrome detection rate\nacross traces",
        "fraction of requests flagged headless", "trace count")

    # 6. Sec-Fetch violation rate per trace
    ax6 = fig.add_subplot(gs[2, 0])
    viol_rates = [
        df.get("sf_any_violation", pd.Series([False]*len(df))).mean()
        for df in traces.values()
    ]
    ax6.bar(range(len(viol_rates)), viol_rates, color=C["amber"])
    ax6.set_xticks(range(len(viol_rates)))
    ax6.set_xticklabels(list(traces.keys()), rotation=90, fontsize=5)
    _ax(ax6, "Sec-Fetch-* violation rate\nper trace", ylabel="violation rate")

    # 7. Client Hints coherence heatmap across traces
    ax7 = fig.add_subplot(gs[2, 1])
    ch_cols = ["ch_brand_matches_ua", "ch_mobile_matches_ua",
               "ch_platform_matches_ua", "ch_any_incoherence"]
    ch_present = [c for c in ch_cols if c in fv.columns]
    if ch_present:
        ch_data = fv[ch_present].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        ).astype(float)
        im = ax7.imshow(ch_data.values.T, aspect="auto",
                        cmap="RdYlGn", vmin=0, vmax=1)
        ax7.set_xticks(range(len(fv)))
        ax7.set_xticklabels(fv.index, rotation=90, fontsize=5)
        ax7.set_yticks(range(len(ch_present)))
        ax7.set_yticklabels([c.replace("ch_","") for c in ch_present], fontsize=6)
        plt.colorbar(im, ax=ax7, fraction=0.025, pad=0.02, label="coherent")
    _ax(ax7, "Client Hints coherence\nper trace (green=OK, red=violation)")

    # 8. Priority correctness rate per trace
    ax8 = fig.add_subplot(gs[2, 2])
    prio_correct = []
    for df in traces.values():
        col = df.get("priority_expected", pd.Series([True]*len(df)))
        prio_correct.append(col.fillna(True).astype(float).mean())
    ax8.bar(range(len(prio_correct)), prio_correct, color=C["green"])
    ax8.set_xticks(range(len(prio_correct)))
    ax8.set_xticklabels(list(traces.keys()), rotation=90, fontsize=5)
    ax8.set_ylim(0, 1.1)
    ax8.axhline(1, color=C["gray"], lw=0.8, linestyle="--")
    _ax(ax8, "Priority header correctness rate\nper trace", ylabel="fraction correct")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── CONSOLE REPORTING ────────────────────────────────────────────────────────

def _sec(title):
    print(f"\n{'─'*70}\n  {title}\n{'─'*70}")


def print_header_order(traces: dict[str, pd.DataFrame]):
    _sec("Header order fingerprint — modal order per request type per trace")
    for name, df in traces.items():
        print(f"\n  [{name}]")
        hop = header_order_profiles(df)
        if hop.empty:
            print("    (no data)")
            continue
        for _, row in hop.iterrows():
            order_short = " → ".join(
                h.replace("Sec-Fetch-","SF-").replace("Sec-Ch-Ua","CH")
                 .replace("Accept-Encoding","AccEnc")
                 .replace("Upgrade-Insecure-Requests","UIR")
                for h in row["modal_order"].split(",")
            )
            print(f"    {row['req_type']:<22} stability={row['order_stability']:.2f}"
                  f"  hash={row['modal_order_hash']}")
            print(f"      order: {order_short}")


def print_sec_fetch(traces: dict[str, pd.DataFrame]):
    _sec("Sec-Fetch-* consistency — violation counts per trace")
    viol_cols = ["sf_known_combo","sf_site_valid","sf_user_correct",
                 "sf_nav_has_user","sf_no_user_on_non_nav","sf_any_violation"]
    for name, df in traces.items():
        present = [c for c in viol_cols if c in df.columns]
        rates   = {c: int((~df[c].fillna(True).astype(bool)).sum())
                   for c in present}
        print(f"  {name}: " + "  ".join(f"{k.replace('sf_','')}={v}"
                                         for k, v in rates.items()))


def print_client_hints(traces: dict[str, pd.DataFrame]):
    _sec("Client Hints coherence — per trace")
    fields = ["ch_brand_matches_ua","ch_mobile_matches_ua",
              "ch_platform_matches_ua","ch_any_incoherence","ch_is_headless"]
    for name, df in traces.items():
        parts = []
        for f in fields:
            if f in df.columns:
                val = df[f].mean()
                parts.append(f"{f.replace('ch_','')}: {val:.0%}")
        print(f"  {name}: " + "  |  ".join(parts))


def print_stability(stability_df: pd.DataFrame):
    _sec("Cross-trace feature stability & discriminative power (top 20)")
    print(stability_df.head(20).to_string(index=False))


def print_discriminative(stability_df: pd.DataFrame):
    _sec("Most discriminative features for agent identification (entropy > 0)")
    disc = stability_df[stability_df["entropy"] > 0]
    if disc.empty:
        print("  All features identical across traces.")
    else:
        print(disc[["field","n_distinct","stability","entropy"]].to_string(index=False))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HTTP header fingerprint analyzer for AI agent identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python http_header_analysis.py --traces ./traces
  python http_header_analysis.py --traces ./traces --out ./results
  python http_header_analysis.py --files trial-001/requests.jsonl trial-002/requests.jsonl
  python http_header_analysis.py --traces ./traces --no-filter
""")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--traces", metavar="DIR",
                     help="root traces directory (trial-NNN/requests.jsonl layout)")
    grp.add_argument("--files", nargs="+",
                     help="explicit list of requests.jsonl files")
    parser.add_argument("--filename", default="requests.jsonl",
                        help="filename inside each trial dir (default: requests.jsonl)")
    parser.add_argument("--out", default=".",
                        help="output directory (default: .)")
    parser.add_argument("--no-filter", action="store_true",
                        help="keep sendBeacon POST /collect rows (skip filter)")
    args = parser.parse_args()

    do_filter = not args.no_filter

    # ── Discover files ────────────────────────────────────────────────────────
    if args.traces:
        filepaths = collect_filepaths(args.traces, filename=args.filename)
    else:
        filepaths = sorted(set(
            Path(p) for pat in args.files for p in glob.glob(pat, recursive=True)
        ))

    if not filepaths:
        print("No files found. Exiting.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {len(filepaths)} trace file(s)…")
    print(f"  sendBeacon filter: {'ON' if do_filter else 'OFF (--no-filter)'}")
    traces = load_all(filepaths, do_filter=do_filter)

    if not traces:
        print("No valid traces loaded. Exiting.")
        sys.exit(1)

    # ── Core outputs ──────────────────────────────────────────────────────────
    all_requests = pd.concat(traces.values(), ignore_index=True)
    all_requests.to_csv(out_dir / "http_header_summary.csv", index=False)
    print(f"\n  Saved: {out_dir / 'http_header_summary.csv'}")

    stability_df = cross_trace_stability(traces)
    stability_df.to_csv(out_dir / "http_header_stability.csv", index=False)
    print(f"  Saved: {out_dir / 'http_header_stability.csv'}")

    fv = build_feature_vector(traces)
    fv.to_csv(out_dir / "http_header_fingerprint_vector.csv")
    print(f"  Saved: {out_dir / 'http_header_fingerprint_vector.csv'}")

    violations = consistency_violations(traces)
    if not violations.empty:
        violations.to_csv(out_dir / "http_consistency_violations.csv", index=False)
        print(f"  Saved: {out_dir / 'http_consistency_violations.csv'}"
              f" ({len(violations)} violation(s))")
    else:
        print("  No consistency violations found.")

    # ── Console reports ───────────────────────────────────────────────────────
    print_header_order(traces)
    print_sec_fetch(traces)
    print_client_hints(traces)
    print_stability(stability_df)
    print_discriminative(stability_df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(traces) == 1:
        name, df = next(iter(traces.items()))
        plot_single_trace(df, name,
                          out_path=str(out_dir / f"http_header_single_{name}.png"))
    else:
        name0, df0 = next(iter(traces.items()))
        plot_single_trace(df0, name0,
                          out_path=str(out_dir / f"http_header_single_{name0}.png"))
        plot_cross_trace(traces, stability_df, fv,
                         out_path=str(out_dir / "http_header_cross_trace.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()