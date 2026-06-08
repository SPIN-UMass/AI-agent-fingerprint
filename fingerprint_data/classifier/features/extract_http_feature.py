"""
agent_header_profiler.py
========================
Extracts HTTP header features directly from requests.jsonl files and builds
agent-level fingerprint profiles for classifier use.

Directory layout expected:
    <base_dir>/
        autogen/
            trial-001/requests.jsonl
            trial-002/requests.jsonl
            ...
        skyvern/
            trial-001/requests.jsonl
            ...

What this script produces
-------------------------
Per-request features (7 groups):
  1. Header order fingerprint  — order_hash, n_headers, unique ratio
  2. Sec-Fetch-* consistency   — sf_site_none, sf_violation_rate, sf_combo
  3. Client Hints coherence    — headless, brand/mobile/platform vs UA
  4. Header presence flags     — has_accept_language, has_priority, …
  5. User-Agent analysis       — ua_family, ua_os, ua_version
  6. Priority header           — urgency, correctness
  7. Non-standard headers      — names outside the known-browser set  ← NEW

Aggregation levels:
  • Request  → one row per HTTP request  (request_features.csv)
  • Trial    → one row per trial         (trial_features.csv)
  • Agent    → one row per agent         (agent_profiles.csv)   ← NEW

Usage
-----
  python agent_header_profiler.py --base ./traces
  python agent_header_profiler.py --base ./traces --out ./results
  python agent_header_profiler.py --base ./traces --agents autogen skyvern
  python agent_header_profiler.py --base ./traces --no-filter
"""

import re
import json
import hashlib
import argparse
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

COLLECT_ENDPOINT  = "/collect"
UNLOAD_WINDOW_MS  = 500
BURST_WINDOW_MS   = 50

# Headers that are standard in real Chrome / headless Chrome browsers.
# Anything outside this set is flagged as non-standard / injected.
STANDARD_HEADERS: set[str] = {
    "accept", "accept-encoding", "accept-language", "user-agent",
    "priority", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
    "sec-fetch-mode", "sec-fetch-dest", "sec-fetch-site", "sec-fetch-user",
    "upgrade-insecure-requests", "referer", "origin",
    "content-type", "content-length", "cache-control",
    "connection", "te", "dnt",
    "if-modified-since", "if-none-match", "range",
    "authorization", "cookie", "host", "transfer-encoding",
}

# Subset tracked for per-request presence flags (has_<name>)
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

# Valid Sec-Fetch (mode, dest) → (valid_sites, user_required)
SEC_FETCH_RULES: dict = {
    ("navigate",   "document"):  ({"none", "same-origin", "cross-site"}, True),
    ("navigate",   "iframe"):    ({"none", "same-origin", "cross-site"}, True),
    ("no-cors",    "script"):    ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",    "style"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",    "image"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("no-cors",    "font"):      ({"same-origin", "cross-site", "same-site"}, False),
    ("cors",       "empty"):     ({"same-origin", "cross-site", "same-site"}, False),
    ("cors",       "script"):    ({"same-origin", "cross-site", "same-site"}, False),
    ("same-origin","empty"):     ({"same-origin"}, False),
    ("same-origin","document"):  ({"same-origin"}, False),
}

# Expected Priority values by request role
PRIORITY_EXPECTED: dict = {
    "navigate":    {"u=0, i", "u=0,i"},
    "subresource": {"u=1", "u=2"},
    "async":       {"u=4, i", "u=4,i", "u=3, i", "u=3,i"},
}

# ─── SENDBEACON FILTER ────────────────────────────────────────────────────────

def filter_senbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove POST /collect rows that are sendBeacon flushes from logger.js.
    Rule 1: POST arrives within UNLOAD_WINDOW_MS before the next page GET.
    Rule 2: Multiple /collect POSTs cluster within BURST_WINDOW_MS.
    Rule 3: Referer header ends in .html.
    Removed when Rule 1 AND (Rule 2 OR Rule 3).
    """
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
    # collect_mask = (df["method"] == "POST") & (df["path"] == COLLECT_ENDPOINT)
    # collect_idx  = df.index[collect_mask].tolist()
    # if not collect_idx:
    #     return df, 0

    # page_ts = df.loc[
    #     df["path"].str.endswith(".html", na=False), "timestamp"
    # ].sort_values()

    # def ms_to_next_page(ts):
    #     future = page_ts[page_ts > ts]
    #     if future.empty:
    #         return float("nan")
    #     return (future.iloc[0] - ts).total_seconds() * 1000

    # flagged = set()
    # for idx in collect_idx:
    #     row  = df.loc[idx]
    #     ts   = row["timestamp"]
    #     gap  = ms_to_next_page(ts)
    #     rule1 = (not np.isnan(gap)) and (gap <= UNLOAD_WINDOW_MS)
    #     if not rule1:
    #         continue
    #     other_ts = df.loc[collect_mask & (df.index != idx), "timestamp"]
    #     rule2 = any(
    #         abs((ts - ot).total_seconds() * 1000) <= BURST_WINDOW_MS
    #         for ot in other_ts
    #     )
    #     rule3 = bool(re.search(r"\.html$", str(row.get("referer", "") or "")))
    #     if rule2 or rule3:
    #         flagged.add(idx)

    # if not flagged:
    #     return df, 0
    # return df.drop(index=list(flagged)).reset_index(drop=True), len(flagged)


# ─── PARSERS ──────────────────────────────────────────────────────────────────

CH_UA_RE = re.compile(r'"([^"]+)";v="(\d+)"')

def parse_ch_ua(s: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in CH_UA_RE.finditer(s or "")]

def ua_browser_family(ua: str) -> str:
    ua = ua or ""
    if "HeadlessChrome" in ua or "headless" in ua.lower(): return "headless-chrome"
    if "Edg/" in ua:                                        return "edge"
    if "Chrome/" in ua:                                     return "chrome"
    if "Firefox/" in ua:                                    return "firefox"
    if "Safari/" in ua and "Chrome" not in ua:              return "safari"
    if "curl/" in ua:                                       return "curl"
    if "python" in ua.lower():                              return "python"
    if "Go-http-client" in ua:                              return "go"
    if "okhttp" in ua.lower():                              return "okhttp"
    return "other"

def ua_os(ua: str) -> str:
    ua = ua or ""
    if "Windows" in ua:                          return "windows"
    if "Macintosh" in ua or "Mac OS" in ua:      return "macos"
    if "Linux" in ua and "Android" not in ua:    return "linux"
    if "Android" in ua:                          return "android"
    if "iPhone" in ua or "iPad" in ua:           return "ios"
    return "other"

def ua_version(ua: str) -> str | None:
    for pat in [r"Chrome/(\d+)", r"Firefox/(\d+)", r"Safari/(\d+)"]:
        m = re.search(pat, ua or "")
        if m:
            return m.group(1)
    return None

def classify_request(method: str, path: str,
                     sf_mode: str, sf_dest: str, sf_site: str) -> str:
    if sf_mode == "navigate":                                   return "navigate"
    if sf_mode == "no-cors":                                    return f"subresource:{sf_dest or 'unknown'}"
    if sf_mode in ("cors", "same-origin") and sf_dest == "empty": return "fetch_xhr"
    if sf_mode == "cors":                                       return f"cors:{sf_dest or 'unknown'}"
    if method == "POST":                                        return "post"
    if path.endswith(".js"):                                    return "subresource:script"
    if path.endswith(".css"):                                   return "subresource:style"
    if path.endswith((".png",".jpg",".jpeg",".gif",".webp",".svg",".ico")):
                                                                return "subresource:image"
    return "other"


# ─── FEATURE: SEC-FETCH CONSISTENCY ──────────────────────────────────────────

def check_sec_fetch(sf_mode: str, sf_dest: str,
                    sf_site: str, sf_user: str, req_type: str) -> dict:
    sf_mode = sf_mode or ""; sf_dest = sf_dest or ""
    sf_site = sf_site or ""; sf_user = sf_user or ""

    out = {}
    key  = (sf_mode, sf_dest)
    rule = SEC_FETCH_RULES.get(key)
    out["sf_known_combo"] = rule is not None

    if rule:
        valid_sites, user_req = rule
        out["sf_site_valid"]   = (sf_site in valid_sites) if sf_site else True
        out["sf_user_correct"] = (sf_user == "?1") if user_req else (sf_user == "")
    else:
        out["sf_site_valid"] = out["sf_user_correct"] = True

    out["sf_nav_has_user"]       = (sf_user == "?1") if sf_mode == "navigate" else True
    out["sf_no_user_on_non_nav"] = (sf_user == "")   if sf_mode != "navigate" else True
    out["sf_cors_dest_empty"]    = (sf_dest == "empty") \
                                   if (sf_mode == "cors" and req_type == "fetch_xhr") else True
    out["sf_any_violation"]      = not all(out.values())
    return out


# ─── FEATURE: CLIENT HINTS COHERENCE ─────────────────────────────────────────

def check_client_hints(ch_ua: str, ch_mobile: str,
                       ch_platform: str, ua: str) -> dict:
    ua     = ua or ""
    brands = parse_ch_ua(ch_ua)
    bl     = [b.lower() for b, _ in brands]
    out    = {}

    out["ch_is_headless"] = any("headless" in b for b in bl)

    ua_is_chrome = "Chrome/" in ua or "Chromium/" in ua
    out["ch_brand_matches_ua"] = (
        any("chromium" in b or "chrome" in b or "edge" in b or "brand" in b for b in bl)
        if (ua_is_chrome and brands) else True
    )

    ua_ver = ua_version(ua)
    if ua_ver and brands:
        ch_vers = [v for _, v in brands if v > 10]
        out["ch_version_near_ua"] = (
            any(abs(v - int(ua_ver)) <= 2 for v in ch_vers) if ch_vers else True
        )
    else:
        out["ch_version_near_ua"] = True

    mobile_flag = (ch_mobile or "").strip() == "?1"
    ua_mobile   = any(kw in ua for kw in ["Mobile", "Android", "iPhone", "iPad"])
    out["ch_mobile_matches_ua"] = (mobile_flag == ua_mobile)

    platform    = (ch_platform or "").strip().strip('"').lower()
    os_map      = {"windows":"windows","macos":"macos","linux":"linux",
                   "android":"android","ios":"ios"}
    exp_platform = os_map.get(ua_os(ua), "")
    out["ch_platform_matches_ua"] = (
        (platform == exp_platform) if (platform and exp_platform) else True
    )

    out["ch_any_incoherence"] = not all(
        v for k, v in out.items() if k != "ch_is_headless"
    )
    return out


# ─── FEATURE: PRIORITY HEADER ─────────────────────────────────────────────────

def classify_priority(priority_val: str, req_type: str) -> dict:
    p = (priority_val or "").strip()
    out = {"priority_raw": p, "priority_urgency": None,
           "priority_incr": None, "priority_expected": None}
    if not p:
        return out
    m = re.match(r"u=(\d+)(?:,\s*i)?", p)
    if m:
        out["priority_urgency"] = int(m.group(1))
        out["priority_incr"]    = "i" in p
    if req_type == "navigate":
        out["priority_expected"] = p in PRIORITY_EXPECTED["navigate"]
    elif req_type.startswith("subresource"):
        out["priority_expected"] = p in PRIORITY_EXPECTED["subresource"]
    elif req_type == "fetch_xhr":
        out["priority_expected"] = p in PRIORITY_EXPECTED["async"]
    else:
        out["priority_expected"] = True
    return out


# ─── FEATURE: NON-STANDARD HEADERS ───────────────────────────────────────────

def detect_nonstandard_headers(ordered_names: list[str]) -> dict:
    """
    Compare the actual header names in this request against STANDARD_HEADERS.
    Returns:
      nonstandard_headers      : sorted comma-separated string of unknown names
      has_nonstandard_headers  : bool flag
      n_nonstandard_headers    : count
    """
    unknown = sorted(
        h for h in ordered_names if h.lower() not in STANDARD_HEADERS
    )
    return {
        "nonstandard_headers":     ",".join(unknown) if unknown else "",
        "has_nonstandard_headers": len(unknown) > 0,
        "n_nonstandard_headers":   len(unknown),
    }


# ─── PER-REQUEST PARSER ───────────────────────────────────────────────────────

def parse_request(rec: dict, agent: str, trial: str) -> dict:
    http            = rec.get("http") or {}
    headers_ordered = http.get("headers_ordered") or []
    headers         = http.get("headers") or {}

    def hval(name: str) -> str:
        vals = headers.get(name)
        return vals[0] if vals else ""

    method   = http.get("method", "")
    path     = http.get("path", "")
    protocol = http.get("protocol", "")

    sf_mode = http.get("sec_fetch_mode") or hval("Sec-Fetch-Mode")
    sf_dest = http.get("sec_fetch_dest") or hval("Sec-Fetch-Dest")
    sf_site = http.get("sec_fetch_site") or hval("Sec-Fetch-Site")
    sf_user = http.get("sec_fetch_user") or hval("Sec-Fetch-User")

    ch_ua       = http.get("sec_ch_ua")          or hval("Sec-Ch-Ua")
    ch_mobile   = http.get("sec_ch_ua_mobile")   or hval("Sec-Ch-Ua-Mobile")
    ch_platform = http.get("sec_ch_ua_platform") or hval("Sec-Ch-Ua-Platform")
    ua          = http.get("user_agent")         or hval("User-Agent")
    priority    = hval("Priority")

    req_type    = classify_request(method, path, sf_mode, sf_dest, sf_site)
    order_names = [h["name"] for h in headers_ordered]
    order_str   = ",".join(order_names)
    order_hash  = hashlib.sha256(order_str.encode()).hexdigest()[:16]

    row: dict = {
        # ── Identifiers ────────────────────────────────────────────────
        "agent":        agent,
        "trial":        trial,
        "request_id":   rec.get("id"),
        "timestamp":    rec.get("timestamp"),
        "method":       method,
        "path":         path,
        "protocol":     protocol,
        "req_type":     req_type,
        # ── 1. Header order fingerprint ────────────────────────────────
        "header_order_str":  order_str,
        "header_order_hash": order_hash,
        "n_headers":         len(order_names),
        # ── 2. Sec-Fetch raw values ────────────────────────────────────
        "sf_mode":  sf_mode,
        "sf_dest":  sf_dest,
        "sf_site":  sf_site,
        "sf_user":  sf_user,
        "sf_combo": f"{sf_mode}|{sf_dest}|{sf_site}",
        # ── 3. Client Hints raw ────────────────────────────────────────
        "ch_ua":           ch_ua,
        "ch_mobile":       ch_mobile,
        "ch_platform":     ch_platform,
        "ch_brand_count":  len(parse_ch_ua(ch_ua)),
        # ── 5. User-Agent ──────────────────────────────────────────────
        "ua":         ua,
        "ua_family":  ua_browser_family(ua),
        "ua_os":      ua_os(ua),
        "ua_version": ua_version(ua),
        # ── 6. Priority raw ────────────────────────────────────────────
        "priority": priority,
        # ── 4. Header presence flags ───────────────────────────────────
        **{f"has_{h.replace('-','_').lower()}": (h in order_names)
           for h in FINGERPRINT_HEADERS},
    }

    # ── 2. Sec-Fetch consistency checks ───────────────────────────────
    row.update(check_sec_fetch(sf_mode, sf_dest, sf_site, sf_user, req_type))

    # ── 3. Client Hints coherence ──────────────────────────────────────
    row.update(check_client_hints(ch_ua, ch_mobile, ch_platform, ua))

    # ── 6. Priority classification ─────────────────────────────────────
    row.update(classify_priority(priority, req_type))

    # ── 7. Non-standard header detection ──────────────────────────────
    row.update(detect_nonstandard_headers(order_names))

    return row


# ─── LOADERS ──────────────────────────────────────────────────────────────────

def load_trial(filepath: Path, agent: str, trial: str,
               do_filter: bool = True) -> pd.DataFrame:
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(parse_request(rec, agent, trial))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if do_filter:
        df, n_removed = filter_senbeacon(df)
        if n_removed:
            print(f"      removed {n_removed} sendBeacon POST(s)")

    return df


def discover_agents(base: Path) -> dict[str, list[Path]]:
    """
    Walk base_dir looking for <agent>/<trial>/requests.jsonl layout.
    Returns {agent_name: [path, ...]} sorted by trial name.
    """
    agents: dict[str, list[Path]] = {}
    for agent_dir in sorted(base.iterdir()):
        if not agent_dir.is_dir():
            continue
        files = sorted(agent_dir.glob("*/requests.jsonl"))
        if not files:
            # Fallback: *.jsonl directly inside agent_dir
            files = sorted(agent_dir.glob("*.jsonl"))
        if files:
            agents[agent_dir.name] = files
    return agents


def load_all(base: Path, agent_names: list[str] | None = None,
             do_filter: bool = True) -> pd.DataFrame:
    """
    Load every trial for every agent and return one concatenated DataFrame
    of per-request features.
    """
    agent_map = discover_agents(base)
    if agent_names:
        agent_map = {k: v for k, v in agent_map.items() if k in agent_names}

    if not agent_map:
        print("[!] No agents found.")
        return pd.DataFrame()

    all_dfs: list[pd.DataFrame] = []
    for agent, filepaths in agent_map.items():
        print(f"\n  Agent: {agent}  ({len(filepaths)} trial(s))")
        for fp in filepaths:
            trial = fp.parent.name
            print(f"    {trial} …", end=" ", flush=True)
            df = load_trial(fp, agent, trial, do_filter=do_filter)
            if df.empty:
                print("empty, skipped")
                continue
            print(f"{len(df)} requests")
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# ─── TRIAL-LEVEL AGGREGATION ──────────────────────────────────────────────────

def build_trial_features(req_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-request rows into one row per (agent, trial).

    All features are request-count-normalised (rates / modal values) so
    that trials with different numbers of requests are comparable.
    """
    rows = []
    for (agent, trial), grp in req_df.groupby(["agent", "trial"], sort=True):
        nav = grp[grp["req_type"] == "navigate"]
        sub = grp[grp["req_type"].str.startswith("subresource", na=False)]
        n   = len(grp)
        n_nav = len(nav)

        def rate(series: pd.Series) -> float:
            s = series.dropna()
            return float(s.astype(bool).mean()) if len(s) else float("nan")

        def modal(series: pd.Series):
            s = series.dropna()
            return s.mode().iloc[0] if len(s) else None

        def n_distinct(series: pd.Series) -> int:
            return series.dropna().nunique()

        # Header-order uniqueness on navigation requests
        if n_nav > 0:
            n_uniq_nav = nav["header_order_hash"].nunique()
            nav_order_unique_ratio = n_uniq_nav / n_nav
            nav_order_stability    = nav["header_order_hash"].value_counts().iloc[0] / n_nav
        else:
            nav_order_unique_ratio = float("nan")
            nav_order_stability    = float("nan")

        # Dominant non-standard header names seen in this trial
        ns_names: list[str] = []
        for raw in grp["nonstandard_headers"].dropna():
            ns_names.extend(h for h in raw.split(",") if h)
        top_ns = sorted({h for h in ns_names if h})  # unique sorted set

        row = {
            # ── Identifiers ────────────────────────────────────────────
            "agent":  agent,
            "trial":  trial,
            "n_requests": n,
            "n_nav":      n_nav,
            "n_sub":      len(sub),

            # ── 1. Header order ────────────────────────────────────────
            "nav_order_unique_ratio": nav_order_unique_ratio,
            "nav_order_stability":    nav_order_stability,
            "n_unique_nav_hashes":    nav["header_order_hash"].nunique() if n_nav else 0,
            "n_headers_mean":         grp["n_headers"].mean(),
            "n_headers_std":          grp["n_headers"].std(),

            # ── 2. Sec-Fetch ───────────────────────────────────────────
            "sf_site_none_rate":     rate(grp["sf_site"] == "none"),
            "sf_violation_rate":     rate(grp["sf_any_violation"]),
            "sf_site_mode":          modal(grp["sf_site"]),

            # ── 3. Client Hints / headless ─────────────────────────────
            "headless_rate":         rate(grp["ch_is_headless"]),
            "ch_incoherence_rate":   rate(grp["ch_any_incoherence"]),
            "ch_brand_matches_rate": rate(grp["ch_brand_matches_ua"]),
            "ch_mobile_matches_rate":rate(grp["ch_mobile_matches_ua"]),
            "ch_platform_matches_rate":rate(grp["ch_platform_matches_ua"]),
            "ch_brand_count_mode":   modal(grp["ch_brand_count"]),

            # ── 4. Header presence rates ───────────────────────────────
            "accept_lang_rate":      rate(grp["has_accept_language"]),
            "has_priority_rate":     rate(grp["has_priority"]),
            "has_sec_ch_ua_rate":    rate(grp["has_sec_ch_ua"]),
            "has_upgrade_ins_rate":  rate(grp["has_upgrade_insecure_requests"]),
            "has_dnt_rate":          rate(grp["has_dnt"]),
            "has_te_rate":           rate(grp["has_te"]),
            "has_connection_rate":   rate(grp["has_connection"]),
            "has_cache_control_rate":rate(grp["has_cache_control"]),

            # ── 5. User-Agent ──────────────────────────────────────────
            "ua_family_mode":        modal(grp["ua_family"]),
            "ua_os_mode":            modal(grp["ua_os"]),
            "ua_version_mode":       modal(grp["ua_version"]),
            "ua_version_n_distinct": n_distinct(grp["ua_version"]),
            "ua_family_n_distinct":  n_distinct(grp["ua_family"]),

            # ── 6. Priority ────────────────────────────────────────────
            "priority_correct_rate": rate(grp["priority_expected"]),
            "priority_urgency_mode": modal(grp["priority_urgency"]),

            # ── 7. Non-standard headers ────────────────────────────────
            "has_nonstandard_rate":  rate(grp["has_nonstandard_headers"]),
            "nonstandard_names":     "|".join(top_ns) if top_ns else "",
            "n_nonstandard_distinct":len(top_ns),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ─── AGENT-LEVEL PROFILE BUILDER ─────────────────────────────────────────────

# Numeric trial features that get mean ± std aggregation
NUMERIC_TRIAL_FEATURES = [
    "nav_order_unique_ratio",
    "nav_order_stability",
    "n_headers_mean",
    "sf_site_none_rate",
    "sf_violation_rate",
    "headless_rate",
    "ch_incoherence_rate",
    "ch_brand_matches_rate",
    "ch_mobile_matches_rate",
    "ch_platform_matches_rate",
    "accept_lang_rate",
    "has_priority_rate",
    "has_sec_ch_ua_rate",
    "has_upgrade_ins_rate",
    "has_dnt_rate",
    "has_te_rate",
    "has_connection_rate",
    "has_cache_control_rate",
    "ua_version_n_distinct",
    "ua_family_n_distinct",
    "priority_correct_rate",
    "has_nonstandard_rate",
    "n_nonstandard_distinct",
]

# Categorical trial features that get mode + consistency-rate aggregation
CATEGORICAL_TRIAL_FEATURES = [
    "ua_family_mode",
    "ua_os_mode",
    "ua_version_mode",
    "sf_site_mode",
    "priority_urgency_mode",
    "ch_brand_count_mode",
]


def build_agent_profiles(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trial-level rows into one row per agent.

    For each numeric feature   → mean, std, min, max across trials
    For each categorical feature → mode value + consistency rate
    Non-standard headers        → union of all observed names + prevalence

    The returned DataFrame has one row per agent, with columns:
      <feature>_mean, <feature>_std  (numeric)
      <feature>_mode, <feature>_consistency  (categorical)
      nonstandard_names_union, nonstandard_prevalence
    """
    rows = []
    for agent, grp in trial_df.groupby("agent", sort=True):
        row: dict = {"agent": agent, "n_trials": len(grp)}

        # ── Numeric features ───────────────────────────────────────────
        for feat in NUMERIC_TRIAL_FEATURES:
            if feat not in grp.columns:
                continue
            vals = pd.to_numeric(grp[feat], errors="coerce").dropna()
            if vals.empty:
                row[f"{feat}_mean"] = float("nan")
                row[f"{feat}_std"]  = float("nan")
                row[f"{feat}_min"]  = float("nan")
                row[f"{feat}_max"]  = float("nan")
            else:
                row[f"{feat}_mean"] = round(vals.mean(), 4)
                row[f"{feat}_std"]  = round(vals.std(), 4)
                row[f"{feat}_min"]  = round(vals.min(), 4)
                row[f"{feat}_max"]  = round(vals.max(), 4)

        # ── Categorical features ───────────────────────────────────────
        for feat in CATEGORICAL_TRIAL_FEATURES:
            if feat not in grp.columns:
                continue
            vals = grp[feat].dropna().astype(str)
            vals = vals[vals.str.strip() != ""]
            if vals.empty:
                row[f"{feat}_mode"]        = None
                row[f"{feat}_consistency"] = float("nan")
            else:
                top   = vals.mode().iloc[0]
                cons  = (vals == top).mean()
                row[f"{feat}_mode"]        = top
                row[f"{feat}_consistency"] = round(cons, 4)

        # ── Non-standard headers: union + prevalence ───────────────────
        all_ns: Counter = Counter()
        for raw in grp["nonstandard_names"].fillna(""):
            for h in raw.split("|"):
                h = h.strip()
                if h:
                    all_ns[h] += 1
        if all_ns:
            prevalence = {h: round(cnt / len(grp), 3)
                          for h, cnt in all_ns.most_common()}
            row["nonstandard_names_union"]   = "|".join(prevalence.keys())
            row["nonstandard_prevalence"]    = str(prevalence)
        else:
            row["nonstandard_names_union"] = ""
            row["nonstandard_prevalence"]  = "{}"

        rows.append(row)

    return pd.DataFrame(rows)


# ─── DISCRIMINATIVE POWER SUMMARY ────────────────────────────────────────────

def feature_discriminability(agent_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric *_mean feature in the agent profile, compute:
      • inter-agent std  (how spread out the agents are — higher = more useful)
      • mean intra-agent std  (how noisy the feature is within each agent)
      • discriminability ratio = inter_std / (mean_intra_std + ε)

    Higher ratio → better classifier feature.
    """
    mean_cols = [c for c in agent_profiles.columns if c.endswith("_mean")]
    rows = []
    for col in mean_cols:
        feat     = col[:-5]   # strip "_mean"
        std_col  = f"{feat}_std"
        inter_std = agent_profiles[col].std()
        intra_std = (agent_profiles[std_col].mean()
                     if std_col in agent_profiles.columns else float("nan"))
        ratio = inter_std / (intra_std + 1e-9) if not np.isnan(intra_std) else float("nan")
        rows.append({
            "feature":        feat,
            "inter_agent_std":round(inter_std, 4),
            "mean_intra_std": round(intra_std, 4),
            "discrim_ratio":  round(ratio, 3),
        })

    return (pd.DataFrame(rows)
            .sort_values("discrim_ratio", ascending=False)
            .reset_index(drop=True))


# ─── CONSOLE SUMMARY ──────────────────────────────────────────────────────────

def print_agent_summary(profiles: pd.DataFrame, discrim: pd.DataFrame) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("  AGENT PROFILES  (key discriminating features)")
    print(sep)

    KEY_FEATURES = [
        ("headless_rate",          "Headless rate"),
        ("accept_lang_rate",       "Accept-Language rate"),
        ("sf_site_none_rate",      "Sec-Fetch-Site=none rate"),
        ("sf_violation_rate",      "Sec-Fetch violation rate"),
        ("ch_incoherence_rate",    "CH incoherence rate"),
        ("nav_order_unique_ratio", "Nav order unique ratio"),
        ("priority_correct_rate",  "Priority correct rate"),
        ("has_nonstandard_rate",   "Non-standard header rate"),
        ("ua_version_n_distinct",  "UA version #distinct"),
    ]

    header = f"  {'Feature':<32}" + "".join(
        f"{a:<14}" for a in profiles["agent"]
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for feat, label in KEY_FEATURES:
        mean_col = f"{feat}_mean"
        std_col  = f"{feat}_std"
        if mean_col not in profiles.columns:
            continue
        line = f"  {label:<32}"
        for _, arow in profiles.iterrows():
            mu  = arow.get(mean_col, float("nan"))
            sig = arow.get(std_col,  float("nan"))
            if np.isnan(mu):
                cell = "--"
            elif np.isnan(sig):
                cell = f"{mu:.3f}"
            else:
                cell = f"{mu:.3f}±{sig:.3f}"
            line += f"{cell:<14}"
        print(line)

    # Categorical
    print()
    for feat in ["ua_family_mode", "sf_site_mode"]:
        mode_col = f"{feat}_mode"
        cons_col = f"{feat}_consistency"
        if mode_col not in profiles.columns:
            continue
        line = f"  {feat:<32}"
        for _, arow in profiles.iterrows():
            val  = str(arow.get(mode_col, "--") or "--")[:12]
            cons = arow.get(cons_col, float("nan"))
            suffix = f"({cons:.0%})" if not np.isnan(cons) else ""
            line += f"{val}{suffix:<14}"[:14]
        print(line)

    # Non-standard headers
    print()
    print(f"  {'Non-standard headers':<32}", end="")
    for _, arow in profiles.iterrows():
        ns = str(arow.get("nonstandard_names_union", "") or "none")[:13]
        print(f"{ns:<14}", end="")
    print()

    print(f"\n{sep}")
    print("  TOP-10 MOST DISCRIMINATIVE FEATURES  (inter/intra std ratio)")
    print(sep)
    print(discrim.head(10).to_string(index=False))
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP header agent profiler — extracts features from requests.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python agent_header_profiler.py --base ./traces
  python agent_header_profiler.py --base ./traces --out ./results
  python agent_header_profiler.py --base ./traces --agents autogen skyvern
  python agent_header_profiler.py --base ./traces --no-filter
""")
    parser.add_argument("--root",    required=True, metavar="DIR",
                        help="root directory containing one sub-dir per agent")
    parser.add_argument("--out",     default=".",   metavar="DIR",
                        help="output directory (default: .)")
    parser.add_argument("--agents",  nargs="+",     metavar="AGENT",
                        help="restrict to these agent names (default: all found)")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip sendBeacon POST /collect filter")
    args = parser.parse_args()

    base    = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base.is_dir():
        sys.exit(f"[!] Base directory not found: {base}")

    do_filter = not args.no_filter
    print(f"\nBase directory : {base.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"sendBeacon filter: {'ON' if do_filter else 'OFF'}")

    # ── 1. Load all requests ──────────────────────────────────────────
    print("\n[1/4] Loading requests …")
    req_df = load_all(base, agent_names=args.agents, do_filter=do_filter)

    if req_df.empty:
        sys.exit("[!] No requests loaded. Check directory layout.")

    req_path = out_dir / "http_request_features.csv"
    req_df.to_csv(req_path, index=False)
    print(f"\n  Saved: {req_path}  ({len(req_df)} rows)")

    # ── 2. Trial-level features ───────────────────────────────────────
    print("\n[2/4] Aggregating trial-level features …")
    trial_df = build_trial_features(req_df)

    trial_path = out_dir / "http_features.csv"
    trial_df.to_csv(trial_path, index=False)
    print(f"  Saved: {trial_path}  ({len(trial_df)} rows)")

    # ── 3. Agent-level profiles ───────────────────────────────────────
    print("\n[3/4] Building agent profiles …")
    agent_df = build_agent_profiles(trial_df)

    agent_path = out_dir / "agent_profiles.csv"
    agent_df.to_csv(agent_path, index=False)
    print(f"  Saved: {agent_path}  ({len(agent_df)} agents)")

    # ── 4. Discriminability ranking ───────────────────────────────────
    print("\n[4/4] Computing feature discriminability …")
    discrim_df = feature_discriminability(agent_df)

    discrim_path = out_dir / "feature_discriminability.csv"
    discrim_df.to_csv(discrim_path, index=False)
    print(f"  Saved: {discrim_path}")

    # ── Console summary ───────────────────────────────────────────────
    print_agent_summary(agent_df, discrim_df)

    print("Done.\n")


if __name__ == "__main__":
    main()