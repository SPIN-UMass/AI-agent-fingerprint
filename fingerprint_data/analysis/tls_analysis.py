"""
TLS Fingerprint Analyzer  —  tls_analysis.py
==============================================
Analyzes TLS / HTTP2 fingerprints from requests.jsonl files collected
across multiple trials, with the goal of distinguishing AI agents by
their TLS client identity.

Directory layout expected (same as trace_analysis.py):
    traces/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
      ...

What is analyzed
----------------
Layer 1 — JA4 decomposition  (JA4_A / JA4_B / JA4_C)
  • protocol          : t=TLS, q=QUIC, d=DTLS
  • tls_version       : negotiated version encoded in JA4 (13=TLS1.3, 12=TLS1.2)
  • sni_presence      : 'd'=domain SNI sent, 'i'=no SNI (IP or absent)
  • cipher_count      : number of ciphers offered (excl. GREASE)
  • ext_count         : number of extensions offered (excl. GREASE)
  • alpn_first        : first ALPN protocol code ('h2', 'http/1.1', '00'=none)
  • ja4_b             : 12-char truncated SHA-256 of sorted cipher suite IDs
  • ja4_c             : 12-char truncated SHA-256 of sorted extension IDs + sig schemes

Layer 2 — Raw TLS fields
  • tls_version       : negotiated version string ("TLS 1.3", "TLS 1.2")
  • negotiated_proto  : ALPN result ("h2", "http/1.1")
  • cipher_suite      : negotiated cipher suite name
  • sni               : SNI hostname (presence/absence)
  • cipher_suites_offered : ordered list (with/without GREASE)
  • elliptic_curves   : ordered list (with/without GREASE)
  • signature_schemes : ordered list
  • supported_versions: ordered list
  • alpn_protocols    : ordered list
  • has_grease        : whether GREASE values appear in cipher/curves

Layer 3 — Akamai HTTP/2 fingerprint
  • h2_settings       : SETTINGS frame values (6 known settings)
  • h2_window_update  : WINDOW_UPDATE initial size
  • h2_pseudo_order   : pseudo-header order string (e.g. "m,a,s,p")
  • h2_priority_tree  : stream priority tree (grows per request in session)
  • h2_stream_count   : number of priority tree entries (= request depth)

Layer 4 — Cross-request consistency (within a trace)
  • Do fingerprints stay identical across all requests in a session?
  • Which fields drift and which are rock-solid?

Layer 5 — Cross-trace consistency (across 30 trials)
  • Stability score per field: fraction of traces where value == modal value
  • Fingerprint uniqueness: how many distinct fingerprints per field?

Layer 6 — Agent-distinguishing signals  (the key output)
  • Features ranked by discriminative power (entropy across agents/systems)
  • A fingerprint feature vector suitable for ML classification

Outputs
-------
  tls_summary.csv              — per-request parsed features, all traces
  tls_stability.csv            — per-field stability scores across traces
  tls_fingerprint_vector.csv   — one row per trace, feature vector
  tls_analysis_single.png      — single-trace detail (6 panels)
  tls_analysis_cross.png       — cross-trace comparison (8 panels)

Usage
-----
  python tls_analysis.py --traces ./traces
  python tls_analysis.py --traces ./traces --out ./results
  python tls_analysis.py --files trial-001/requests.jsonl trial-002/requests.jsonl
"""

import json, re, sys, glob, argparse, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── LOOKUP TABLES ────────────────────────────────────────────────────────────

CIPHER_NAMES = {
    4865:  "TLS_AES_128_GCM_SHA256",
    4866:  "TLS_AES_256_GCM_SHA384",
    4867:  "TLS_CHACHA20_POLY1305_SHA256",
    49195: "ECDHE-ECDSA-AES128-GCM",
    49199: "ECDHE-RSA-AES128-GCM",
    49196: "ECDHE-ECDSA-AES256-GCM",
    49200: "ECDHE-RSA-AES256-GCM",
    52393: "ECDHE-ECDSA-CHACHA20",
    52392: "ECDHE-RSA-CHACHA20",
    49171: "ECDHE-RSA-AES128-CBC",
    49172: "ECDHE-RSA-AES256-CBC",
    156:   "RSA-AES128-GCM",
    157:   "RSA-AES256-GCM",
    47:    "RSA-AES128-CBC",
    53:    "RSA-AES256-CBC",
    0xFF:  "EMPTY_RENEGOTIATION",
}

CURVE_NAMES = {
    29:    "x25519",
    23:    "secp256r1",
    24:    "secp384r1",
    25:    "secp521r1",
    25497: "x25519Kyber768",   # PQ hybrid — Chrome 124+
    4588:  "X25519MLKEM768",   # PQ hybrid — Chrome 131+
}

SIG_NAMES = {
    1027: "ecdsa_secp256r1_sha256",
    2052: "rsa_pss_rsae_sha256",
    1025: "rsa_pkcs1_sha256",
    1283: "ecdsa_secp384r1_sha384",
    2053: "rsa_pss_rsae_sha384",
    1281: "rsa_pkcs1_sha384",
    2054: "rsa_pss_rsae_sha512",
    1537: "rsa_pkcs1_sha512",
    1540: "rsa_pkcs1_sha1",
    515:  "ecdsa_sha1",
}

# supported_versions numeric IDs → readable names
# 0x?A?A values are GREASE (RFC 8701)
SUPPORTED_VERSION_NAMES = {
    772: "TLS 1.3",    # 0x0304
    771: "TLS 1.2",    # 0x0303
    770: "TLS 1.1",    # 0x0302
    769: "TLS 1.0",    # 0x0301
    768: "SSL 3.0",    # 0x0300
}

# ALPN protocol string → short readable label
ALPN_LABEL = {
    "h2":            "HTTP/2",
    "http/1.1":      "HTTP/1.1",
    "http/1.0":      "HTTP/1.0",
    "spdy/3.1":      "SPDY/3.1",
    "spdy/3":        "SPDY/3",
    "quic":          "QUIC",
}

# GREASE values — RFC 8701: any value of the form 0x?A?A
def is_grease(val: int) -> bool:
    return isinstance(val, int) and (val & 0x0F0F) == 0x0A0A

# ── sendBeacon filter constants ──────────────────────────────────────────────
# POST /collect rows caused by logger.js sendBeacon are instrumentation
# artifacts — not genuine browsing traffic. Three rules (same as
# trace_analysis.py) identify and remove them:
#   Rule 1: POST arrives within UNLOAD_WINDOW_MS of the next page GET
#            (sendBeacon fires synchronously at beforeunload/pagehide)
#   Rule 2: Multiple /collect POSTs cluster within BURST_WINDOW_MS of each
#            other (beforeunload + pagehide + visibilitychange all fire together)
#   Rule 3: Referer header ends in .html (logger.js always sets this)
# Rows are removed when Rule 1 AND (Rule 2 OR Rule 3) are true.
COLLECT_ENDPOINT   = "/collect"
UNLOAD_WINDOW_MS   = 500   # max ms gap between POST and next page GET (Rule 1)
BURST_WINDOW_MS    = 50    # max ms gap between POSTs in same burst (Rule 2)

# Known JA4 ALPN two-char codes → readable
ALPN_CODE = {
    "h2":   "h2",
    "h1":   "http/1.1",
    "00":   "none",
    "ftp":  "ftp",
    "imap": "imap",
    "smtp": "smtp",
}

# ─── JA4 PARSER ───────────────────────────────────────────────────────────────

def parse_ja4(ja4: str) -> dict:
    """
    Decompose a JA4 hash string into its constituent fields.

    JA4 format:  {JA4_A}_{JA4_B}_{JA4_C}
    JA4_A format: {protocol}{version}{sni}{cipher_count}{ext_count}{alpn_first}
                   e.g.  t  13       d    15            16          h2
    """
    result = {
        "ja4_raw":      ja4,
        "ja4_a":        None,
        "ja4_b":        None,
        "ja4_c":        None,
        "ja4_protocol": None,
        "ja4_version":  None,
        "ja4_sni":      None,
        "ja4_cipher_count": None,
        "ja4_ext_count":    None,
        "ja4_alpn_code":    None,
    }
    if not ja4 or not isinstance(ja4, str):
        return result

    parts = ja4.split("_")
    if len(parts) != 3:
        return result

    a, b, c = parts
    result["ja4_a"] = a
    result["ja4_b"] = b
    result["ja4_c"] = c

    # Parse JA4_A character by character
    # Format: t13d1516h2  (10 chars)
    if len(a) >= 10:
        result["ja4_protocol"]    = a[0]           # t/q/d
        result["ja4_version"]     = a[1:3]         # 13/12/10
        result["ja4_sni"]         = a[3]           # d=domain, i=IP
        result["ja4_cipher_count"]= int(a[4:6]) if a[4:6].isdigit() else None
        result["ja4_ext_count"]   = int(a[6:8]) if a[6:8].isdigit() else None
        result["ja4_alpn_code"]   = a[8:10]        # h2/h1/00 etc.

    return result


# ─── AKAMAI PARSER ───────────────────────────────────────────────────────────

def parse_akamai(fp: str) -> dict:
    """
    Decompose Akamai HTTP/2 fingerprint into its four sections.

    Format: {SETTINGS}|{WINDOW_UPDATE}|{PRIORITY_FRAMES}|{PSEUDO_HEADER_ORDER}

    SETTINGS: param_id:value pairs, semicolon-separated
      1=HEADER_TABLE_SIZE, 2=ENABLE_PUSH, 3=MAX_CONCURRENT_STREAMS,
      4=INITIAL_WINDOW_SIZE, 5=MAX_FRAME_SIZE, 6=MAX_HEADER_LIST_SIZE

    PRIORITY_FRAMES: stream_id:exclusive:dep_stream:weight, comma-separated
      — grows by one entry per new request on the same connection

    PSEUDO_HEADER_ORDER: m=:method, a=:authority, s=:scheme, p=:path
    """
    result = {
        "ak_raw":              fp,
        "ak_settings_str":     None,
        "ak_window_update":    None,
        "ak_priority_str":     None,
        "ak_pseudo_order":     None,
        "ak_stream_count":     None,   # depth of priority tree
        "ak_header_table":     None,
        "ak_enable_push":      None,
        "ak_max_concurrent":   None,
        "ak_init_window":      None,
        "ak_max_frame":        None,
        "ak_max_header_list":  None,
    }
    if not fp or not isinstance(fp, str):
        return result

    sections = fp.split("|")
    if len(sections) < 2:
        return result

    result["ak_settings_str"]  = sections[0] if len(sections) > 0 else None
    result["ak_window_update"]  = int(sections[1]) if len(sections) > 1 and sections[1].isdigit() else None
    result["ak_priority_str"]   = sections[2] if len(sections) > 2 else None
    result["ak_pseudo_order"]   = sections[3] if len(sections) > 3 else None

    # Parse stream priority count
    if result["ak_priority_str"]:
        result["ak_stream_count"] = len(result["ak_priority_str"].split(","))

    # Parse SETTINGS key:value pairs
    setting_map = {"1": "ak_header_table", "2": "ak_enable_push",
                   "3": "ak_max_concurrent", "4": "ak_init_window",
                   "5": "ak_max_frame",      "6": "ak_max_header_list"}
    for pair in sections[0].split(";"):
        if ":" in pair:
            k, v = pair.split(":", 1)
            col = setting_map.get(k.strip())
            if col and v.strip().lstrip("-").isdigit():
                result[col] = int(v.strip())

    return result


# ─── CIPHER / CURVE SEQUENCE FEATURES ────────────────────────────────────────

def sequence_features(seq: list, name: str, lookup: dict) -> dict:
    """
    Extract features from an ordered sequence of IDs (ciphers, curves, sigs).
    Returns a flat dict of named features.
    """
    if not seq:
        return {f"{name}_count": 0, f"{name}_has_grease": False,
                f"{name}_grease_pos": None, f"{name}_seq_str": "",
                f"{name}_no_grease_str": ""}

    has_grease   = any(is_grease(v) for v in seq)
    grease_pos   = next((i for i, v in enumerate(seq) if is_grease(v)), None)
    no_grease    = [v for v in seq if not is_grease(v)]
    named        = [lookup.get(v, str(v)) for v in no_grease]

    return {
        f"{name}_count":        len(no_grease),
        f"{name}_has_grease":   has_grease,
        f"{name}_grease_pos":   grease_pos,          # position of first GREASE (0=first)
        f"{name}_seq_str":      ",".join(str(v) for v in seq),
        f"{name}_no_grease_str":"_".join(named),     # human-readable, GREASE stripped
    }


# ─── SINGLE-REQUEST PARSER ───────────────────────────────────────────────────

def parse_request(rec: dict, trace_label: str) -> dict:
    """Parse one JSON record from requests.jsonl into a flat feature dict."""
    tls  = rec.get("tls") or {}
    h2   = rec.get("http2") or {}
    http = rec.get("http") or {}

    row = {
        "trace":       trace_label,
        "request_id":  rec.get("id"),
        "timestamp":   rec.get("timestamp"),
        "source_ip":   rec.get("source_ip"),
        "method":      http.get("method"),
        "path":        http.get("path"),
        "protocol":    http.get("protocol"),
        "proc_ms":     rec.get("processing_time_ms"),
    }

    # ── Raw TLS fields ────────────────────────────────────────────────────
    row["tls_version"]        = tls.get("version")
    row["negotiated_proto"]   = tls.get("negotiated_protocol")
    row["cipher_suite"]       = tls.get("cipher_suite")
    row["sni"]                = tls.get("server_name")
    row["sni_present"]        = bool(tls.get("server_name"))
    row["ja3"]                = tls.get("ja3_hash")
    row["ja4_raw"]            = tls.get("ja4_hash")
    # ── ALPN protocols (full ordered list) ───────────────────────────────
    alpn_list = tls.get("alpn_protocols") or []
    row["alpn_offered_str"]  = ",".join(alpn_list)
    row["alpn_first"]        = alpn_list[0] if alpn_list else None
    row["alpn_count"]        = len(alpn_list)
    row["alpn_has_h2"]       = "h2" in alpn_list
    row["alpn_has_http11"]   = "http/1.1" in alpn_list
    row["alpn_h2_position"]  = alpn_list.index("h2") if "h2" in alpn_list else None
    # Preference: h2_only / h2_first / http11_first / http11_only / other
    if alpn_list == ["h2"]:
        row["alpn_preference"] = "h2_only"
    elif alpn_list and alpn_list[0] == "h2":
        row["alpn_preference"] = "h2_first"
    elif alpn_list == ["http/1.1"]:
        row["alpn_preference"] = "http11_only"
    elif alpn_list and alpn_list[0] == "http/1.1":
        row["alpn_preference"] = "http11_first"
    elif not alpn_list:
        row["alpn_preference"] = "none"
    else:
        row["alpn_preference"] = "other"

    # ── Supported versions (full ordered list) ────────────────────────────
    ver_list = tls.get("supported_versions") or []
    ver_no_grease  = [v for v in ver_list if not is_grease(v)]
    ver_named      = [SUPPORTED_VERSION_NAMES.get(v, f"0x{v:04X}") for v in ver_no_grease]
    row["supported_ver_str"]        = ",".join(str(v) for v in ver_list)
    row["supported_ver_names"]      = ",".join(ver_named)          # e.g. "TLS 1.3,TLS 1.2"
    row["supported_ver_count"]      = len(ver_no_grease)
    row["supported_ver_has_grease"] = any(is_grease(v) for v in ver_list)
    row["supported_ver_has_tls13"]  = 772 in ver_no_grease
    row["supported_ver_has_tls12"]  = 771 in ver_no_grease
    row["supported_ver_has_tls11"]  = 770 in ver_no_grease
    row["supported_ver_grease_pos"] = next(
        (i for i, v in enumerate(ver_list) if is_grease(v)), None
    )
    # Canonical profile string (GREASE stripped, ordered as offered)
    # e.g. "TLS1.3+TLS1.2" — stable identifier for cross-agent comparison
    row["supported_ver_profile"]    = "+".join(
        v.replace("TLS ", "TLS").replace(" ", "") for v in ver_named
    ) or "none"
    # Is TLS 1.3 offered first (before 1.2)? Browsers always do this.
    if 772 in ver_no_grease and 771 in ver_no_grease:
        row["supported_ver_13_before_12"] = ver_no_grease.index(772) < ver_no_grease.index(771)
    else:
        row["supported_ver_13_before_12"] = None

    # ── Cipher suites ─────────────────────────────────────────────────────
    ciphers = tls.get("cipher_suites_offered") or []
    row.update(sequence_features(ciphers, "cipher", CIPHER_NAMES))

    # ── Elliptic curves ───────────────────────────────────────────────────
    curves = tls.get("elliptic_curves") or []
    row.update(sequence_features(curves, "curve", CURVE_NAMES))

    # ── Signature schemes ─────────────────────────────────────────────────
    sigs = tls.get("signature_schemes") or []
    row.update(sequence_features(sigs, "sig", SIG_NAMES))

    # PQ crypto presence (x25519Kyber768 / X25519MLKEM768)
    row["has_pq_crypto"] = any(v in (25497, 4588) for v in curves)

    # ── JA4 decomposition ─────────────────────────────────────────────────
    row.update(parse_ja4(tls.get("ja4_hash", "")))

    # ── Akamai / HTTP2 fingerprint ────────────────────────────────────────
    row.update(parse_akamai(h2.get("akamai_fingerprint", "")))
    row["h2_pseudo_order"] = ",".join(h2.get("pseudo_header_order") or [])

    return row


# ─── TRACE LOADER ────────────────────────────────────────────────────────────

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


# ─── SENBEACON FILTER ────────────────────────────────────────────────────────

def filter_senbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove POST /collect rows that are sendBeacon flushes from logger.js.

    These are experiment instrumentation artifacts — a real website would not
    have them.  Removal criteria (all three rules derived from logger.js source):

      Rule 1 — Temporal: the POST arrives within UNLOAD_WINDOW_MS before the
                next page GET.  sendBeacon fires synchronously at beforeunload/
                pagehide so the gap is typically < 50 ms on the same connection.

      Rule 2 — Burst: multiple /collect POSTs within BURST_WINDOW_MS of each
                other (beforeunload + pagehide + visibilitychange triple-fire).

      Rule 3 — Referer: Referer header ends in .html — logger.js always posts
                from the page being unloaded, so the Referer is always a page URL.

    A row is removed when Rule 1 AND (Rule 2 OR Rule 3) are satisfied.
    Returns (filtered_df, n_removed).
    """
    if df.empty:
        return df, 0

    collect_mask = (df["method"] == "POST") & (df["path"] == COLLECT_ENDPOINT)
    collect_idx  = df.index[collect_mask].tolist()
    if not collect_idx:
        return df, 0

    # Timestamps of page GETs for Rule 1
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
        row = df.loc[idx]
        ts  = row["timestamp"]

        # Rule 1
        gap   = ms_to_next_page(ts)
        rule1 = (not np.isnan(gap)) and (gap <= UNLOAD_WINDOW_MS)
        if not rule1:
            continue

        # Rule 2 — burst cluster
        other_ts = df.loc[collect_mask & (df.index != idx), "timestamp"]
        rule2 = any(
            abs((ts - ot).total_seconds() * 1000) <= BURST_WINDOW_MS
            for ot in other_ts
        )

        # Rule 3 — Referer ends in .html
        rule3 = bool(re.search(r"\.html$", str(row.get("referer", "") or "")))

        if rule2 or rule3:
            flagged.add(idx)

    if not flagged:
        return df, 0

    filtered = df.drop(index=list(flagged)).reset_index(drop=True)
    return filtered, len(flagged)


def label_for(fp: Path) -> str:
    return fp.parent.name if fp.parent.name else fp.stem


def collect_filepaths(traces_dir: str | Path,
                      filename: str = "requests.jsonl") -> list[Path]:
    root  = Path(traces_dir)
    found = sorted(root.glob(f"*/{filename}"))
    return found if found else sorted(root.glob("*.jsonl"))


def load_all(filepaths: list[Path],
             filter_senbeacon: bool = True) -> dict[str, pd.DataFrame]:
    traces = {}
    for fp in filepaths:
        label = label_for(fp)
        df = load_trace(fp, label)
        if df.empty:
            print(f"  [warn] {fp} – empty or unparseable, skipped")
            continue
        if filter_senbeacon:
            df, n_removed = globals()["filter_senbeacon"](df)
            suffix = f", removed {n_removed} sendBeacon POST(s)" if n_removed else ""
        else:
            suffix = " (no filter)"
        traces[label] = df
        print(f"  {label}: {len(df)} requests{suffix}")
    return traces


# ─── WITHIN-TRACE CONSISTENCY ────────────────────────────────────────────────

# Fields that should be rock-solid within a single browser session
CONSISTENCY_FIELDS = [
    "tls_version", "negotiated_proto", "ja3", "ja4_raw",
    "ja4_a", "ja4_b", "ja4_c",
    "ja4_protocol", "ja4_version", "ja4_sni",
    "ja4_cipher_count", "ja4_ext_count", "ja4_alpn_code",
    "cipher_count", "cipher_seq_str", "cipher_no_grease_str",
    "curve_count", "curve_seq_str",
    "sig_count", "sig_seq_str",
    "cipher_has_grease", "curve_has_grease",
    "has_pq_crypto",
    "alpn_first", "alpn_offered_str", "alpn_count",
    "alpn_has_h2", "alpn_has_http11", "alpn_preference",
    "alpn_h2_position",
    "supported_ver_str", "supported_ver_names", "supported_ver_count",
    "supported_ver_has_tls13", "supported_ver_has_tls12", "supported_ver_has_tls11",
    "supported_ver_has_grease", "supported_ver_profile", "supported_ver_13_before_12",
    "ak_window_update", "ak_pseudo_order",
    "ak_header_table", "ak_enable_push", "ak_init_window",
    "ak_max_header_list",
]

def within_trace_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each field in CONSISTENCY_FIELDS, compute how many distinct values
    appear across all requests in this trace (should be 1 if stable).
    Also note whether ak_stream_count grows monotonically (expected behaviour).
    """
    rows = []
    for field in CONSISTENCY_FIELDS:
        if field not in df.columns:
            continue
        vals    = df[field].dropna()
        n_uniq  = vals.nunique()
        modal   = vals.mode().iloc[0] if not vals.empty else None
        rows.append({
            "field":    field,
            "n_unique": n_uniq,
            "stable":   n_uniq <= 1,
            "modal":    str(modal),
        })
    # Special: ak_stream_count should grow monotonically
    if "ak_stream_count" in df.columns:
        sc = df["ak_stream_count"].dropna()
        rows.append({
            "field":    "ak_stream_count",
            "n_unique": sc.nunique(),
            "stable":   bool((sc.diff().dropna() >= 0).all()),   # monotonic?
            "modal":    f"range {sc.min()}–{sc.max()}",
        })
    return pd.DataFrame(rows)


# ─── CROSS-TRACE STABILITY ────────────────────────────────────────────────────

def cross_trace_stability(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each field, compute:
      - modal value across all traces
      - fraction of traces where every request has that modal value (stability)
      - Shannon entropy across traces (higher = more discriminative)
    """
    # One representative value per trace = modal value of that field
    rep = {}
    for name, df in traces.items():
        row = {}
        for field in CONSISTENCY_FIELDS + ["ak_stream_count"]:
            if field in df.columns:
                vals = df[field].dropna()
                row[field] = str(vals.mode().iloc[0]) if not vals.empty else None
        rep[name] = row

    rep_df = pd.DataFrame(rep).T   # rows=traces, cols=fields

    results = []
    for field in rep_df.columns:
        vals = rep_df[field].dropna()
        if vals.empty:
            continue
        counts  = Counter(vals)
        total   = len(vals)
        modal   = counts.most_common(1)[0][0]
        modal_f = counts[modal] / total
        # Shannon entropy
        probs   = np.array([c / total for c in counts.values()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        results.append({
            "field":          field,
            "n_distinct":     len(counts),
            "modal_value":    modal,
            "stability":      modal_f,        # 1.0 = identical across all traces
            "entropy":        entropy,         # 0 = no info, higher = discriminative
            "discriminative": entropy > 0.1,   # heuristic threshold
        })
    return pd.DataFrame(results).sort_values("entropy", ascending=False).reset_index(drop=True)


# ─── FINGERPRINT FEATURE VECTOR ──────────────────────────────────────────────

# Fields chosen for their discriminative value across AI agents
VECTOR_FIELDS = [
    # JA4 components
    "ja4_protocol", "ja4_version", "ja4_sni",
    "ja4_cipher_count", "ja4_ext_count", "ja4_alpn_code",
    "ja4_b", "ja4_c",
    # Raw TLS
    "tls_version", "negotiated_proto",
    "cipher_count", "cipher_no_grease_str",
    "curve_count",  "curve_no_grease_str",
    "sig_count",    "sig_no_grease_str",
    "cipher_has_grease", "curve_has_grease",
    "has_pq_crypto",
    # ALPN — full ordered list analysis
    "alpn_first", "alpn_count", "alpn_preference",
    "alpn_has_h2", "alpn_has_http11", "alpn_h2_position",
    "alpn_offered_str",
    # Supported versions — full ordered list analysis
    "supported_ver_profile", "supported_ver_count",
    "supported_ver_has_tls13", "supported_ver_has_tls12",
    "supported_ver_has_tls11", "supported_ver_has_grease",
    "supported_ver_13_before_12",
    "supported_ver_names", "supported_ver_str",
    # HTTP/2 Akamai
    "ak_window_update", "ak_pseudo_order",
    "ak_header_table", "ak_enable_push",
    "ak_init_window",  "ak_max_header_list",
    "ak_settings_str",
    # SNI
    "sni_present",
]

def build_feature_vector(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per trace — modal value of each VECTOR_FIELD."""
    rows = []
    for name, df in traces.items():
        row = {"trace": name}
        for field in VECTOR_FIELDS:
            if field in df.columns:
                vals = df[field].dropna()
                row[field] = vals.mode().iloc[0] if not vals.empty else None
            else:
                row[field] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("trace")


# ─── PLOTTING HELPERS ─────────────────────────────────────────────────────────

C = {
    "blue":   "#2563EB", "teal":   "#0D9488",
    "amber":  "#D97706", "red":    "#DC2626",
    "gray":   "#6B7280", "purple": "#7C3AED",
    "green":  "#16A34A", "pink":   "#DB2777",
}

def _ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── SINGLE-TRACE PLOT ───────────────────────────────────────────────────────

def plot_single_trace(df: pd.DataFrame, trace_name: str, out_path=None):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(f"TLS fingerprint detail — {trace_name}",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

    # 1. JA4 component stability across requests
    ax1 = fig.add_subplot(gs[0, :2])
    ja4_fields = ["ja4_a", "ja4_b", "ja4_c",
                  "ja4_protocol", "ja4_version", "ja4_sni",
                  "ja4_cipher_count", "ja4_ext_count", "ja4_alpn_code"]
    stability = []
    labels    = []
    for f in ja4_fields:
        if f in df.columns:
            n = df[f].dropna().nunique()
            stability.append(1 if n <= 1 else n)
            labels.append(f.replace("ja4_", ""))
    colors = [C["green"] if s == 1 else C["red"] for s in stability]
    ax1.bar(labels, stability, color=colors)
    ax1.axhline(1, color=C["gray"], lw=0.8, linestyle="--")
    ax1.set_ylabel("n distinct values\n(1=stable)", fontsize=7)
    stable_patch = mpatches.Patch(color=C["green"], label="stable (1 value)")
    drift_patch  = mpatches.Patch(color=C["red"],   label="drifts across requests")
    ax1.legend(handles=[stable_patch, drift_patch], fontsize=7, loc="upper right")
    _ax(ax1, "JA4 component stability within session", "JA4 field")

    # 2. Akamai stream count growth
    ax2 = fig.add_subplot(gs[0, 2])
    if "ak_stream_count" in df.columns:
        sc = df["ak_stream_count"].dropna().reset_index(drop=True)
        ax2.plot(sc.index, sc.values, color=C["teal"], lw=1.5, marker="o", ms=4)
        ax2.set_xlabel("request index", fontsize=7)
    _ax(ax2, "Akamai priority tree depth\n(grows per request)", ylabel="stream entries")

    # 3. Cipher suites offered (frequency bar — first request)
    ax3 = fig.add_subplot(gs[1, 0])
    first = df.iloc[0]
    cipher_str = first.get("cipher_seq_str", "")
    if cipher_str:
        ids    = [int(x) for x in cipher_str.split(",") if x.isdigit()]
        names  = [CIPHER_NAMES.get(c, str(c)) if not is_grease(c) else "GREASE" for c in ids]
        colors_c = [C["amber"] if n == "GREASE" else C["blue"] for n in names]
        y = range(len(names))
        ax3.barh(list(y), [1]*len(names), color=colors_c)
        ax3.set_yticks(list(y))
        ax3.set_yticklabels(names, fontsize=5)
        ax3.set_xticks([])
    _ax(ax3, "Cipher suites offered\n(in order, ← = first offered)")

    # 4. Elliptic curves offered
    ax4 = fig.add_subplot(gs[1, 1])
    curve_str = first.get("curve_seq_str", "")
    if curve_str:
        ids   = [int(x) for x in curve_str.split(",") if x.isdigit()]
        names = [CURVE_NAMES.get(c, str(c)) if not is_grease(c) else "GREASE" for c in ids]
        colors_r = [C["amber"] if n == "GREASE" else C["teal"] for n in names]
        y = range(len(names))
        ax4.barh(list(y), [1]*len(names), color=colors_r)
        ax4.set_yticks(list(y))
        ax4.set_yticklabels(names, fontsize=7)
        ax4.set_xticks([])
        # annotate PQ
        for i, n in enumerate(names):
            if "Kyber" in n or "MLKEM" in n:
                ax4.text(0.5, i, " ← PQ", va="center", fontsize=6, color=C["purple"])
    _ax(ax4, "Elliptic curves offered\n(in order)")

    # 5. Signature schemes offered
    ax5 = fig.add_subplot(gs[1, 2])
    sig_str = first.get("sig_seq_str", "")
    if sig_str:
        ids   = [int(x) for x in sig_str.split(",") if x.isdigit()]
        names = [SIG_NAMES.get(s, str(s)) for s in ids]
        y = range(len(names))
        ax5.barh(list(y), [1]*len(names), color=C["purple"])
        ax5.set_yticks(list(y))
        ax5.set_yticklabels(names, fontsize=6)
        ax5.set_xticks([])
    _ax(ax5, "Signature schemes offered\n(in order)")

    # 6. HTTP/2 SETTINGS values
    ax6 = fig.add_subplot(gs[2, 0])
    settings_fields = {
        "HEADER_TABLE_SIZE":  "ak_header_table",
        "ENABLE_PUSH":        "ak_enable_push",
        "INIT_WINDOW_SIZE":   "ak_init_window",
        "MAX_HEADER_LIST":    "ak_max_header_list",
        "WINDOW_UPDATE":      "ak_window_update",
    }
    s_labels, s_vals = [], []
    for label, col in settings_fields.items():
        val = df[col].dropna().iloc[0] if col in df.columns and not df[col].dropna().empty else None
        if val is not None:
            s_labels.append(label)
            s_vals.append(val)
    if s_vals:
        ax6.barh(s_labels, s_vals, color=C["green"])
        ax6.set_xscale("log")
    _ax(ax6, "HTTP/2 SETTINGS values\n(log scale)", xlabel="value")

    # 7. JA4 field summary table
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.axis("off")
    ja4_display = [
        ("JA4 (full)",      first.get("ja4_raw", "—")),
        ("JA4_A",           first.get("ja4_a",   "—")),
        ("JA4_B",           first.get("ja4_b",   "—")),
        ("JA4_C",           first.get("ja4_c",   "—")),
        ("protocol",        first.get("ja4_protocol", "—")),
        ("TLS version",     first.get("ja4_version",  "—")),
        ("SNI",             first.get("ja4_sni",      "—")),
        ("cipher count",    str(first.get("ja4_cipher_count", "—"))),
        ("ext count",       str(first.get("ja4_ext_count",    "—"))),
        ("ALPN code",       first.get("ja4_alpn_code", "—")),
        ("has GREASE",      str(first.get("cipher_has_grease", "—"))),
        ("PQ crypto",       str(first.get("has_pq_crypto", "—"))),
        ("pseudo-hdr order",first.get("ak_pseudo_order", "—")),
    ]
    tbl = ax7.table(
        cellText=[[k, str(v)] for k, v in ja4_display],
        colLabels=["Field", "Value"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.15)
    _ax(ax7, "JA4 & key fields (first request)")

    # 8. ALPN offered list (first request) — ordered bar
    ax8 = fig.add_subplot(gs[2, 2])
    alpn_str = str(first.get("alpn_offered_str", "") or "")
    alpn_items = [a.strip() for a in alpn_str.split(",") if a.strip()]
    if alpn_items:
        colors_a = [C["blue"] if a == "h2" else
                    C["teal"] if a == "http/1.1" else C["gray"]
                    for a in alpn_items]
        ax8.barh(range(len(alpn_items)), [1]*len(alpn_items), color=colors_a)
        ax8.set_yticks(range(len(alpn_items)))
        ax8.set_yticklabels(
            [f"{i}. {ALPN_LABEL.get(a, a)}" for i, a in enumerate(alpn_items)],
            fontsize=8
        )
        ax8.set_xticks([])
        # annotate preference
        pref = first.get("alpn_preference", "")
        ax8.set_title(f"ALPN offered (order matters)\npreference: {pref}",
                      fontsize=10, fontweight="bold", pad=6)
    else:
        _ax(ax8, "ALPN offered", "", "")

    # ── Extra row: supported_versions detail ──────────────────────────────
    # Reuse gs row 2 col 0 area — add a text summary below the table
    ax_ver = fig.add_subplot(gs[2, 0])
    ver_str   = str(first.get("supported_ver_str", "") or "")
    ver_names = str(first.get("supported_ver_names", "") or "")
    ver_items_raw = [v.strip() for v in ver_str.split(",") if v.strip()]
    ver_labels    = []
    ver_colors    = []
    for v in ver_items_raw:
        if v.lstrip("-").isdigit():
            vi = int(v)
            if is_grease(vi):
                ver_labels.append("GREASE")
                ver_colors.append(C["amber"])
            else:
                ver_labels.append(SUPPORTED_VERSION_NAMES.get(vi, f"0x{vi:04X}"))
                ver_colors.append(C["blue"] if vi == 772 else
                                  C["teal"] if vi == 771 else C["gray"])
        else:
            ver_labels.append(v)
            ver_colors.append(C["gray"])

    if ver_labels:
        ax_ver.barh(range(len(ver_labels)), [1]*len(ver_labels), color=ver_colors)
        ax_ver.set_yticks(range(len(ver_labels)))
        ax_ver.set_yticklabels(
            [f"{i}. {n}" for i, n in enumerate(ver_labels)], fontsize=8
        )
        ax_ver.set_xticks([])
        profile = first.get("supported_ver_profile", "")
        ax_ver.set_title(f"Supported TLS versions (order matters)\nprofile: {profile}",
                         fontsize=10, fontweight="bold", pad=6)
    else:
        _ax(ax_ver, "Supported versions", "", "")
    ax_ver.spines["top"].set_visible(False)
    ax_ver.spines["right"].set_visible(False)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── CROSS-TRACE PLOT ─────────────────────────────────────────────────────────

def plot_cross_trace(traces: dict[str, pd.DataFrame],
                     stability_df: pd.DataFrame,
                     fv: pd.DataFrame,
                     out_path=None):
    fig = plt.figure(figsize=(20, 14))
    # fig.suptitle(f"TLS fingerprint cross-trace analysis — {len(traces)} traces",
    #              fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.8, wspace=0.42)

    # 1. Field stability heatmap (top discriminative fields)
    ax1 = fig.add_subplot(gs[0, :2])
    top_fields = stability_df.head(20)["field"].tolist()
    stab_vals  = stability_df.set_index("field").loc[top_fields, "stability"].values
    colors_s   = [C["green"] if s >= 0.95 else C["amber"] if s >= 0.5 else C["red"]
                  for s in stab_vals]
    ax1.barh(top_fields[::-1], stab_vals[::-1], color=colors_s[::-1])
    ax1.axvline(1.0, color=C["gray"], lw=0.8, linestyle="--")
    ax1.set_xlim(0, 1.1)
    _ax(ax1, "Field stability across traces\n(1.0 = identical in all traces, sorted by entropy)",
        "stability score")
    for patch, val in zip(ax1.patches, stab_vals[::-1]):
        ax1.text(patch.get_width() + 0.01, patch.get_y() + patch.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=6)

    # 2. Entropy ranking (discriminative power)
    ax2 = fig.add_subplot(gs[0, 2])
    top_ent = stability_df[stability_df["entropy"] > 0].head(12)
    ax2.barh(top_ent["field"][::-1], top_ent["entropy"][::-1], color=C["purple"])
    _ax(ax2, "Discriminative power\n(Shannon entropy, higher=more useful)",
        "entropy (bits)")

    # 3. JA4_B distribution across traces
    ax3 = fig.add_subplot(gs[1, 0])
    if "ja4_b" in fv.columns:
        cnt = fv["ja4_b"].value_counts()
        ax3.bar(range(len(cnt)), cnt.values, color=C["blue"])
        ax3.set_xticks(range(len(cnt)))
        ax3.set_xticklabels(cnt.index, rotation=45, ha="right", fontsize=6)
    _ax(ax3, "JA4_B (cipher hash) distribution\nacross traces", ylabel="trace count")

    # 4. JA4_C distribution across traces
    ax4 = fig.add_subplot(gs[1, 1])
    if "ja4_c" in fv.columns:
        cnt = fv["ja4_c"].value_counts()
        ax4.bar(range(len(cnt)), cnt.values, color=C["teal"])
        ax4.set_xticks(range(len(cnt)))
        ax4.set_xticklabels(cnt.index, rotation=45, ha="right", fontsize=6)
    _ax(ax4, "JA4_C (ext/sig hash) distribution\nacross traces", ylabel="trace count")

    # 5. ALPN preference distribution across traces
    ax5 = fig.add_subplot(gs[1, 2])
    if "alpn_preference" in fv.columns:
        cnt = fv["alpn_preference"].value_counts()
        bars = ax5.bar(cnt.index, cnt.values, color=C["amber"])
        for bar, val in zip(bars, cnt.values):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                     str(val), ha="center", fontsize=8)
    _ax(ax5, "ALPN preference distribution\nacross traces", "preference", "trace count")

    # 6. Supported versions profile distribution
    ax6 = fig.add_subplot(gs[2, 0])
    if "supported_ver_profile" in fv.columns:
        cnt = fv["supported_ver_profile"].value_counts()
        bars = ax6.bar(cnt.index, cnt.values, color=C["green"])
        for bar, val in zip(bars, cnt.values):
            ax6.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                     str(val), ha="center", fontsize=8)
        ax6.tick_params(axis="x", labelsize=7)
    _ax(ax6, "Supported versions profile\nacross traces", "version profile", "trace count")

    # 7. Akamai SETTINGS comparison heatmap
    ax7 = fig.add_subplot(gs[2, 1:])
    ak_fields = ["ak_header_table", "ak_enable_push",
                 "ak_init_window", "ak_max_header_list", "ak_window_update",
                 "alpn_count", "alpn_h2_position",
                 "supported_ver_count", "supported_ver_has_tls13",
                 "supported_ver_has_tls12", "supported_ver_has_grease"]
    ak_present = [f for f in ak_fields if f in fv.columns]
    if ak_present:
        ak_data = fv[ak_present].apply(pd.to_numeric, errors="coerce").astype(float)
        # normalise each column for colour scale
        norm = (ak_data - ak_data.min()) / (ak_data.max() - ak_data.min() + 1e-9)
        im = ax7.imshow(norm.values.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax7.set_xticks(range(len(fv)))
        ax7.set_xticklabels(fv.index, rotation=90, fontsize=5)
        ax7.set_yticks(range(len(ak_present)))
        ax7.set_yticklabels([f.replace("ak_","") for f in ak_present], fontsize=7)
        plt.colorbar(im, ax=ax7, fraction=0.03, label="normalised value")
    _ax(ax7, "HTTP/2 SETTINGS per trace\n(normalised)", "trace")

    # 7b. ALPN preference distribution across traces
    ax_alpn = fig.add_subplot(gs[3, 0])
    if "alpn_preference" in fv.columns:
        cnt = fv["alpn_preference"].value_counts()
        ax_alpn.bar(cnt.index, cnt.values, color=C["blue"])
        ax_alpn.set_xticklabels(cnt.index, rotation=20, ha="right", fontsize=7)
    _ax(ax_alpn, "ALPN preference distribution\nacross traces", "preference", "traces")

    # 7c. Supported version profile distribution across traces
    ax_ver = fig.add_subplot(gs[3, 1])
    if "supported_ver_profile" in fv.columns:
        cnt = fv["supported_ver_profile"].value_counts()
        ax_ver.bar(cnt.index, cnt.values, color=C["teal"])
        ax_ver.set_xticklabels(cnt.index, rotation=20, ha="right", fontsize=7)
    _ax(ax_ver, "Supported version profile\nacross traces", "profile", "traces")

    # 8. PQ crypto and GREASE presence across traces
    ax8 = fig.add_subplot(gs[3, 2])
    flags = pd.concat(
        [df[["cipher_has_grease","curve_has_grease","has_pq_crypto",
             "supported_ver_has_grease"]]
         .mode().iloc[[0]].assign(trace=name)
         for name, df in traces.items()]
    ).set_index("trace")
    x = np.arange(len(flags))
    w = 0.22
    for i, (col, color, label) in enumerate([
        ("cipher_has_grease",        C["amber"],  "cipher GREASE"),
        ("curve_has_grease",         C["teal"],   "curve GREASE"),
        ("supported_ver_has_grease", C["red"],    "version GREASE"),
        ("has_pq_crypto",            C["purple"], "PQ crypto"),
    ]):
        if col in flags.columns:
            ax8.bar(x + i*w, flags[col].astype(float), w,
                    color=color, label=label, alpha=0.85)
    ax8.set_xticks(x + 1.5*w)
    ax8.set_xticklabels(flags.index, rotation=90, fontsize=5)
    ax8.set_yticks([0, 1]); ax8.set_yticklabels(["False","True"], fontsize=7)
    ax8.legend(fontsize=6)
    _ax(ax8, "GREASE & PQ crypto presence\nper trace")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── CONSOLE REPORTING ───────────────────────────────────────────────────────

def _sec(title):
    print(f"\n{'─'*70}\n  {title}\n{'─'*70}")


def print_ja4_summary(fv: pd.DataFrame):
    _sec("JA4 decomposition — modal values per trace")
    cols = ["ja4_a","ja4_b","ja4_c","ja4_protocol","ja4_version",
            "ja4_sni","ja4_cipher_count","ja4_ext_count","ja4_alpn_code"]
    cols = [c for c in cols if c in fv.columns]
    print(fv[cols].to_string())


def print_tls_summary(fv: pd.DataFrame):
    _sec("TLS layer — modal values per trace")
    cols = ["tls_version","negotiated_proto","alpn_first","cipher_count",
            "curve_count","sig_count","cipher_has_grease","has_pq_crypto","sni_present"]
    cols = [c for c in cols if c in fv.columns]
    print(fv[cols].to_string())


def print_h2_summary(fv: pd.DataFrame):
    _sec("HTTP/2 Akamai — modal values per trace")
    cols = ["ak_settings_str","ak_window_update","ak_pseudo_order",
            "ak_header_table","ak_enable_push","ak_init_window","ak_max_header_list"]
    cols = [c for c in cols if c in fv.columns]
    print(fv[cols].to_string())


def print_alpn_versions(fv: pd.DataFrame):
    _sec("ALPN protocols — full list analysis per trace")
    cols = ["alpn_offered_str", "alpn_count", "alpn_preference",
            "alpn_has_h2", "alpn_has_http11", "alpn_h2_position"]
    cols = [c for c in cols if c in fv.columns]
    print(fv[cols].to_string())

    _sec("Supported versions — full list analysis per trace")
    cols = ["supported_ver_names", "supported_ver_profile", "supported_ver_count",
            "supported_ver_has_tls13", "supported_ver_has_tls12",
            "supported_ver_has_tls11", "supported_ver_has_grease",
            "supported_ver_13_before_12"]
    cols = [c for c in cols if c in fv.columns]
    print(fv[cols].to_string())


def print_stability(stability_df: pd.DataFrame):
    _sec("Cross-trace field stability & discriminative power (top 25)")
    print(stability_df.head(25).to_string(index=False))


def print_discriminative(stability_df: pd.DataFrame):
    _sec("Fields useful for distinguishing AI agents (entropy > 0, stability < 1)")
    disc = stability_df[
        (stability_df["entropy"] > 0) & (stability_df["stability"] < 1.0)
    ]
    if disc.empty:
        print("  All fields are identical across traces — fingerprint is fully stable.")
    else:
        print(disc[["field","n_distinct","stability","entropy"]].to_string(index=False))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TLS fingerprint analyzer for AI agent identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python tls_analysis.py --traces ./traces
  python tls_analysis.py --traces ./traces --out ./results
  python tls_analysis.py --files trial-001/requests.jsonl trial-002/requests.jsonl
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

    # ── Discover files ───────────────────────────────────────────────────────
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
    do_filter = not args.no_filter
    print(f"\nLoading {len(filepaths)} trace file(s)…")
    print(f"  sendBeacon filter: {'OFF (--no-filter)' if not do_filter else 'ON'}")
    traces = load_all(filepaths, filter_senbeacon=do_filter)
    if not traces:
        print("No valid traces loaded. Exiting.")
        sys.exit(1)

    # ── Core outputs ─────────────────────────────────────────────────────────
    # 1. Full parsed request table
    all_requests = pd.concat(traces.values(), ignore_index=True)
    all_requests.to_csv(out_dir / "tls_summary.csv", index=False)
    print(f"\n  Saved: {out_dir / 'tls_summary.csv'}")

    # 2. Cross-trace stability & entropy
    stability_df = cross_trace_stability(traces)
    stability_df.to_csv(out_dir / "tls_stability.csv", index=False)
    print(f"  Saved: {out_dir / 'tls_stability.csv'}")

    # 3. Feature vector (one row per trace)
    fv = build_feature_vector(traces)
    fv.to_csv(out_dir / "tls_fingerprint_vector.csv")
    print(f"  Saved: {out_dir / 'tls_fingerprint_vector.csv'}")

    # ── Console reports ───────────────────────────────────────────────────────
    print_ja4_summary(fv)
    print_tls_summary(fv)
    print_h2_summary(fv)
    print_alpn_versions(fv)
    print_stability(stability_df)
    print_discriminative(stability_df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(traces) == 1:
        name, df = next(iter(traces.items()))
        plot_single_trace(df, name,
                          out_path=str(out_dir / f"tls_single_{name}.png"))
    else:
        name0, df0 = next(iter(traces.items()))
        plot_single_trace(df0, name0,
                          out_path=str(out_dir / f"tls_single_{name0}.png"))
        plot_cross_trace(traces, stability_df, fv,
                         out_path=str(out_dir / "tls_cross_trace.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()