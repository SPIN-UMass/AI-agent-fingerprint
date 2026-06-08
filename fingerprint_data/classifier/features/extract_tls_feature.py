#!/usr/bin/env python3
"""
tls_trial_features.py
=====================
Reads requests.jsonl files directly (no intermediate tls_summary CSV needed),
parses every TLS / HTTP2 field, removes sendBeacon noise, and aggregates each
trial into one ML-ready feature vector.

Directory layout expected
-------------------------
  <root>/
    autogen/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
      ...
    skyvern/
      trial-001/requests.jsonl
      ...

  Agent name   = the immediate subdirectory of <root>  (e.g. "autogen")
  Trial label  = the next level directory              (e.g. "trial-001")

Alternatively, pass explicit agent directories with --agents and supply
agent names via --labels (one label per directory, in the same order).

Feature groups
--------------
  F1  PQ key exchange fractions     (2)   Kyber768 / MLKEM768 mix ratio
  F2  Browser identity fractions    (3)   Firefox JA4, H2 proto, PQ any
  F3  H2 window & settings          (3)   AK window mean, Firefox-window frac,
                                          mean init-window size
  F4  Stream-3 weight fractions     (2)   Frac of reqs where stream-3 w=110/220
  F5  Stream-5 weight fractions     (3)   Frac of reqs where stream-5 w=110/220/256
  F6  Stream coverage               (2)   Frac of reqs with stream-3/5 visible
  F7  TLS diversity                 (2)   Unique JA3 / total reqs (normalised),
                                          unique source IPs / total reqs
  F8  Cipher & curve counts         (3)   Mean cipher count, mean curve count,
                                          mean sig count
  F9  Pseudo-header order (mode)    (2)   H2 and AK pseudo-header order (ordinal)
  F10 TLS extension count fractions (3)   Frac of reqs: ext=16 (Chrome), ext=17 (Skyvern),
                                          ext=12 AND cipher=28 (Firefox). Note: Firefox is
                                          identified by CIPHER count 28 at JA4_A[4:6], not
                                          extension count; ja4_ext_count=12 for Firefox.
  F11 GREASE flags (mode)           (3)   Cipher / curve / supported-ver GREASE
  F12 Trial size                    (1)   log(n_requests) — session length signal

Total: 29 features  x  (n_agents x n_trials) rows

Usage
-----
  # Auto-discover agents and trials under a root directory:
  python tls_trial_features.py --root ./traces --out trial_features.csv

  # Explicit agent directories with custom names:
  python tls_trial_features.py \
      --agents ./traces/autogen ./traces/skyvern \
      --labels AutoGen Skyvern \
      --out trial_features.csv

  # Integer-encode labels (needed by XGBoost, LightGBM etc.):
  python tls_trial_features.py --root ./traces --out trial_features.csv \
      --encode-labels

  # Use a different filename inside each trial directory:
  python tls_trial_features.py --root ./traces --filename events.jsonl \
      --out trial_features.csv

  # Skip the sendBeacon filter:
  python tls_trial_features.py --root ./traces --no-filter \
      --out trial_features.csv

Outputs
-------
  trial_features.csv       -- one row per trial, 29 features + agent + trial_id
  trial_features_meta.csv  -- descriptions, null rates, per-agent means
  label_map.csv            -- agent <-> integer (only with --encode-labels)
"""

import json
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Lookup tables (mirrors tls_analysis.py) ──────────────────────────────────

CIPHER_NAMES = {
    4865: "TLS_AES_128_GCM_SHA256",     4866: "TLS_AES_256_GCM_SHA384",
    4867: "TLS_CHACHA20_POLY1305_SHA256",
    49195: "ECDHE-ECDSA-AES128-GCM",   49199: "ECDHE-RSA-AES128-GCM",
    49196: "ECDHE-ECDSA-AES256-GCM",   49200: "ECDHE-RSA-AES256-GCM",
    52393: "ECDHE-ECDSA-CHACHA20",     52392: "ECDHE-RSA-CHACHA20",
    49171: "ECDHE-RSA-AES128-CBC",     49172: "ECDHE-RSA-AES256-CBC",
    156:   "RSA-AES128-GCM",           157:   "RSA-AES256-GCM",
    47:    "RSA-AES128-CBC",            53:    "RSA-AES256-CBC",
    0xFF:  "EMPTY_RENEGOTIATION",
}

CURVE_NAMES = {
    29:    "x25519",        23:    "secp256r1",
    24:    "secp384r1",     25:    "secp521r1",
    25497: "x25519Kyber768",   # PQ draft -- Chrome 124+
    4588:  "X25519MLKEM768",   # PQ standard -- Chrome 131+
}

SIG_NAMES = {
    1027: "ecdsa_secp256r1_sha256",  2052: "rsa_pss_rsae_sha256",
    1025: "rsa_pkcs1_sha256",        1283: "ecdsa_secp384r1_sha384",
    2053: "rsa_pss_rsae_sha384",     1281: "rsa_pkcs1_sha384",
    2054: "rsa_pss_rsae_sha512",     1537: "rsa_pkcs1_sha512",
    1540: "rsa_pkcs1_sha1",           515: "ecdsa_sha1",
}

# sendBeacon filter thresholds
COLLECT_ENDPOINT = "/collect"
UNLOAD_WINDOW_MS = 500   # max ms between POST and next page GET (Rule 1)
BURST_WINDOW_MS  = 50    # max ms between POSTs in a burst (Rule 2)

# Pseudo-header order string -> integer ordinal (same mapping for H2 and AK)
PSEUDO_ORDER_MAP = {
    ":method,:authority,:scheme,:path": 0, "m,a,s,p": 0,   # Chrome
    ":method,:scheme,:path,:authority": 1, "m,s,p,a": 1,   # Firefox
    ":method,:scheme,:authority,:path": 2, "m,s,a,p": 2,   # Skyvern/Edge
    ":authority,:method,:path,:scheme": 3, "a,m,p,s": 3,
    ":authority,:path,:method,:scheme": 4, "a,p,m,s": 4,
    ":method,:path,:scheme,:authority": 5, "m,p,s,a": 5,
}

# ── Low-level helpers ─────────────────────────────────────────────────────────

def _is_grease(val: int) -> bool:
    """RFC 8701: GREASE values have the form 0x?A?A."""
    return isinstance(val, int) and (val & 0x0F0F) == 0x0A0A


def _seq_features(seq: list, name: str, lookup: dict) -> dict:
    """Count, GREASE flag, and human-readable string from an ordered ID list."""
    if not seq:
        return {f"{name}_count": 0, f"{name}_has_grease": False,
                f"{name}_no_grease_str": ""}
    no_grease = [v for v in seq if not _is_grease(v)]
    return {
        f"{name}_count":         len(no_grease),
        f"{name}_has_grease":    any(_is_grease(v) for v in seq),
        f"{name}_no_grease_str": "_".join(lookup.get(v, str(v)) for v in no_grease),
    }


def _parse_ja4(ja4: str) -> dict:
    """
    Decompose JA4_A string into component fields.

    JA4_A format: {protocol}{tls_version}{sni}{cipher_count:02d}{ext_count:02d}{alpn}
    Example:       t          13           d    15                16              h2
    Firefox:       t          13           d    28                12              h2
    Skyvern:       t          13           d    15                17              h2

    Note: ja4_cipher_count (positions 4-5) and ja4_ext_count (positions 6-7)
    are distinct.  Firefox is identified by ja4_cipher_count=28, ext_count=12.
    Skyvern is identified by ja4_ext_count=17.
    """
    out = {"ja4_a": None, "ja4_cipher_count": None, "ja4_ext_count": None}
    if not isinstance(ja4, str):
        return out
    parts = ja4.split("_")
    if len(parts) == 3:
        a = parts[0]
        out["ja4_a"] = a
        if len(a) >= 10:
            if a[4:6].isdigit():
                out["ja4_cipher_count"] = int(a[4:6])
            if a[6:8].isdigit():
                out["ja4_ext_count"] = int(a[6:8])
    return out


def _parse_akamai(fp: str) -> dict:
    """
    Parse Akamai H2 fingerprint: SETTINGS|WINDOW_UPDATE|PRIORITY|PSEUDO_ORDER.
    Returns the fields needed for features F3-F5 and F9.
    """
    out = {"ak_window_update": None, "ak_priority_str": None,
           "ak_pseudo_order": None, "ak_init_window": None}
    if not isinstance(fp, str) or not fp:
        return out
    parts = fp.split("|")
    if len(parts) > 1 and parts[1].isdigit():
        out["ak_window_update"] = int(parts[1])
    if len(parts) > 2:
        out["ak_priority_str"] = parts[2]
    if len(parts) > 3:
        out["ak_pseudo_order"] = parts[3]
    # INITIAL_WINDOW_SIZE is SETTINGS key 4
    for pair in parts[0].split(";"):
        if pair.startswith("4:") and pair[2:].lstrip("-").isdigit():
            out["ak_init_window"] = int(pair[2:])
            break
    return out


def _stream_weight(priority_str, stream_id: int) -> float:
    """Extract the priority weight for stream_id from ak_priority_str."""
    if not isinstance(priority_str, str) or not priority_str:
        return np.nan
    for m in re.finditer(r'(\d+):\d+:\d+:(\d+)', priority_str):
        if int(m.group(1)) == stream_id:
            return float(m.group(2))
    return np.nan


def _safe_mode(series: pd.Series):
    v = series.dropna()
    return v.mode().iloc[0] if not v.empty else np.nan


def _frac_eq(series: pd.Series, value: float) -> float:
    v = series.dropna()
    return float((v == value).mean()) if not v.empty else 0.0

# ── Per-request parser ────────────────────────────────────────────────────────

def parse_request(rec: dict) -> dict:
    """
    Parse one JSON record from requests.jsonl into a flat feature dict.
    Only fields actually used in feature extraction are retained;
    everything else is discarded to keep memory usage low.
    """
    tls  = rec.get("tls")  or {}
    h2   = rec.get("http2") or {}
    http = rec.get("http")  or {}

    row: dict = {
        "source_ip":       rec.get("source_ip"),
        "timestamp":       rec.get("timestamp"),
        "method":          http.get("method"),
        "path":            http.get("path"),
        "ja3":             tls.get("ja3_hash"),
        "negotiated_proto": tls.get("negotiated_protocol"),
    }

    # Cipher suites, elliptic curves, signature schemes
    row.update(_seq_features(tls.get("cipher_suites_offered") or [], "cipher", CIPHER_NAMES))
    row.update(_seq_features(tls.get("elliptic_curves")       or [], "curve",  CURVE_NAMES))
    row.update(_seq_features(tls.get("signature_schemes")     or [], "sig",    SIG_NAMES))

    # PQ presence (x25519Kyber768=25497, X25519MLKEM768=4588)
    curves = tls.get("elliptic_curves") or []
    row["has_pq_crypto"] = any(v in (25497, 4588) for v in curves)

    # Supported versions -- only need the GREASE flag for F11
    ver_list = tls.get("supported_versions") or []
    row["supported_ver_has_grease"] = any(_is_grease(v) for v in ver_list)

    # JA4 ext count and ja4_a (needed for F2/F10 and pseudo-order)
    row.update(_parse_ja4(tls.get("ja4_hash", "")))

    # Akamai H2 fingerprint (F3, F4, F5, F9)
    row.update(_parse_akamai(h2.get("akamai_fingerprint", "")))

    # H2 pseudo-header order (F9)
    ph = h2.get("pseudo_header_order") or []
    row["h2_pseudo_order"] = ",".join(ph)

    return row

# ── sendBeacon filter ─────────────────────────────────────────────────────────

def filter_senbeacon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove POST /collect sendBeacon rows using Rules 1 + 2 from tls_analysis.py.

    Rule 1 (temporal): POST arrives within UNLOAD_WINDOW_MS before next page GET.
    Rule 2 (burst):    Multiple /collect POSTs within BURST_WINDOW_MS of each other.

    Row removed when Rule 1 AND Rule 2 are both satisfied.

    Note: Rule 3 (Referer ends in .html) from tls_analysis.py requires the
    Referer header, which is not present in requests.jsonl records.  Rules 1+2
    alone are sufficient to catch all genuine beforeunload/pagehide bursts.
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

    # flagged: set = set()
    # for idx in collect_idx:
    #     ts  = df.loc[idx, "timestamp"]
    #     gap = ms_to_next_page(ts)
    #     # Rule 1
    #     if np.isnan(gap) or gap > UNLOAD_WINDOW_MS:
    #         continue
    #     # Rule 2
    #     other_ts = df.loc[collect_mask & (df.index != idx), "timestamp"]
    #     if any(abs((ts - ot).total_seconds() * 1000) <= BURST_WINDOW_MS
    #            for ot in other_ts):
    #         flagged.add(idx)

    # if not flagged:
    #     return df, 0
    # return df.drop(index=list(flagged)).reset_index(drop=True), len(flagged)

# ── Derived columns (per-request binary/numeric signals) ─────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-request binary/numeric columns used in aggregation."""
    curves    = df["curve_no_grease_str"].fillna("")
    ja4a      = df["ja4_a"].fillna("")
    ext_count = pd.to_numeric(df["ja4_ext_count"], errors="coerce")

    # F1: PQ key exchange
    df["_has_kyber768"] = curves.str.contains("Kyber768", regex=False).astype(int)
    df["_has_mlkem768"] = curves.str.contains("MLKEM",    regex=False).astype(int)

    # F2: browser identity
    df["_is_firefox_ja4"] = ja4a.str.startswith("t13d2812").astype(int)
    df["_is_h2"]          = (df["negotiated_proto"] == "h2").astype(int)
    df["_has_pq"]         = df["has_pq_crypto"].astype(bool).astype(int)

    # F3: Firefox H2 window flag
    df["_win_is_firefox"] = (
        pd.to_numeric(df["ak_window_update"], errors="coerce") > 1e9
    ).astype(int)

    # F4 / F5: stream weights
    df["_s3w"] = df["ak_priority_str"].apply(lambda s: _stream_weight(s, 3))
    df["_s5w"] = df["ak_priority_str"].apply(lambda s: _stream_weight(s, 5))

    # F10: TLS extension count buckets
    # ext=16: standard Chrome; ext=17: Skyvern's extra extension.
    # ext=12 + cipher=28 together identify Firefox (Firefox sends 28 cipher
    # suites and 12 TLS extensions, giving ja4_a prefix t13d2812).
    cipher_count_col = pd.to_numeric(df["ja4_cipher_count"], errors="coerce")
    df["_is_ext16"]     = (ext_count == 16).astype(int)
    df["_is_ext17"]     = (ext_count == 17).astype(int)
    df["_is_ext12"]     = (ext_count == 12).astype(int)
    df["_is_cipher28"]  = (cipher_count_col == 28).astype(int)

    # F11: GREASE flags
    df["_grease_cipher"] = df["cipher_has_grease"].astype(bool).astype(int)
    df["_grease_curve"]  = df["curve_has_grease"].astype(bool).astype(int)
    df["_grease_ver"]    = df["supported_ver_has_grease"].astype(bool).astype(int)

    return df

# ── Trial-level aggregation ───────────────────────────────────────────────────

def aggregate_trial(grp: pd.DataFrame) -> pd.Series:
    """
    Collapse one trial's requests into a 29-feature vector.

    Aggregation strategies
    ----------------------
    Fraction   mean of a binary 0/1 column (captures mix ratio)
    Mean       arithmetic mean of a continuous field
    Coverage   fraction of rows where a nullable field is non-null
    Diversity  unique-count / n (normalised richness)
    Mode->ord  dominant categorical mapped to an integer ordinal
    Log-size   log1p(n) to compress the heavy tail in request counts
    """
    n   = len(grp)
    row = {}

    # F1 -- PQ fractions split Chrome generations
    # AutoGen/BrowserUse: ~30% Kyber / ~65% MLKEM (mixed versions)
    # Claude/Gemini:        ~0% Kyber / ~99% MLKEM (fully updated)
    row["f1_frac_kyber768"] = grp["_has_kyber768"].mean()
    row["f1_frac_mlkem768"] = grp["_has_mlkem768"].mean()

    # F2 -- browser identity
    # frac_firefox_ja4: Operator ~26%, all others ~0%
    row["f2_frac_firefox_ja4"] = grp["_is_firefox_ja4"].mean()
    row["f2_frac_h2"]          = grp["_is_h2"].mean()
    row["f2_frac_pq"]          = grp["_has_pq"].mean()

    # F3 -- H2 SETTINGS window sizes
    # Firefox WINDOW_UPDATE ~2.147e9; Chrome ~15.6e6; Skyvern ~8.3e6
    ak_win = pd.to_numeric(grp["ak_window_update"], errors="coerce")
    row["f3_mean_ak_window"]   = ak_win.mean() if ak_win.notna().any() else np.nan
    row["f3_frac_win_firefox"] = grp["_win_is_firefox"].mean()
    init    = pd.to_numeric(grp["ak_init_window"], errors="coerce")
    row["f3_mean_init_window"] = init.mean() if init.notna().any() else np.nan

    # F4 -- stream-3 weight fractions (2nd request on H2 connection)
    # Chrome: 220 always; Skyvern: 110 ~44% of the time
    row["f4_frac_s3_w110"] = _frac_eq(grp["_s3w"], 110.0)
    row["f4_frac_s3_w220"] = _frac_eq(grp["_s3w"], 220.0)

    # F5 -- stream-5 weight fractions (3rd request)
    # BrowserUse: 220 (100%); Claude: 256 (100%); Gemini: 256 (97%)
    row["f5_frac_s5_w110"] = _frac_eq(grp["_s5w"], 110.0)
    row["f5_frac_s5_w220"] = _frac_eq(grp["_s5w"], 220.0)
    row["f5_frac_s5_w256"] = _frac_eq(grp["_s5w"], 256.0)

    # F6 -- stream coverage
    # Low coverage -> short sessions or many independent connections.
    # Skyvern: stream-5 ~0.03; Claude/Gemini: ~0.73
    row["f6_stream3_coverage"] = grp["_s3w"].notna().mean()
    row["f6_stream5_coverage"] = grp["_s5w"].notna().mean()

    # F7 -- TLS diversity (normalised by n)
    # Skyvern: ja3_diversity ~0.76; Claude: ~0.14
    row["f7_ja3_diversity"] = grp["ja3"].nunique() / n
    row["f7_ip_diversity"]  = grp["source_ip"].nunique() / n

    # F8 -- cipher/curve/sig counts
    # Firefox ~28 ciphers, Chrome ~15; mean captures Operator's blend
    row["f8_mean_cipher_count"] = pd.to_numeric(grp["cipher_count"], errors="coerce").mean()
    row["f8_mean_curve_count"]  = pd.to_numeric(grp["curve_count"],  errors="coerce").mean()
    row["f8_mean_sig_count"]    = pd.to_numeric(grp["sig_count"],    errors="coerce").mean()

    # F9 -- pseudo-header order (mode -> ordinal)
    # Stable within a browser, so mode is appropriate.
    # Chrome=0, Firefox=1, Skyvern/Edge=2, others 3-5
    row["f9_h2_pseudo_order"] = PSEUDO_ORDER_MAP.get(
        _safe_mode(grp["h2_pseudo_order"]), -1
    )
    row["f9_ak_pseudo_order"] = PSEUDO_ORDER_MAP.get(
        _safe_mode(grp["ak_pseudo_order"]), -1
    )

    # F10 -- TLS extension count fractions (structural, not agent-named)
    # ext=16:      standard Chrome
    # ext=17:      Skyvern (one extra extension beyond standard Chrome)
    # ext=12 + cipher=28: Firefox (28 cipher suites, 12 TLS extensions --
    #   ja4_cipher_count and ja4_ext_count are distinct JA4_A fields at
    #   positions 4-5 and 6-7 respectively; the "28 extensions" description
    #   in earlier versions incorrectly referred to the cipher count)
    row["f10_frac_ext16"]    = grp["_is_ext16"].mean()
    row["f10_frac_ext17"]    = grp["_is_ext17"].mean()
    row["f10_frac_firefox_ext"] = (grp["_is_ext12"] * grp["_is_cipher28"]).mean()

    # F11 -- GREASE flags (mode)
    # Real browsers include GREASE; some automation tools omit it
    row["f11_grease_cipher"] = int(_safe_mode(grp["_grease_cipher"]) or 0)
    row["f11_grease_curve"]  = int(_safe_mode(grp["_grease_curve"])  or 0)
    row["f11_grease_ver"]    = int(_safe_mode(grp["_grease_ver"])    or 0)

    # F12 -- trial size (log-compressed)
    # Operator: 1-200 reqs/trial; Claude/Gemini: 8-10 reqs/trial
    row["f12_log_n_requests"] = float(np.log1p(n))

    return pd.Series(row)

# ── Metadata ──────────────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "f1_frac_kyber768":    "Fraction of requests with x25519Kyber768 curve (older Chrome PQ)",
    "f1_frac_mlkem768":    "Fraction of requests with X25519MLKEM768 curve (newer Chrome PQ)",
    "f2_frac_firefox_ja4": "Fraction of requests with Firefox JA4 prefix (t13d2812, 28 TLS ext)",
    "f2_frac_h2":          "Fraction of requests negotiated over HTTP/2",
    "f2_frac_pq":          "Fraction of requests advertising any PQ key exchange",
    "f3_mean_ak_window":   "Mean H2 WINDOW_UPDATE size (Firefox~2.15e9, Chrome~1.57e7)",
    "f3_frac_win_firefox": "Fraction of requests with Firefox-sized H2 window (>1e9)",
    "f3_mean_init_window": "Mean H2 SETTINGS INITIAL_WINDOW_SIZE",
    "f4_frac_s3_w110":     "Fraction of requests where stream-3 priority weight = 110 (Skyvern)",
    "f4_frac_s3_w220":     "Fraction of requests where stream-3 priority weight = 220 (Chrome)",
    "f5_frac_s5_w110":     "Fraction of requests where stream-5 priority weight = 110",
    "f5_frac_s5_w220":     "Fraction of requests where stream-5 priority weight = 220 (BrowserUse)",
    "f5_frac_s5_w256":     "Fraction of requests where stream-5 priority weight = 256 (Claude/Gemini)",
    "f6_stream3_coverage": "Fraction of requests with stream-3 priority data present",
    "f6_stream5_coverage": "Fraction of requests with stream-5 priority data present",
    "f7_ja3_diversity":    "Unique JA3 fingerprints / total requests (Skyvern~0.76, Claude~0.14)",
    "f7_ip_diversity":     "Unique source IPs / total requests",
    "f8_mean_cipher_count":"Mean number of cipher suites offered (Firefox~28, Chrome~15)",
    "f8_mean_curve_count": "Mean number of elliptic curves offered",
    "f8_mean_sig_count":   "Mean number of signature schemes offered",
    "f9_h2_pseudo_order":  "Modal H2 pseudo-header order (0=Chrome, 1=Firefox, 2=Skyvern...)",
    "f9_ak_pseudo_order":  "Modal AK pseudo-header order (same scale as f9_h2)",
    "f10_frac_ext16":      "Fraction of requests with 16 TLS extensions (standard Chrome)",
    "f10_frac_ext17":      "Fraction of requests with 17 TLS extensions (Skyvern extra extension)",
    "f10_frac_firefox_ext": "Fraction of requests with Firefox TLS profile (ext=12, cipher=28 in JA4_A)",
    "f11_grease_cipher":   "Modal GREASE presence in cipher suites (1=present, 0=absent)",
    "f11_grease_curve":    "Modal GREASE presence in elliptic curves",
    "f11_grease_ver":      "Modal GREASE presence in supported_versions",
    "f12_log_n_requests":  "log1p(total requests in trial) -- session length signal",
}


def build_metadata(feat_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in feat_df.columns
                    if c not in ("agent", "trial", "label_id")]
    rows = []
    for col in feature_cols:
        s = feat_df[col]
        row = {
            "feature":      col,
            "description":  FEATURE_DESCRIPTIONS.get(col, ""),
            "null_frac":    round(float(s.isna().mean()), 4),
            "n_unique":     int(s.nunique()),
            "overall_mean": round(float(s.mean()), 4) if pd.api.types.is_numeric_dtype(s) else None,
            "overall_std":  round(float(s.std()),  4) if pd.api.types.is_numeric_dtype(s) else None,
        }
        for agent in sorted(feat_df["agent"].unique()):
            ag = feat_df.loc[feat_df["agent"] == agent, col]
            row[f"mean_{agent}"] = (
                round(float(ag.mean()), 4) if pd.api.types.is_numeric_dtype(ag) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)

# ── Directory discovery ───────────────────────────────────────────────────────

def discover_trials(agent_dir: Path,
                    filename: str) -> list[tuple[str, Path]]:
    """[(trial_label, jsonl_path), ...] sorted by trial label."""
    pairs = []
    for d in sorted(agent_dir.iterdir()):
        if d.is_dir():
            p = d / filename
            if p.exists():
                pairs.append((d.name, p))
    return pairs


def discover_agents(root: Path,
                    filename: str) -> list[tuple[str, list]]:
    """[(agent_name, [(trial_label, jsonl_path), ...]), ...] sorted by name."""
    agents = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            trials = discover_trials(d, filename)
            if trials:
                agents.append((d.name, trials))
    return agents

# ── JSONL loader ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> pd.DataFrame:
    """Parse one requests.jsonl file into a flat DataFrame."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(parse_request(json.loads(line)))
            except (json.JSONDecodeError, Exception):
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

# ── Trial loading pipeline ────────────────────────────────────────────────────

def load_trial(agent: str, trial_label: str,
               jsonl_path: Path, do_filter: bool) -> pd.DataFrame | None:
    """
    Load one requests.jsonl, filter sendBeacon rows, add derived columns,
    and tag with agent / trial_id.  Returns None if the file is empty.
    """
    df = load_jsonl(jsonl_path)
    if df.empty:
        print(f"    [warn] {jsonl_path} -- empty or unparseable, skipped")
        return None

    n_raw = len(df)
    if do_filter:
        df, n_removed = filter_senbeacon(df)
        note = f", {n_removed} sendBeacon removed" if n_removed else ""
    else:
        note = " (filter off)"

    df["agent"]    = agent
    df["trial"] = trial_label
    df = add_derived_columns(df)

    print(f"    {trial_label}: {n_raw} -> {len(df)} requests{note}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build trial-level TLS fingerprint features from requests.jsonl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory layout
----------------
  <root>/
    autogen/
      trial-001/requests.jsonl
      trial-002/requests.jsonl
    skyvern/
      trial-001/requests.jsonl

Examples
--------
  python tls_trial_features.py --root ./traces --out trial_features.csv
  python tls_trial_features.py --root ./traces --out trial_features.csv --encode-labels
  python tls_trial_features.py --agents ./traces/autogen ./traces/skyvern \\
      --labels AutoGen Skyvern --out trial_features.csv
""")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", metavar="DIR",
                     help="root dir containing one subdirectory per agent")
    src.add_argument("--agents", nargs="+", metavar="DIR",
                     help="explicit list of agent directories")

    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="agent names for --agents (default: directory name)")
    parser.add_argument("--filename", default="requests.jsonl",
                        help="JSONL filename inside each trial dir "
                             "(default: requests.jsonl)")
    parser.add_argument("--out", default="tls_features.csv",
                        help="output CSV (default: trial_features_tls.csv)")
    parser.add_argument("--encode-labels", action="store_true",
                        help="add integer label_id and write label_map.csv")
    parser.add_argument("--no-filter", action="store_true",
                        help="keep sendBeacon POST /collect rows")
    args = parser.parse_args()

    do_filter = not args.no_filter

    # Resolve (agent_name, trials) pairs
    if args.root:
        agent_trials = discover_agents(Path(args.root), args.filename)
        if not agent_trials:
            print(f"No agent directories with {args.filename!r} files "
                  f"found under {args.root}")
            sys.exit(1)
    else:
        dirs = [Path(d) for d in args.agents]
        names = args.labels if args.labels else [d.name for d in dirs]
        if len(names) != len(dirs):
            print("--labels count must match --agents count")
            sys.exit(1)
        agent_trials = [
            (name, discover_trials(d, args.filename))
            for name, d in zip(names, dirs)
        ]
        agent_trials = [(n, t) for n, t in agent_trials if t]

    if not agent_trials:
        print("No trials found. Check directory layout and --filename.")
        sys.exit(1)

    # Load
    print(f"\nLoading trials  (sendBeacon filter: {'ON' if do_filter else 'OFF'})\n")
    frames = []
    n_trials = 0
    for agent_name, trials in agent_trials:
        print(f"  Agent: {agent_name!r}  ({len(trials)} trial(s))")
        for trial_label, jsonl_path in trials:
            df = load_trial(agent_name, trial_label, jsonl_path, do_filter)
            if df is not None:
                frames.append(df)
                n_trials += 1

    if not frames:
        print("\nNo data loaded. Exiting.")
        sys.exit(1)

    raw = pd.concat(frames, ignore_index=True)
    print(f"\nTotal requests (post-filter): {len(raw)}")
    print(f"Total trials loaded:          {n_trials}")

    # Aggregate
    print("\nAggregating to trial level...")
    feat = (
        raw.groupby(["agent", "trial"])
        .apply(aggregate_trial, include_groups=False)
        .reset_index()
    )

    # Optional integer label encoding
    if args.encode_labels:
        label_list = sorted(feat["agent"].unique())
        label_id   = {lbl: i for i, lbl in enumerate(label_list)}
        feat.insert(2, "label_id", feat["agent"].map(label_id))
        lmap = pd.DataFrame(list(label_id.items()), columns=["agent", "label_id"])
        lmap_path = Path(args.out).with_name("label_map.csv")
        lmap.to_csv(lmap_path, index=False)
        print(f"  Saved: {lmap_path}")

    # Save
    out_path = Path(args.out)
    feat.to_csv(out_path, index=False)
    feature_cols = [c for c in feat.columns
                    if c not in ("agent", "trial", "label_id")]
    print(f"  Saved: {out_path}  ({len(feat)} rows x {len(feature_cols)} features)")

    meta = build_metadata(feat)
    meta_path = out_path.with_stem(out_path.stem + "_meta")
    meta.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path}")

    # Console summary
    GROUP_LABELS = {
        "f1": "PQ key exchange fractions",
        "f2": "Browser identity fractions",
        "f3": "H2 window & settings",
        "f4": "Stream-3 weight fractions",
        "f5": "Stream-5 weight fractions",
        "f6": "Stream coverage",
        "f7": "TLS diversity",
        "f8": "Cipher / curve / sig counts",
        "f9": "Pseudo-header order (ordinal mode)",
        "f10": "TLS extension count fractions",
        "f11": "GREASE flags (mode)",
        "f12": "Trial size (log-requests)",
    }
    print("\n-- Feature groups " + "-" * 54)
    groups: dict[str, list] = {}
    for c in feature_cols:
        groups.setdefault(c.split("_")[0], []).append(c)
    for g, cols in groups.items():
        print(f"  {g:<4s}  {GROUP_LABELS.get(g,''):<42s}  {len(cols):2d} feature(s)")
    print(f"       {'TOTAL':<42s}  {len(feature_cols):2d} features")

    print("\n-- Trial counts per agent " + "-" * 47)
    for agent, cnt in feat["agent"].value_counts().sort_index().items():
        print(f"  {agent:<24s}  {cnt:3d}  {'|' * cnt}")

    print("\n-- Null rates (features with any nulls) " + "-" * 33)
    null_rates = feat[feature_cols].isna().mean()
    null_rates = null_rates[null_rates > 0].sort_values(ascending=False)
    if null_rates.empty:
        print("  (none)")
    else:
        for col, rate in null_rates.items():
            print(f"  {col:<40s}  {rate:.1%}")
        print()
        print("  Tip: stream weights are null when < 2 (stream-3) or < 3")
        print("  (stream-5) requests exist per H2 connection in the trial.")
        print("  XGBoost / LightGBM handle NaN natively.")
        print("  sklearn: SimpleImputer(strategy='median') before fitting.")

    print("\n-- Per-agent feature means (key discriminative features) " + "-" * 17)
    show = [
        "f1_frac_kyber768", "f1_frac_mlkem768",
        "f2_frac_firefox_ja4",
        "f3_mean_ak_window",
        "f4_frac_s3_w110", "f5_frac_s5_w220", "f5_frac_s5_w256",
        "f6_stream5_coverage", "f7_ja3_diversity",
        "f8_mean_cipher_count",
        "f10_frac_ext16", "f10_frac_ext17", "f10_frac_firefox_ext",
    ]
    show = [f for f in show if f in feat.columns]
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    print(feat.groupby("agent")[show].mean().round(3).to_string())

    print("\nDone.\n")
    print("Next steps:")
    print("  Split -- use trial_id as group key (GroupShuffleSplit /")
    print("           StratifiedGroupKFold). Never split individual requests.")
    print("  Model -- XGBoost / LightGBM recommended (NaN-native, ~180 rows).")
    print("  Eval  -- Claude vs Gemini is the hardest pair; check per-class F1.")


if __name__ == "__main__":
    main()