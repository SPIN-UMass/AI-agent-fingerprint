#'''
#Generate a LaTeX comparison table of HTTP header fingerprints across
#six AI-agent users, after filtering out sendBeacon (POST /collect) traffic.
#
#Usage:
#    python generate_header_fingerprint_table.py [base_dir]
#
#    base_dir defaults to the current directory.
#    Expected layout:
#        <base_dir>/
#            autogen_websurfer/http_header_summary.csv
#            browser_use/http_header_summary.csv
#            claude_computer_use/http_header_summary.csv
#            gemini_computer_use/http_header_summary.csv
#            operator/http_header_summary.csv
#            skyvern/http_header_summary.csv
#
#Output:
#    header_fingerprint_table.tex
#
#Required LaTeX packages:
#    \usepackage{booktabs, xcolor, graphicx, rotating}
#'''

import sys
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

USERS = [
    "autogen_websurfer",
    "browser_use",
    "claude_computer_use",
    "gemini_computer_use",
    "operator",
    "skyvern",
]

USER_LABELS = {
    "autogen_websurfer":   "AutoGen",
    "browser_use":         "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator":            "Operator",
    "skyvern":             "Skyvern",
}

# Standard browser header names (lowercase); anything outside this set
# is treated as a non-standard / injected header.
STANDARD_HEADERS = {
    "accept", "accept-encoding", "accept-language", "user-agent",
    "priority", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
    "sec-fetch-mode", "sec-fetch-dest", "sec-fetch-site", "sec-fetch-user",
    "upgrade-insecure-requests", "referer", "origin", "content-type",
    "content-length", "cache-control", "connection", "te", "dnt",
    "if-modified-since", "if-none-match", "range", "authorization",
    "cookie", "host", "transfer-encoding",
}

# ─────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────

def latex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    for ch, rep in [
        ("\\", r"\textbackslash{}"),
        ("%",  r"\%"),
        ("&",  r"\&"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(ch, rep)
    return s


def to_bool(series: pd.Series) -> pd.Series:
    """Coerce True/False strings and native booleans to bool dtype."""
    return series.map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
        if pd.notna(v) else False
    ).astype(bool)


def pct_cell(series: pd.Series,
             anomaly_if_low: float = None,
             anomaly_if_high: float = None) -> str:
    """
    Return a percentage string.
    Coloured red when the value crosses the anomaly threshold.
    """
    if series.empty:
        return "--"
    b = to_bool(series)
    p = 100.0 * b.sum() / max(len(b), 1)
    s = rf"{p:.0f}\%"
    if anomaly_if_low is not None and p <= anomaly_if_low:
        s = rf"\textcolor{{red}}{{{s}}}"
    elif anomaly_if_high is not None and p > anomaly_if_high:
        s = rf"\textcolor{{red}}{{{s}}}"
    return s


def mode_cell(series: pd.Series, max_len: int = 20) -> str:
    """
    Most-common value.
    Appends superscript * when more than one distinct value appears.
    """
    s = series.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        return "--"
    top = s.mode().iloc[0]
    suffix = r"\textsuperscript{*}" if s.nunique() > 1 else ""
    return latex_escape(str(top)[:max_len]) + suffix


def unique_hash_cell(series: pd.Series) -> str:
    """
    'n_unique / n_total' for header_order_hash.
    Printed in red with a star when every ordering is unique
    (fully randomised – the primary bot signal).
    """
    if series.empty:
        return "--"
    n_total  = len(series)
    n_unique = series.nunique()
    star = r" $\bigstar$" if n_unique == n_total else ""
    cell = rf"{n_unique}/{n_total}{star}"
    if n_unique == n_total:
        cell = rf"\textcolor{{red}}{{{cell}}}"
    return cell


def mean_std_cell(series: pd.Series) -> str:
    """'mu +/- sigma' for a numeric series."""
    num = pd.to_numeric(series, errors="coerce").dropna()
    if num.empty:
        return "--"
    return rf"{num.mean():.1f}$\pm${num.std():.1f}"


def detect_custom_headers(df: pd.DataFrame) -> str:
    """
    Scan header_order_str for names outside the standard browser set.
    Returns up to two examples or 'none'.
    """
    if "header_order_str" not in df.columns:
        return r"\textit{n/a}"
    found: set[str] = set()
    for row in df["header_order_str"].dropna():
        for h in str(row).split(","):
            h = h.strip().lower()
            if h and h not in STANDARD_HEADERS:
                found.add(h)
    if not found:
        return r"\textit{none}"
    examples = sorted(found)[:2]
    return latex_escape(", ".join(rf"\texttt{{{e}}}" for e in examples))


def sf_site_none_pct(nav: pd.DataFrame) -> str:
    """
    Percentage of navigation requests with Sec-Fetch-Site == 'none'.
    100 % means the referrer chain is always broken (direct-open pattern).
    """
    if nav.empty or "sf_site" not in nav.columns:
        return "--"
    p = 100.0 * (nav["sf_site"] == "none").sum() / max(len(nav), 1)
    s = rf"{p:.0f}\%"
    if p == 100.0 and len(nav) > 0:
        s = rf"\textcolor{{red}}{{{s}}}"
    return s


# ─────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────

def load_user(csv_path: Path) -> dict:
    """
    Load one user CSV, strip sendBeacon rows, and return sub-frames:
        'all'  – every request except POST /collect
        'nav'  – navigation GET requests  (req_type == 'navigate')
        'sub'  – subresource requests     (req_type starts with 'subresource')
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter out sendBeacon (POST /collect)
    is_beacon = (
        (df["method"] == "POST") &
        (df["path"].str.contains("/collect", na=False))
    )
    df = df[~is_beacon].copy()

    nav = df[df["req_type"] == "navigate"].copy()
    sub = df[df["req_type"].str.startswith("subresource", na=False)].copy()

    return {"all": df, "nav": nav, "sub": sub}


# ─────────────────────────────────────────────────────────────────────
#  LaTeX table builder
# ─────────────────────────────────────────────────────────────────────

def build_table(user_data: dict, users: list) -> str:
    """Build and return the complete LaTeX table string."""

    n         = len(users)
    col_spec  = "l" + "c" * n
    col_heads = " & ".join(
        rf"\rotatebox{{55}}{{\textbf{{{USER_LABELS.get(u, latex_escape(u))}}}}}"
        for u in users
    )

    # ── helpers ──────────────────────────────────────────────────────

    def sec_header(title: str) -> list:
        """Return the three lines that open a grouped section."""
        return [
            r"      \midrule",
            (rf"      \multicolumn{{{n+1}}}{{l}}{{"
             rf"\small\textit{{\textbf{{{title}}}}}}}" + r" \\"),
            r"      \midrule[0.3pt]",
        ]

    def row(label: str, cells: list) -> str:
        return "      " + label + " & " + " & ".join(cells) + r" \\"

    # ── build body ───────────────────────────────────────────────────
    body: list[str] = []

    # ── 1. Browser Identity ──────────────────────────────────────────
    body.extend(sec_header("1. Browser Identity  (navigation GETs)"))

    body.append(row(
        r"UA family (mode)",
        [mode_cell(user_data[u]["nav"]["ua_family"]) for u in users]))

    body.append(row(
        r"UA OS (mode)",
        [mode_cell(user_data[u]["nav"]["ua_os"]) for u in users]))

    body.append(row(
        r"UA version --- major (mode)",
        [mode_cell(user_data[u]["nav"]["ua_version"].astype(str))
         for u in users]))

    cells = []
    for u in users:
        p = 100.0 * to_bool(user_data[u]["nav"]["ch_is_headless"]).mean()
        s = rf"{p:.0f}\%"
        if p > 10:
            s = rf"\textcolor{{red}}{{{s}}}"
        cells.append(s)
    body.append(row(r"Headless Chrome detected (\%)", cells))

    body.append(row(
        r"\texttt{Sec-Ch-Ua} brand count (mode)",
        [mode_cell(user_data[u]["nav"]["ch_brand_count"].astype(str))
         for u in users]))

    # ── 2. Header Order Randomisation ────────────────────────────────
    body.extend(sec_header(
        r"2. Header Order Randomisation  (navigation GETs) \quad"
        r"{\footnotesize $\bigstar$\,=\,all orderings unique}"))

    body.append(row(
        r"Unique orderings / total nav requests",
        [unique_hash_cell(user_data[u]["nav"]["header_order_hash"])
         for u in users]))

    body.append(row(
        r"Num.\ headers per request ($\mu \pm \sigma$)",
        [mean_std_cell(user_data[u]["nav"]["n_headers"]) for u in users]))

    # ── 3. Header Field Presence ──────────────────────────────────────
    body.extend(sec_header(
        r"3. Header Field Presence  (\% of navigation GETs)"))

    PRESENCE_ROWS = [
        ("has_accept_language",
         r"\texttt{Accept-Language}  \textit{(absent = bot signal)}",
         {"anomaly_if_low": 50}),
        ("has_sec_fetch_user",
         r"\texttt{Sec-Fetch-User: ?1}  \textit{(absent = bot signal)}",
         {"anomaly_if_low": 50}),
        ("has_priority",
         r"\texttt{Priority}",
         {}),
        ("has_sec_ch_ua",
         r"\texttt{Sec-Ch-Ua}",
         {}),
        ("has_sec_fetch_mode",
         r"\texttt{Sec-Fetch-Mode}",
         {}),
        ("has_upgrade_insecure_requests",
         r"\texttt{Upgrade-Insecure-Requests}",
         {}),
        ("has_cache_control",
         r"\texttt{Cache-Control}  \textit{(unusual on nav GETs)}",
         {"anomaly_if_high": 5}),
        ("has_connection",
         r"\texttt{Connection}  \textit{(HTTP/1.1 artefact on h2)}",
         {"anomaly_if_high": 5}),
        ("has_te",
         r"\texttt{TE}  \textit{(HTTP/1.1 artefact on h2)}",
         {"anomaly_if_high": 5}),
        ("has_dnt",
         r"\texttt{DNT}",
         {}),
    ]

    for col, label, kw in PRESENCE_ROWS:
        cells = []
        for u in users:
            nav = user_data[u]["nav"]
            if col not in nav.columns:
                cells.append("--")
            else:
                cells.append(pct_cell(nav[col], **kw))
        body.append(row(label, cells))

    body.append(row(
        r"Non-standard headers (examples)",
        [detect_custom_headers(user_data[u]["nav"]) for u in users]))

    # ── 4. Sec-Fetch Signals ──────────────────────────────────────────
    body.extend(sec_header(r"4. Sec-Fetch Signals  (navigation GETs)"))

    body.append(row(
        r"\texttt{Sec-Fetch-Site} value (mode)",
        [mode_cell(user_data[u]["nav"]["sf_site"]) for u in users]))

    body.append(row(
        r"\texttt{Sec-Fetch-Site} always \texttt{none} (\%)"
        r"  \textit{(= broken referrer chain)}",
        [sf_site_none_pct(user_data[u]["nav"]) for u in users]))

    body.append(row(
        r"\texttt{Sec-Fetch-User} value (mode)",
        [mode_cell(
            user_data[u]["nav"]["sf_user"].fillna("absent").astype(str))
         for u in users]))

    cells = []
    for u in users:
        nav = user_data[u]["nav"]
        col = "sf_sf_any_violation"
        if col not in nav.columns:
            cells.append("--")
            continue
        p = 100.0 * to_bool(nav[col]).mean()
        s = rf"{p:.0f}\%"
        if p > 0:
            s = rf"\textcolor{{red}}{{{s}}}"
        cells.append(s)
    body.append(row(r"Sec-Fetch logic violation (\%)", cells))

    # ── 5. Client Hints Coherence ─────────────────────────────────────
    body.extend(sec_header(
        r"5. Client Hints (\texttt{Sec-Ch-Ua}) Coherence  (navigation GETs)"))

    cells = []
    for u in users:
        nav = user_data[u]["nav"]
        col = "ch_any_incoherence"
        if col not in nav.columns:
            cells.append("--")
            continue
        p = 100.0 * to_bool(nav[col]).mean()
        s = rf"{p:.0f}\%"
        if p > 10:
            s = rf"\textcolor{{red}}{{{s}}}"
        cells.append(s)
    body.append(row(r"UA vs.\ \texttt{Sec-Ch-Ua} incoherence (\%)", cells))

    body.append(row(
        r"Mobile hint matches UA (\%)",
        [pct_cell(user_data[u]["nav"]["ch_mobile_matches_ua"])
         for u in users]))

    body.append(row(
        r"Platform hint matches UA (\%)",
        [pct_cell(user_data[u]["nav"]["ch_platform_matches_ua"])
         for u in users]))

    # ── 6. Priority Header ────────────────────────────────────────────
    body.extend(sec_header(
        r"6. \texttt{Priority} Header  (navigation GETs, "
        r"expected: \texttt{u=0, i})"))

    body.append(row(
        r"Urgency value (mode)",
        [mode_cell(user_data[u]["nav"]["priority_urgency"].astype(str))
         for u in users]))

    body.append(row(
        r"Incremental flag (mode)",
        [mode_cell(user_data[u]["nav"]["priority_incr"].astype(str))
         for u in users]))

    body.append(row(
        r"Priority value correct (\%)",
        [pct_cell(user_data[u]["nav"]["priority_expected"])
         for u in users]))

    # ── 7. Subresource requests ───────────────────────────────────────
    body.extend(sec_header(
        r"7. Subresource Requests  (script / image / …)"))

    body.append(row(
        r"Subresource requests observed",
        [str(len(user_data[u]["sub"])) for u in users]))

    body.append(row(
        r"\texttt{Accept-Language} present (\%)",
        [pct_cell(user_data[u]["sub"]["has_accept_language"])
         if "has_accept_language" in user_data[u]["sub"].columns
         else "--"
         for u in users]))

    cells = []
    for u in users:
        sub = user_data[u]["sub"]
        if sub.empty or "header_order_hash" not in sub.columns:
            cells.append("--")
        else:
            cells.append(unique_hash_cell(sub["header_order_hash"]))
    body.append(row(r"Unique orderings / total subresource requests", cells))

    # ── Assemble final LaTeX ──────────────────────────────────────────
    comment = (
        "% ─────────────────────────────────────────────────────────\n"
        "% Required LaTeX packages:\n"
        "%   \\usepackage{booktabs, xcolor, graphicx, rotating}\n"
        "%\n"
        "% Legend:\n"
        "%   RED cell   = bot-indicative / anomalous value\n"
        "%   *          = value inconsistent across trials\n"
        "%   $\\bigstar$ = all orderings unique (fully randomised)\n"
        "% ─────────────────────────────────────────────────────────\n"
    )

    caption_lines = [
        r"    HTTP header fingerprint comparison across six AI-agent "
        r"frameworks (30 trials each).",
        r"    Only navigation \texttt{GET} requests are analysed "
        r"(\texttt{POST /collect} sendBeacon traffic excluded).",
        (r"    \textcolor{red}{Red}\,=\,bot-indicative anomaly;\quad "
         r"$\bigstar$\,=\,all orderings unique (fully randomised);\quad "
         r"\textsuperscript{*}\,=\,value inconsistent across trials.%"),
    ]

    lines = [
        comment,
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{%",
        *[f"  {c}" for c in caption_lines],
        r"  }",
        r"  \label{tab:agent-header-fingerprint}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"    \begin{{tabular}}{{{col_spec}}}",
        r"      \toprule",
        rf"      \textbf{{Feature}} & {col_heads} \\",
        r"      \toprule",
        *body,
        r"      \bottomrule",
        r"    \end{tabular}%",
        r"  }",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not base.is_dir():
        sys.exit(f"[!] Not a directory: {base}")

    print(f"[*] Base directory : {base.resolve()}\n")

    user_data: dict = {}
    for u in USERS:
        csv_path = base / u / "http_header_summary.csv"
        if not csv_path.is_file():
            print(f"  [!] Missing: {csv_path}  -- skipping")
            continue

        data = load_user(csv_path)
        user_data[u] = data

        n_all  = len(data["all"])
        n_nav  = len(data["nav"])
        n_sub  = len(data["sub"])
        n_uniq = data["nav"]["header_order_hash"].nunique() if n_nav else 0
        tag    = ("*** RANDOMISED ***"
                  if n_nav > 0 and n_uniq == n_nav else "consistent")

        print(
            f"  [{u}]  total={n_all}  nav={n_nav}  sub={n_sub}  "
            f"nav-order-hashes: {n_uniq}/{n_nav} unique  ({tag})"
        )

    if not user_data:
        sys.exit("\n[!] No CSV files found. Check directory layout.")

    present = [u for u in USERS if u in user_data]
    print(f"\n[*] Building table for {len(present)} users: {present}\n")

    latex = build_table(user_data, present)

    out = Path("header_fingerprint_table.tex")
    out.write_text(latex, encoding="utf-8")
    print(f"[+] Written : {out.resolve()}")
    print("\n" + "=" * 70)
    print(latex)


if __name__ == "__main__":
    main()

