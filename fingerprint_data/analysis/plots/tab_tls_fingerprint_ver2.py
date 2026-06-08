#!/usr/bin/env python3
"""
TLS Fingerprint Discrimination Table Generator
-----------------------------------------------
Removes POST /collect (sendBeacon) traffic from each user file,
then generates a LaTeX table summarising the key fingerprint fields
that distinguish the six users.

Usage:
    python tls_fingerprint_table.py

Output:
    fingerprint_table.tex   — standalone LaTeX document (compile with pdflatex)
    fingerprint_table.txt   — plain-text preview of the table values
"""

import os
import re
import pandas as pd

# ── 1. Load and clean ────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), ".")
agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}

def load_user(f: str, data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df_cleaned, h2_df_cleaned) for user n (1-based)."""
    # path = os.path.join(data_dir, f)
    path = os.path.join(f)
    df = pd.read_csv(path)

    # Remove sendBeacon traffic: POST /collect
    before = len(df)
    df = df[~((df["method"] == "POST") & (df["path"] == "/collect"))].copy()
    after = len(df)
    removed = before - after
    print(f"  User {f.split('/')[1]}: removed {removed:4d} POST /collect rows "
          f"({removed / before * 100:.1f}%)  →  {after} rows remain")

    h2 = df[df["negotiated_proto"] == "h2"].copy()
    return df, h2


print("Loading files and removing sendBeacon traffic …")
users: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

files  = [f"analysis/{agent}/tls_summary.csv" for agent in agents]

for i, f in enumerate(files):
    users[i] = load_user(f)
print()

# ── 2. Extract per-user metrics ──────────────────────────────────────────────

def pct(series: pd.Series, mask: pd.Series) -> str:
    """Return '36%' style string."""
    return f"{mask.mean() * 100:.0f}\\%"

def mode_val(series: pd.Series) -> str:
    return str(series.mode().iloc[0])

def stream5_weight(h2: pd.DataFrame) -> str:
    """Dominant weight assigned to HTTP/2 stream 5 (3rd request)."""
    has_s5 = h2["ak_priority_str"].dropna()
    has_s5 = has_s5[has_s5.str.contains(r"5:1:", na=False)]
    if has_s5.empty:
        return "—"
    weights = has_s5.str.extract(r"5:1:\d+:(\d+)")[0].value_counts()
    top_w = weights.index[0]
    top_pct = weights.iloc[0] / len(has_s5) * 100
    return f"{top_w} ({top_pct:.0f}\\%)"


rows = []
for i, a in enumerate(agents):
    df, h2 = users[i]

    kyber_mask  = h2["curve_no_grease_str"].str.contains("Kyber768", na=False)
    mlkem_mask  = h2["curve_no_grease_str"].str.contains("MLKEM",    na=False)
    firefox_mask = h2["ja4_a"].str.startswith("t13d2812", na=False)

    row = {
        "user":          a,
        "h2_rows":       len(h2),
        "source_ips":    df["source_ip"].nunique(),
        "ja3_unique":    h2["ja3"].nunique(),
        # PQ key exchange
        "kyber768_pct":  pct(h2, kyber_mask),
        "mlkem_pct":     pct(h2, mlkem_mask),
        # JA4 fingerprint
        "ja4_ext_count": mode_val(h2["ja4_ext_count"]),
        "ja4_c":         mode_val(h2["ja4_c"]),
        # Browser mix
        "firefox_pct":   pct(h2, firefox_mask),
        # H2 settings
        "ak_window":     f"{int(h2['ak_window_update'].mode().iloc[0]):,}",
        "ak_pseudo":     mode_val(h2["ak_pseudo_order"]),
        # H2 priority — stream 5 weight distinguishes users 3 vs 4
        "s5_weight":     stream5_weight(h2),
    }
    rows.append(row)

# ── 3. Print plain-text preview ──────────────────────────────────────────────

print("Plain-text preview of table values:")
print("-" * 70)
for r in rows:
    print(f"User {r['user']}  |  "
          f"Kyber={r['kyber768_pct']:>4s}  MLKEM={r['mlkem_pct']:>4s}  "
          f"FF={r['firefox_pct']:>4s}  "
          f"ext={r['ja4_ext_count']}  JA4c={r['ja4_c']}  "
          f"IPs={r['source_ips']:2d}  JA3={r['ja3_unique']:3d}  "
          f"s5_w={r['s5_weight']}")
print()

# ── 4. Build LaTeX ───────────────────────────────────────────────────────────

def bold(s: str) -> str:
    return r"\textbf{" + s + "}"

def cell(s: str, highlight: bool = False) -> str:
    return bold(s) if highlight else s


def build_latex(rows: list[dict]) -> str:
    """Construct a standalone LaTeX document containing the table."""

    # Identify cells that are unique across all users (for highlighting)
    def unique_vals(key: str) -> set[str]:
        """Values that appear in exactly one user's row."""
        from collections import Counter
        c = Counter(r[key] for r in rows)
        return {v for v, n in c.items() if n == 1}

    unique_kyber   = unique_vals("kyber768_pct")
    unique_mlkem   = unique_vals("mlkem_pct")
    unique_ext     = unique_vals("ja4_ext_count")
    unique_ja4c    = unique_vals("ja4_c")
    unique_ff      = unique_vals("firefox_pct")
    unique_window  = unique_vals("ak_window")
    unique_pseudo  = unique_vals("ak_pseudo")
    unique_s5      = unique_vals("s5_weight")
    unique_ips     = unique_vals("source_ips")
    unique_ja3     = unique_vals("ja3_unique")

    def hl(val: str, unique_set: set) -> str:
        return bold(val) if val in unique_set else val

    lines: list[str] = []

    # Preamble
    lines += [
        # r"\documentclass[10pt]{article}",
        # r"\usepackage[margin=1.5cm, landscape]{geometry}",
        # r"\usepackage{booktabs}",
        # r"\usepackage{xcolor}",
        # r"\usepackage{array}",
        # r"\usepackage{makecell}",
        # r"\usepackage{helvet}",
        # r"\renewcommand{\familydefault}{\sfdefault}",
        # r"\setlength{\tabcolsep}{6pt}",
        # r"\renewcommand{\arraystretch}{1.25}",
        # r"",
        # r"\begin{document}",
        # r"\pagestyle{empty}",
        # r"",
        # r"{\small",
        r"\begin{table*}[ht]",
        r"\centering",
        (r"\caption{TLS/H2 fingerprint discrimination after removing "
         r"\texttt{POST /collect} (sendBeacon) traffic. "
         r"\textbf{Bold} values are unique to that user.}"),
        r"\label{tab:tls_fingerprints}",
        r"",
    ]

    # Column spec  (l + 10 centred cols)
    col_spec = (
        r"@{}"
        r"l"                # User
        r"r"                # H2 rows
        r"r"                # Source IPs
        r"r"                # JA3 unique
        r"r"                # Kyber768 %
        r"r"                # MLKEM %
        r"r"                # Firefox %
        r"c"                # JA4 ext count
        r"l"                # JA4_c hash (dominant)
        r"r"                # AK window update
        r"c"                # AK pseudo order
        r"c"                # Stream-5 weight
        r"@{}"
    )
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row 1
    lines.append(
        r"\multicolumn{4}{l}{\textit{Session}} & "
        r"\multicolumn{3}{c}{\textit{PQ Key Exchange (\% of H2)}} & "
        r"\multicolumn{2}{c}{\textit{JA4}} & "
        r"\multicolumn{3}{c}{\textit{HTTP/2 Settings}} \\"
    )
    lines.append(r"\cmidrule(lr){1-4} \cmidrule(lr){5-7} "
                 r"\cmidrule(lr){8-9} \cmidrule(lr){10-12}")

    # Header row 2
    lines.append(
        r"Agent & "
        r"\makecell[r]{H2\\rows} & "
        r"\makecell[r]{Source\\IPs} & "
        r"\makecell[r]{JA3\\unique} & "
        r"\makecell[r]{Kyber768\\(\%)} & "
        r"\makecell[r]{MLKEM768\\(\%)} & "
        r"\makecell[r]{Firefox\\JA4 (\%)} & "
        r"\makecell[c]{Ext\\count} & "
        r"\makecell[l]{JA4\_c hash\\(dominant)} & "
        r"\makecell[r]{AK window\\update} & "
        r"\makecell[c]{AK pseudo\\order} & "
        r"\makecell[c]{Stream-5\\weight} \\"
    )
    lines.append(r"\midrule")

    # Data rows
    for r in rows:
        u     = str(r["user"])
        h2r   = str(r["h2_rows"])
        ips   = hl(str(r["source_ips"]), unique_ips)
        ja3   = hl(str(r["ja3_unique"]), unique_ja3)
        kyber = hl(r["kyber768_pct"],    unique_kyber)
        mlkem = hl(r["mlkem_pct"],       unique_mlkem)
        ff    = hl(r["firefox_pct"],     unique_ff)
        ext   = hl(r["ja4_ext_count"],   unique_ext)
        ja4c  = hl(r["ja4_c"],           unique_ja4c)
        win   = hl(r["ak_window"],       unique_window)
        ps    = hl(r["ak_pseudo"],       unique_pseudo)
        s5    = hl(r["s5_weight"],       unique_s5)

        # Monospace for hash: wrap value in \texttt, preserve any \textbf
        if ja4c.startswith(r"\textbf{"):
            inner = ja4c[len(r"\textbf{"):-1]   # strip \textbf{ … }
            ja4c_tt = r"\textbf{\texttt{" + inner + "}}"
        else:
            ja4c_tt = r"\texttt{" + ja4c + "}"

        lines.append(
            f"\\textbf{{{agent_labels[u]}}} & {h2r} & {ips} & {ja3} & "
            f"{kyber} & {mlkem} & {ff} & "
            f"{ext} & {ja4c_tt} & "
            f"{win} & \\texttt{{{ps}}} & {s5} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\medskip",
        r"\noindent\small",
        (r"\textit{Notes:} "
         r"sendBeacon traffic (\texttt{POST /collect}) removed before analysis. "
         r"All metrics computed on HTTP/2 connections only, except "
         r"\textit{Source IPs} (all protocols). "
         r"PQ key-exchange columns show the fraction of H2 handshakes "
         r"advertising that curve group. "
         r"JA4\_c is the sorted, hashed TLS extension list "
         r"(dominant value shown). "
         r"\textit{Stream-5 weight} is the HTTP/2 priority weight Chrome "
         r"assigns to the 3rd concurrent request (stream ID 5), "
         r"the only reliable signal separating Users 3 and 4."),
        r"\end{table*}",
    ]

    return "\n".join(lines)


latex = build_latex(rows)

# ── 5. Write outputs ─────────────────────────────────────────────────────────

tex_path = os.path.join("analysis/plots/tls_fingerprint_table.tex")
# txt_path = os.path.join(os.path.dirname(__file__), "fingerprint_table.txt")

with open(tex_path, "w") as f:
    f.write(latex)

# Plain-text version (strip LaTeX commands)
plain = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", latex)   # \cmd{text} → text
plain = re.sub(r"\\[a-zA-Z]+",            "",   plain)     # bare commands
plain = re.sub(r"[{}\[\]]",               "",   plain)     # braces/brackets
plain = re.sub(r"[ \t]{2,}",             " ",  plain)      # collapse spaces
# with open(txt_path, "w") as f:
#     f.write(plain)

print(f"Wrote:  {tex_path}")
# print(f"        {txt_path}")
print()
print("Compile with:  pdflatex fingerprint_table.tex")
print()

# ── 6. Print the generated LaTeX for inspection ──────────────────────────────
print("=" * 70)
print("Generated LaTeX:")
print("=" * 70)
print(latex)