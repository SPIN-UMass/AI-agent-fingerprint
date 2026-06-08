import pandas as pd

# BOOL_FEATS = [
#     ("SNI present",             "sni_present"),
#     ("ALPN: h2 offered",        "alpn_has_h2"),
#     ("ALPN: http/1.1 offered",  "alpn_has_http11"),
#     ("H2 negotiated",           "h2_negotiated"), #
#     ("GREASE (ciphers)",        "grease_cipher"), #
#     ("GREASE (versions)",       "grease_version"), #
#     ("Post-quantum crypto",     "has_pq_crypto"),
#     ("TLS 1.1 advertised",      "tls11_adv"), #
#     ("H2 SETTINGS present",     "h2_settings"), #
# ]

agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

files  = [f"analysis/{agent}/tls_fingerprint_vector.csv" for agent in agents]

agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}

FEATURES = {
    "sni_present": "SNI",
    "alpn_has_h2": "ALPN h2",
    "alpn_has_http11": "ALPN h1.1",
    "h2_negotiated": "H2 Neg.",
    "cipher_has_grease": "GREASE Ciphers",
    "supported_ver_has_grease": "GREASE Ver.",
    "has_pq_crypto": "PQ Crypto",
    "supported_ver_has_tls11": "TLS1.1",
    "h2_settings": "H2 Settings",
}

def add_derived(df):
    df = df.copy()
    df["h2_negotiated"] = df["negotiated_proto"] == "h2"
    df["h2_settings"] = df["ak_settings_str"].notna() & (df["ak_settings_str"] != "")
    return df

FEATURES_EXT = list(FEATURES.keys()) #+ ["h2_negotiated", "h2_settings"]

# --------- HELPERS ---------
def summarize(series):
    """Return (status, symbol, counts)."""
    n_true = (series == True).sum()
    n_false = (series == False).sum()

    counts = {"True": int(n_true), "False": int(n_false)}

    if n_true > 0 and n_false == 0:
        return "TRUE_ONLY", r"\checkmark", counts
    elif n_false > 0 and n_true == 0:
        return "FALSE_ONLY", r"\xmark", counts
    else:
        return "MIXED", rf"${n_true}/{n_false}$", counts


def to_latex_bool(status):
    if status == "TRUE_ONLY":
        return r"\checkmark"
    elif status == "FALSE_ONLY":
        return r"\xmark"
    else:
        return r"\textbf{⚠}"

# --------- LOAD + PROCESS ---------
rows = []
diagnostics = []

for file, agent in zip(files, agents):
    df = pd.read_csv(file)
    df = add_derived(df)

    label = agent_labels[agent]
    row_symbols = []

    diagnostics.append(f"\n===== {label} =====")

    for col in FEATURES_EXT:
        status, symbol, counts = summarize(df[col])
        row_symbols.append(symbol)

        # store diagnostics only for problematic cases
        if status == "MIXED":
            diagnostics.append(
                f"{col}: MIXED -> {counts} (n={len(df)})"
            )

    rows.append((label, row_symbols))

# --------- BUILD LATEX TABLE ---------
header = r"""
\begin{table*}[ht]
\centering
\caption{TLS / HTTP Fingerprint Features Across Agents}
\label{tab:tls-feature}
\begin{tabular}{lccccccccc}
\toprule
Agent &
SNI &
ALPN h2 &
ALPN h1.1 &
H2 Neg. &
GREASE Ciphers &
GREASE Ver. &
PQ Crypto &
TLS1.1 &
H2 Settings \\
\midrule
"""

body = ""

for name, symbols in rows:
    body += name + " & " + " & ".join(symbols) + r" \\" + "\n"

footer = r"""
\bottomrule
\end{tabular}
\end{table*}
"""

latex_table = header + body + footer

# --------- SAVE OUTPUT ---------
print(latex_table)

with open("analysis/plots/consistency_diagnostics.txt", "w") as f:
    f.write("\n".join(diagnostics))

print("LaTeX table written to table.tex")