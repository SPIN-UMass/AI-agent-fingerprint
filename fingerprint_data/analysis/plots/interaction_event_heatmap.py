"""
heatmap_users_events.py
────────────────────────────────────────────────────────────────────────────────
Draws a single heatmap:  Users (rows) × [Page → Event] (columns)

Each cell shows the proportion of that event type on that page for that user.
Pages and events are auto-selected by their cross-user variance so only the
most discriminating combinations appear in the plot.

Usage
─────
Place this script in the same folder as the 6 CSV files and run:
    python heatmap_users_events.py

Or point DATA_DIR at the folder containing the files.

Dependencies
────────────
    pip install pandas matplotlib numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR        = "analysis/"                               # folder with the 6 CSV files
FILE_PATTERN    = "interaction_event_profile.csv"
N_USERS         = 6
OUTPUT_PATH     = "analysis/plots/heatmap_users_events.pdf"
AGENTS = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

# Minimum cross-user variance for an event to appear (per-user-mean variance)
EVENT_VAR_THRESHOLD = 0.0005
# Minimum cross-user variance for a page to appear (mean-over-events variance)
PAGE_VAR_THRESHOLD  = 0.005

plt.rcParams.update({
    "font.size": 16,          # base font size
    # "axes.titlesize": 14,     # title size
    # "axes.labelsize": 20,     # x/y label size
    # "xtick.labelsize": 16,    # x tick labels
    # "ytick.labelsize": 16,    # y tick labels
    # "legend.fontsize": 16,    # legend
    # "figure.titlesize": 16    # figure title
})

agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}
page_labels = {
    "S1-subscribe-v1":"S1_v1",
    "S2-subscribe-v2":"S1_v2",
    "S3-subscribe-v3":"S1_v3",
    "S4-scroll-gate":"S2",
    "S5-hover-reveal":"S3",
    "S6-dom-mismatch":"S4",
    "Delayed Feedback — UX Behavior":"S5"
}
LABELS = list(agent_labels.values())
PAGE_LABEL = list(page_labels.values())

# Redundant, low-signal, or non-interaction columns to always exclude
DROP_EVENTS = {
    # ── non-interaction / lifecycle events (excluded by request) ──
    "app_event",   # application-state signals, not user input
    "page.1",      # page load/unload lifecycle, not user input
    # ── redundant symmetric pairs (keep only the more informative one) ──
    "keyup",       # mirrors keydown
    "mouseup",     # mirrors mousedown
    "mousedown",   # mirrors mouseup
    "focus",       # mirrors blur
    "blur",        # mirrors focus
    # ── near-zero variance across all users ──
    "change",      # near-zero for all users
    "selection",   # near-zero for all users
    "visibility",  # near-zero for all users
}
 
ALL_EVENT_COLS = [
    "app_event", "blur", "change", "click", "focus", "input",
    "keydown",   "keyup", "mousedown", "mousemove", "mouseup",
    "page.1",    "scroll", "visibility", "selection",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_data(data_dir, file_pattern, n_users, all_event_cols):
    """Load all CSVs into a single long-form DataFrame."""
    records = []
    for a in AGENTS:
        path = os.path.join(data_dir, f"{a}/interaction_event_profile.csv")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rec = {"user": agent_labels[a], "page": page_labels[row["page"]]}
            for c in all_event_cols:
                v = row[c] if c in row.index else np.nan
                rec[c] = float(v) if pd.notna(v) else 0.0
            records.append(rec)
    return pd.DataFrame(records)


def select_noticeable_events(df, all_event_cols, drop_events, threshold):
    """
    Rank events by the variance of their per-user mean across users.
    Keep those above the threshold, excluding known redundant columns.
    Returns: (sorted list of event names, dict of variances)
    """
    candidates = [c for c in all_event_cols if c not in drop_events and c in df.columns]
    variances = {
        c: df.groupby("user")[c].mean().var()
        for c in candidates
    }
    selected = sorted(
        [c for c, v in variances.items() if v >= threshold],
        key=lambda c: -variances[c],
    )
    return selected, variances

def select_noticeable_events(df, all_event_cols, drop_events, threshold):
    """
    Rank events by the variance of their per-user mean across users.
    Keep those above the threshold, excluding known redundant columns.
    Returns: (sorted list of event names, dict of variances)
    """
    candidates = [c for c in all_event_cols if c not in drop_events and c in df.columns]
    variances = {
        c: df.groupby("user")[c].mean().var()
        for c in candidates
    }
    selected = sorted(
        [c for c, v in variances.items() if v >= threshold],
        key=lambda c: -variances[c],
    )
    return selected, variances
 
 
def select_noticeable_pages(df, event_cols, threshold):
    """
    Keep only pages that:
      1. Are present for ALL users (no missing rows).
      2. Have cross-user variance (mean over selected events) above threshold.
    Pages are returned sorted naturally (S1 → S6) for readability.
    Returns: (sorted list of page names, dict of variances)
    """
    n_users = df["user"].nunique()
    page_counts = df.groupby("page")["user"].nunique()
    full_pages = page_counts[page_counts == n_users].index.tolist()
 
    variances = {}
    for page in full_pages:
        sub = df[df["page"] == page]
        ec = [c for c in event_cols if c in sub.columns]
        user_vecs = sub.groupby("user")[ec].mean()
        variances[page] = float(user_vecs.values.var())
 
    selected = sorted(
        [p for p, v in variances.items() if v >= threshold]
    )
    return selected, variances
 
 
def build_heatmap_matrix(df, users, pages, events):
    """
    Build a (n_users × n_pages*n_events) DataFrame with a MultiIndex column.
    Missing page/event combinations are filled with 0.
    """
    idx = pd.MultiIndex.from_product([pages, events], names=["page", "event"])
    matrix = pd.DataFrame(index=users, columns=idx, dtype=float)
    for user in users:
        udf = df[df["user"] == user].set_index("page")
        for page in pages:
            for event in events:
                if page in udf.index and event in udf.columns:
                    matrix.loc[user, (page, event)] = float(udf.loc[page, event])
                else:
                    matrix.loc[user, (page, event)] = 0.0
    return matrix.astype(float)
 
 
def make_colormap():
    """
    Sequential colormap: dark blue (low proportion) → light blue → amber/orange (high).
    Avoids diverging palettes since there is no meaningful zero-crossing midpoint.
    """
    colors = ["#0C447C", "#378ADD", "#E6F1FB", "#EF9F27", "#E85D24"]
    return LinearSegmentedColormap.from_list("user_heatmap", colors)
 
 
def draw_heatmap(matrix, pages, events, event_variances, page_variances, output_path):
    n_users  = len(matrix)
    n_pages  = len(pages)
    n_events = len(events)
    n_cols   = n_pages * n_events
 
    # Auto-size figure — reserve a fixed strip on the right for the colorbar
    cell_w, cell_h = 0.68, 0.72
    cbar_gap   = 0.18          # inches gap between heatmap and colorbar
    cbar_strip = 0.55          # inches for colorbar bar + tick labels + axis label
    fig_w = 2.2 + n_cols * cell_w + (n_pages - 1) * 0.12 + cbar_gap + cbar_strip + 0.15
    fig_h = 1.8 + n_users * cell_h + 1.8
 
    fig = plt.figure(figsize=(fig_w, fig_h))
 
    # Heatmap axes — occupies everything left of the reserved colorbar strip
    hm_left   = 1.8 / fig_w
    hm_bottom = 1.8 / fig_h
    hm_width  = (n_cols * cell_w + (n_pages - 1) * 0.12) / fig_w
    hm_height = (n_users * cell_h) / fig_h
    ax = fig.add_axes([hm_left, hm_bottom, hm_width, hm_height])
 
    # Dedicated colorbar axes — same height as heatmap, fixed position to the right
    cbar_left = hm_left + hm_width + cbar_gap / fig_w
    cbar_w    = (cbar_strip * 0.30) / fig_w
    cax = fig.add_axes([cbar_left, hm_bottom, cbar_w, hm_height])
 
    vals = matrix.values
    cmap = make_colormap()
    im = ax.imshow(vals, aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=0.35, interpolation="nearest")
 
    # ── cell value labels ─────────────────────────────────────────────────────
    for r in range(n_users):
        for c in range(n_cols):
            v = vals[r, c]
            text_col = "white" if v > 0.22 else ("#1a1a1a" if v >= 0.10 else "#4a4a4a")
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=12, color=text_col)
 
    # ── y-axis: user labels ───────────────────────────────────────────────────
    ax.set_yticks(range(n_users))
    ax.set_yticklabels(matrix.index, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=6)
 
    # ── x-axis: event labels (bottom) ────────────────────────────────────────
    short = {
        "page.1": "page", "app_event": "app_event",
    }
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [short.get(events[c % n_events], events[c % n_events]) for c in range(n_cols)],
        rotation=38, ha="right",
    )
    ax.tick_params(axis="x", length=0, pad=3)
 
    # ── page group labels (secondary x-axis at top) ───────────────────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    page_mid_ticks = [pi * n_events + (n_events - 1) / 2 for pi in range(n_pages)]
    short_page_names = [p.replace("-v", "\nv").replace("subscribe", "sub") for p in pages]
    ax2.set_xticks(page_mid_ticks)
    ax2.set_xticklabels(short_page_names, fontweight="bold")
    ax2.tick_params(length=0, pad=6)
    ax2.spines["top"].set_visible(False)
 
    # ── vertical dividers between page groups ─────────────────────────────────
    for pi in range(1, n_pages):
        ax.axvline(pi * n_events - 0.5, color="white", linewidth=3)
 
    # ── minor grid for cell borders ───────────────────────────────────────────
    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_users)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
 
    # ── per-page variance footnote (bottom of each column group) ─────────────
    # for pi, page in enumerate(pages):
    #     mid_col = pi * n_events + (n_events - 1) / 2
    #     ax.text(mid_col, n_users - 0.42,
    #             f"σ²={page_variances[page]:.4f}",
    #             ha="center", va="top", fontsize=6,
    #             color="#888780", transform=ax.transData)
 
    # ── colorbar — drawn into its own axes so it never overlaps the heatmap ──
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Event proportion", labelpad=8)
    cbar.ax.tick_params(labelsize=12)
    cbar.locator   = mticker.MultipleLocator(0.05)
    cbar.formatter = mticker.FormatStrFormatter("%.2f")
    cbar.update_ticks()
 
    # ── title ─────────────────────────────────────────────────────────────────
    # ax.set_title(
    #     "User × page × event heatmap  —  event proportion (auto-selected by cross-user variance)",
    #     fontsize=10, fontweight="500", pad=38, loc="left",
    # )
 
    # ── event key legend (bottom-left footnote) ───────────────────────────────
    legend_entries = [f"{short.get(e, e)} = {e}" for e in events if e in short]
    if legend_entries:
        fig.text(0.01, 0.005, "  ·  ".join(legend_entries),
                 color="#888780", va="bottom")
 
    # tight_layout is intentionally omitted — axes positions are set manually above
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved → {output_path}")
    plt.show()

 
# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_DIR, FILE_PATTERN, N_USERS, ALL_EVENT_COLS)
 
    sel_events, ev_var = select_noticeable_events(
        df, ALL_EVENT_COLS, DROP_EVENTS, EVENT_VAR_THRESHOLD)
    sel_pages, pg_var  = select_noticeable_pages(
        df, sel_events, PAGE_VAR_THRESHOLD)
 
    print(f"Selected events ({len(sel_events)}): {sel_events}")
    print(f"Selected pages  ({len(sel_pages)}):  {sel_pages}")
 
    # users  = [f"User {i}" for i in range(1, N_USERS + 1)]
    matrix = build_heatmap_matrix(df, LABELS, sel_pages, sel_events)
 
    draw_heatmap(matrix, sel_pages, sel_events, ev_var, pg_var, OUTPUT_PATH)