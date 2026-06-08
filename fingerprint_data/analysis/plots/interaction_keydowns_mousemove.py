import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from adjustText import adjust_text

# --- Config ---
DATA_DIR = "."  # folder containing the 6 CSV files
agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

N_USERS = 6

plt.rcParams.update({
    "font.size": 20,          # base font size
    # "axes.titlesize": 14,     # title size
    # "axes.labelsize": 20,     # x/y label size
    # "xtick.labelsize": 16,    # x tick labels
    # "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 16,    # legend
    # "figure.titlesize": 16    # figure title
})

COLORS = ["#378ADD", "#BA7517", "#1D9E75", "#D85A30", "#7F77DD", "#D4537E"]
agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}
LABELS = list(agent_labels.values())

# --- Load data ---
dfs = []
for i, a in enumerate(agents):
    path = os.path.join(DATA_DIR, f"analysis/{a}/interaction_session_summary.csv")
    df = pd.read_csv(path)
    df["user"] = agent_labels[a]
    df["user_id"] = i
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 6))

texts = []

for i, (user_label, color) in enumerate(zip(LABELS, COLORS)):
    subset = data[data["user_id"] == i]

    ax.scatter(
        subset["n_keydowns"],
        subset["n_mousemoves"],
        label=user_label,
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=70,
    )

    cx = subset["n_keydowns"].mean()
    cy = subset["n_mousemoves"].mean()

    text = ax.annotate(
        user_label,
        xy=(cx, cy),
        fontsize=16,
        fontweight="bold",
        color=color,
    )

    texts.append(text)

adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.0, mutation_scale=12),
)

ax.set_xlabel("Keydowns/Session")
ax.set_ylabel("Mouse moves/Session")
# ax.set_title("User separation: keydowns vs mouse moves\n(each point = one session)", fontsize=13)

legend_patches = [
    mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)
]
ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_facecolor("#f9f9f9")
fig.tight_layout()

output_path = "analysis/plots/plot_keydowns_vs_mousemoves.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
# plt.show()