import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 26,          # base font size
    # "axes.titlesize": 14,     # title size
    "axes.labelsize": 20,     # x/y label size
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 16,    # legend
    # "figure.titlesize": 16    # figure title
})

agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

scenario_map = {
    "S1-subscribe-v1": "S1-v1",
    "S2-subscribe-v2": "S1-v2",
    "S3-subscribe-v3": "S1-v3",
    "S4-scroll-gate":  "S2",
    "S5-hover-reveal": "S3",
    "S6-dom-mismatch": "S4",
    "Delayed Feedback — UX Behavior": "S5"
}
scenario_order = [
    "S1-v1",
    "S1-v2",
    "S1-v3",
    "S2",
    "S3",
    "S4",
    "S5"
]
agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}

all_rows = []
for agent in agents:
    ct_page = pd.read_csv(f'analysis/{agent}/interaction_page_summary.csv')

    if "first_interact_ms_mean" in ct_page.columns:
        df = ct_page[[
            "page",
            "first_interact_ms_mean",
            "first_interact_ms_std"
        ]].copy()

        df["first_interact_ms_mean"] /= 1000.0
        df["first_interact_ms_std"] /= 1000.0

        # Rename scenarios
        df["page"] = (
            df["page"]
            .astype(str)
            .str.strip()
        )

        df["page"] = df["page"].replace(scenario_map)

        # Add condition/agent name
        df["agent"] = agent

        # Keep only needed columns
        df = df[[
            "agent",
            "page",
            "first_interact_ms_mean",
            "first_interact_ms_std"
        ]]

        all_rows.append(df)

final_df = pd.concat(all_rows, ignore_index=True)

colors = {
    "autogen_websurfer": "#2563EB",
    "browser_use": "#0D9488",
    "claude_computer_use": "#D97706",
    "gemini_computer_use": "#DC2626",
    "operator": "#7C3AED",
    "skyvern": "#16A34A",
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(scenario_order))

for agent in agents:

    sub = (
        final_df[final_df["agent"] == agent]
        .groupby("page", as_index=True)
        .agg({
            "first_interact_ms_mean": "mean",
            "first_interact_ms_std": "mean"
        })
        .reindex(scenario_order)
    )

    ax.errorbar(
        x,
        sub["first_interact_ms_mean"],
        yerr=sub["first_interact_ms_std"],
        label=agent_labels.get(agent, agent),
        marker='o',
        linewidth=2,
        markersize=6,
        capsize=4,
        color=colors.get(agent, None)
    )

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(scenario_order)

ax.set_xlabel("Page")
ax.set_ylabel("Latency (s)")
# ax.set_title("Mouse trajectory across pages")

# Cleaner style
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
ax.legend(ncol=2, loc="upper right")

plt.tight_layout()

plt.savefig(
    "analysis/plots/plot_first_interaction_latency.pdf",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()