import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 27,          # base font size
    # "axes.titlesize": 14,     # title size
    # "axes.labelsize": 20,     # x/y label size
    # "xtick.labelsize": 16,    # x tick labels
    # "ytick.labelsize": 16,    # y tick labels
    # "legend.fontsize": 16,    # legend
    # "figure.titlesize": 16    # figure title
})

agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]

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
    ct_page = pd.read_csv(f'analysis/{agent}/interaction_session_summary.csv')

    if "iei_std_ms" in ct_page.columns:
        df = ct_page[[
            "trace",
            "iei_mean_ms",
            "iei_std_ms"
        ]].copy()

        iei_mean_avg = df["iei_mean_ms"].mean()
        iei_std_avg = df["iei_std_ms"].mean()

        # Add condition/agent name
        df["agent"] = agent
        df["iei_mean_avg"] = iei_mean_avg
        df["iei_std_avg"] = iei_std_avg

        # Keep only needed columns
        df = df[[
            "agent",
            "iei_mean_avg",
            "iei_std_avg"
        ]]

        all_rows.append(df)

final_df = pd.concat(all_rows, ignore_index=True)
final_df["agent_label"] = final_df["agent"].map(agent_labels)
final_df["iei_mean_s"] = final_df["iei_mean_avg"] / 1000
final_df["iei_std_s"] = final_df["iei_std_avg"] / 1000

colors = {
    "autogen_websurfer": "#2563EB",
    "browser_use": "#0D9488",
    "claude_computer_use": "#D97706",
    "gemini_computer_use": "#DC2626",
    "operator": "#7C3AED",
    "skyvern": "#16A34A",
}

fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(final_df["agent_label"], final_df["iei_mean_s"], marker='o', label="Mean")
plt.plot(final_df["agent_label"], final_df["iei_std_s"], marker='o', label="Std")

plt.xlabel("Agents")
plt.ylabel("Inter-Event-Intervals (s)")
plt.xticks(rotation=45)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
plt.legend(ncols=2)

plt.tight_layout()

plt.savefig(
    "analysis/plots/plot_iei_mean_std.pdf",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()