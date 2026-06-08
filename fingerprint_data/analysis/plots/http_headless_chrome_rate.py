import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,          # base font size
    # "axes.titlesize": 14,     # title size
    # "axes.labelsize": 20,     # x/y label size
    # "xtick.labelsize": 16,    # x tick labels
    # "ytick.labelsize": 16,    # y tick labels
    # "legend.fontsize": 16,    # legend
    # "figure.titlesize": 16    # figure title
})
# plt.rcParams.update({
#     "text.usetex": False,
#     "mathtext.default": "regular",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

agents = ["autogen_websurfer", "browser_use", "claude_computer_use", "gemini_computer_use", "operator", "skyvern"]
files  = [f"analysis/{agent}/http_header_summary.csv" for agent in agents]

agent_labels = {
    "autogen_websurfer": "AutoGen",
    "browser_use": "Browser Use",
    "claude_computer_use": "Claude",
    "gemini_computer_use": "Gemini",
    "operator": "Operator",
    "skyvern": "Skyvern"
}
def load_all(path: str) -> pd.DataFrame:
    frames = []
    # for i, p in enumerate(path, 1):
    df = pd.read_csv(path)
    # df["agent"] = i
    frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    mask = (combined["method"] == "POST") & (combined["path"] == "/collect")
    return combined[~mask].copy()
 
# df = load_all(files)

all_rows = []
for file in files:
    ct_page = load_all(file)

    if "ch_is_headless" in ct_page.columns:
        df = ct_page[[
            "trace",
            "ch_is_headless"
        ]].copy()

        headless_rates = (
            df.groupby("trace")["ch_is_headless"]
            .mean()
            .tolist()
        )

        # Add condition/agent name
        agent = file.split('/')[1]
        # df["headless_rates"] = np.mean(headless_rates)

        # Keep only needed columns
        all_rows.append({
            "agent": agent,
            "agent_label": agent_labels[agent],
            "headless_rates": np.mean(headless_rates)
        })

final_df = pd.DataFrame(all_rows)

colors = {
    "autogen_websurfer": "#2563EB",
    "browser_use": "#0D9488",
    "claude_computer_use": "#D97706",
    "gemini_computer_use": "#DC2626",
    "operator": "#7C3AED",
    "skyvern": "#16A34A",
}

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = [
    colors[agent]
    for agent in final_df["agent"]
]
bars = plt.bar(final_df["agent_label"], final_df["headless_rates"], color=bar_colors)

# plt.bar_label(
#     bars,
#     fmt="%.2f",
#     padding=3
# )
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), #+0.01 for margin
             "{:.2f}".format(bar.get_height()), ha='center', va='bottom')

plt.xlabel("Agents")
plt.ylabel("Headless Chrome Rate")
plt.ylim([0, 1.06])
# plt.xticks(rotation=20)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
# plt.legend(ncols=2)

plt.tight_layout()

plt.savefig(
    "analysis/plots/plot_headless_chrome_rate.pdf",
    dpi=300,
    bbox_inches="tight"
)
