import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({
    "font.size": 20,          # base font size
    # "axes.titlesize": 14,     # title size
    "axes.labelsize": 24,     # x/y label size
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
    ct_page = pd.read_csv(f'analysis/{agent}/trace_summary.csv')

    if "req_rate_hz" in ct_page.columns:
        df = ct_page[[
            "trace",
            "req_rate_hz",
        ]].copy()

        # Add agent name
        df["agent"] = agent

        all_rows.append(df)

# Combine all agents
final_df = pd.concat(all_rows, ignore_index=True)

# Create sequential index within each agent
final_df["idx"] = final_df.groupby("agent").cumcount() + 1

# Pivot to wide format
final_df = final_df.pivot(
    index="agent",
    columns="idx",
    values="req_rate_hz"
)

# Rename columns
final_df.columns = [
    f"req_rate_hz_{i}" for i in final_df.columns
]

# Convert index back to column
final_df = final_df.reset_index()

final_df["agent"] = final_df["agent"].map(agent_labels)

value_cols = [c for c in final_df.columns if c.startswith("req_rate_hz")]

cv_dict = {}

for _, row in final_df.iterrows():
    values = row[value_cols].astype(float).values

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    cv = std / mean
    cv_dict[row["agent"]] = cv

# Sort by CV (optional)
order = sorted(cv_dict, key=cv_dict.get, reverse=True)

colors = {
    "AutoGen": "#0072B2",
    "Browser Use": "#E69F00",
    "Claude": "#009E3F",
    "Gemini": "#CC79A7",
    "Operator": "#D52300",
    "Skyvern": "#56D5E9",
}

fig, ax1 = plt.subplots(figsize=(12, 6))
# plt.bar(final_df["agent_label"], final_df["req_rate_avg"])
######## Draw box plot with mean ########
long_df = final_df.melt(
    id_vars="agent",
    value_vars=value_cols,
    var_name="request",
    value_name="req_rate_hz"
)

# Plot
sns.boxplot(
    data=long_df,
    x="agent",
    y="req_rate_hz",
    order=order,
    palette=colors,
    ax=ax1
)

# Optional: show individual points
sns.stripplot(
    data=long_df,
    x="agent",
    y="req_rate_hz",
    order=order,
    color="black",
    alpha=0.5,
    size=3,
    ax=ax1
)
ax1.set_ylabel("Request Rate (Hz)")
ax1.set_xlabel("Agents")
ax1.set_ylim([0, 1])

ax2 = ax1.twinx()

cv_values = [cv_dict[a] for a in order]

ax2.plot(
    range(len(order)),
    cv_values,
    color="black",
    marker="o",
    linestyle="--",
    linewidth=1.5,
    label="CV"
)

ax2.set_ylabel("Coefficient of Variation (CV)")
ax2.grid(False)
ax2.legend(loc="upper right")


# plt.xlabel("Agents")

# plt.xticks(rotation=20)

######## coefficient of variation is actually more interesting than the rate itself, add CV ########


# plt.xticks(rotation=45)
ax1.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
# plt.legend(ncols=2)

plt.tight_layout()

plt.savefig(
    "analysis/plots/plot_req_rate.pdf",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()