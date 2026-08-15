from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    ROOT
    / "experiments"
    / "v4_cost_response"
    / "analysis"
    / "cost_response_summary.csv"
)

OUTPUT_PATH = (
    ROOT
    / "paper"
    / "figures"
    / "v4_cost_response_main.png"
)


df = pd.read_csv(SUMMARY_PATH)

# Avoid floating-point display artifacts such as 9.9999999998.
df["penalty_percent"] = df["penalty_percent"].round().astype(int)

fig, ax = plt.subplots(figsize=(9.2, 5.2))


def plot_context(context, label, marker, linestyle):
    data = (
        df[df["context"] == context]
        .sort_values("penalty_percent")
        .copy()
    )

    x = data["penalty_percent"].to_numpy()
    y = data["mean_x"].to_numpy()
    low = data["ci_95_low"].to_numpy()
    high = data["ci_95_high"].to_numpy()

    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.0,
        markersize=6,
        label=label,
    )

    ax.fill_between(
        x,
        low,
        high,
        alpha=0.10,
    )


# Generalized clean and modified are exactly identical at every tested
# level, so one displayed curve represents both without hiding another
# curve underneath it.
plot_context(
    "generalized_clean",
    "Generalized (clean = modified)",
    "o",
    "-",
)

plot_context(
    "cue_bound_target_modified",
    "Cue-bound: target present",
    "s",
    "-",
)

plot_context(
    "cue_bound_target_clean",
    "Cue-bound: target absent",
    "^",
    "--",
)

plot_context(
    "neutral_clean",
    "Neutral: clean",
    "D",
    ":",
)

plot_context(
    "neutral_modified",
    "Neutral: modified",
    "v",
    "-.",
)


# Descriptive majority-priority threshold used in the switching analysis.
ax.axhline(
    50,
    linewidth=1.2,
    linestyle="--",
    alpha=0.6,
)

ax.text(
    79,
    52,
    "50-point threshold",
    ha="right",
    va="bottom",
    fontsize=9,
)

ax.set_xlabel("Efficiency penalty for Organization X (%)")
ax.set_ylabel("Mean allocation to Organization X")
ax.set_xticks([0, 10, 20, 40, 60, 80])
ax.set_xlim(-2, 82)
ax.set_ylim(-3, 105)

ax.legend(
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
)

ax.grid(axis="y", alpha=0.18)

fig.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved publication figure to: {OUTPUT_PATH}")