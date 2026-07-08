"""Single bar chart: one bar per packer, showing the average cost across
all 5 K's (mean of the 5 per-K means shown in cost_vs_k_bar.png)."""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def plot(csv_path: str = os.path.join(RESULTS_DIR, "compare_packers_final.csv"),
         il_csv_path: str = os.path.join(RESULTS_DIR, "il_baseline_costs.csv"),
         out_path: str = os.path.join(RESULTS_DIR, "cost_overall_bar.png")):
    df = pd.read_csv(csv_path)
    il_df = pd.read_csv(il_csv_path)

    per_k = df.groupby("K")[["rl_cost", "py3d_cost", "greedy_cost"]].mean()
    per_k["il_cost"] = il_df.groupby("K")["il_cost"].mean()
    overall = per_k.mean()  # average of the 5 per-K means

    methods = [
        ("rl_cost", "RL", "#1f77b4"),
        ("py3d_cost", "py3dbp packer", "#d62728"),
        ("greedy_cost", "Greedy packer", "#2ca02c"),
        ("il_cost", "IL baseline (pre-RL)", "#9467bd"),
    ]
    labels = [lbl for _, lbl, _ in methods]
    colors = [c for _, _, c in methods]
    costs = [overall[col] for col, _, _ in methods]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(labels, costs, color=colors, width=0.6)

    lo, hi = min(costs), max(costs)
    pad = max((hi - lo) * 0.5, hi * 0.02)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(250))

    for bar, cost in zip(bars, costs):
        ax.annotate(f"{cost:,.0f}", (bar.get_x() + bar.get_width() / 2, cost),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean total cost (averaged across K = 100, 500, 1000, 3000, 5000)")
    ax.set_title("Overall average cost by packer -- same assignment (synthetic_test)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    plot()
