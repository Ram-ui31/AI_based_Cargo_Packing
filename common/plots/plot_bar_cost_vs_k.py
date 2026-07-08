"""Histogram/bar-chart form: one panel per K, each zoomed to its own range,
with bars for RL / py3dbp / Greedy / IL baseline. Reads the already-computed
compare_packers_final.csv and il_baseline_costs.csv.
"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def plot(csv_path: str = os.path.join(RESULTS_DIR, "compare_packers_final.csv"),
         il_csv_path: str = os.path.join(RESULTS_DIR, "il_baseline_costs.csv"),
         out_path: str = os.path.join(RESULTS_DIR, "cost_vs_k_bar.png")):
    df = pd.read_csv(csv_path)
    il_df = pd.read_csv(il_csv_path)

    summary = df.groupby("K")[["rl_cost", "py3d_cost", "greedy_cost"]].mean().sort_index()
    summary["il_cost"] = il_df.groupby("K")["il_cost"].mean()
    k_values = summary.index.to_numpy()

    methods = [
        ("rl_cost", "RL", "#1f77b4"),
        ("py3d_cost", "py3dbp packer", "#d62728"),
        ("greedy_cost", "Greedy packer", "#2ca02c"),
        ("il_cost", "IL baseline (pre-RL)", "#9467bd"),
    ]
    labels = [lbl for _, lbl, _ in methods]
    colors = [c for _, _, c in methods]

    fig, axes = plt.subplots(1, len(k_values), figsize=(3.6 * len(k_values), 5.5), sharey=False)
    for ax, k in zip(axes, k_values):
        costs = [summary.loc[k, col] for col, _, _ in methods]
        bars = ax.bar(labels, costs, color=colors, width=0.65)

        lo, hi = min(costs), max(costs)
        pad = max((hi - lo) * 0.6, hi * 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1250))

        for bar, cost in zip(bars, costs):
            ax.annotate(f"{cost:,.0f}", (bar.get_x() + bar.get_width() / 2, cost),
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(f"K = {int(k)}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean total cost")
    fig.suptitle("Cost by packer, per K -- same assignment (synthetic_test), each panel zoomed to its own range")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    plot()
