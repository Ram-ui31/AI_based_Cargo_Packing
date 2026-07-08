"""Standard X/Y line graph: K on the x-axis, mean cost on the y-axis, one
colored line per packer (RL / py3dbp / Greedy / IL baseline). Reads the
already-computed compare_packers_final.csv and il_baseline_costs.csv rather
than re-running the full evaluation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def plot(csv_path: str = os.path.join(RESULTS_DIR, "compare_packers_final.csv"),
         il_csv_path: str = os.path.join(RESULTS_DIR, "il_baseline_costs.csv"),
         out_path: str = os.path.join(RESULTS_DIR, "cost_vs_k_line.png")):
    df = pd.read_csv(csv_path)
    il_df = pd.read_csv(il_csv_path)

    summary = df.groupby("K")[["rl_cost", "py3d_cost", "greedy_cost"]].mean().sort_index()
    summary["il_cost"] = il_df.groupby("K")["il_cost"].mean()
    k_values = summary.index.to_numpy()

    series = [
        ("rl_cost", "RL", "#1f77b4", "o"),
        ("py3d_cost", "py3dbp packer", "#d62728", "s"),
        ("greedy_cost", "Greedy packer", "#2ca02c", "^"),
        ("il_cost", "IL baseline (pre-RL)", "#9467bd", "D"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for col, label, color, marker in series:
        ax.plot(k_values, summary[col], marker=marker, color=color, linewidth=2, label=label)

    # stagger label offsets per series so the 4 close-together values at each K don't overlap
    offsets = {"rl_cost": (0, -16), "py3d_cost": (0, 8), "greedy_cost": (30, 8), "il_cost": (30, -16)}
    for col, _, color, _ in series:
        for k, v in zip(k_values, summary[col]):
            ax.annotate(f"{v:,.0f}", (k, v), textcoords="offset points", xytext=offsets[col],
                        ha="center", fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel("K (spread cost coefficient)")
    ax.set_ylabel("Mean total cost")
    ax.set_title("Cost vs K -- same assignment, different packers (synthetic_test)")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(int(k)) for k in k_values])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1250))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    plot()
