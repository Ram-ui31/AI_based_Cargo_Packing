"""X/Y line graph: K on the x-axis, mean cost on the y-axis, comparing
clusterer A (prior-art, spread_loss-trained but K-blind at inference) vs
clusterer B (fresh, K-aware at inference, hinge-loss-trained) -- same
Phase A packer for both, isolating the assignment stage's effect."""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def plot(csv_path: str = os.path.join(RESULTS_DIR, "new_vs_old_clusterer_costs.csv"),
         out_path: str = os.path.join(RESULTS_DIR, "new_vs_old_clusterer.png")):
    df = pd.read_csv(csv_path)
    summary = df.groupby("K")[["new_cost", "old_cost"]].mean().sort_index()
    k_values = summary.index.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(k_values, summary["new_cost"], marker="o", color="#1f77b4", linewidth=2,
            label="Clusterer B (new)")
    ax.plot(k_values, summary["old_cost"], marker="s", color="#ff7f0e", linewidth=2,
            label="Clusterer A (prior-art)")

    offsets = {"new_cost": (0, -16), "old_cost": (0, 8)}
    colors = {"new_cost": "#1f77b4", "old_cost": "#ff7f0e"}
    for col in ("new_cost", "old_cost"):
        for k, v in zip(k_values, summary[col]):
            ax.annotate(f"{v:,.0f}", (k, v), textcoords="offset points", xytext=offsets[col],
                        ha="center", fontsize=8, color=colors[col], fontweight="bold")

    ax.set_xlabel("K (spread cost coefficient)")
    ax.set_ylabel("Mean total cost")
    ax.set_title("Cost vs K -- Clusterer A (prior-art) vs Clusterer B (new), same Phase A packer")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(int(k)) for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    plot()
