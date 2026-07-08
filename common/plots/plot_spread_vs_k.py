"""X/Y line graph: K on the x-axis, mean priority spread (number of ULDs
used for priority packages) on the y-axis, comparing clusterer A (prior-art,
spread_loss-trained but K-blind at inference) vs clusterer B (fresh,
K-aware at inference, hinge-loss-trained)."""

from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def plot(csv_path: str = os.path.join(RESULTS_DIR, "new_vs_old_clusterer_costs.csv"),
         out_path: str = os.path.join(RESULTS_DIR, "spread_vs_k.png")):
    df = pd.read_csv(csv_path)
    summary = df.groupby("K")[["new_spread", "old_spread"]].mean().sort_index()
    k_values = summary.index.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(k_values, summary["new_spread"], marker="o", color="#1f77b4", linewidth=2,
            label="Clusterer B (new)")
    ax.plot(k_values, summary["old_spread"], marker="s", color="#ff7f0e", linewidth=2,
            label="Clusterer A (prior-art)")

    offsets = {"new_spread": (0, -16), "old_spread": (0, 8)}
    colors = {"new_spread": "#1f77b4", "old_spread": "#ff7f0e"}
    for col in ("new_spread", "old_spread"):
        for k, v in zip(k_values, summary[col]):
            ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=offsets[col],
                        ha="center", fontsize=8, color=colors[col], fontweight="bold")

    ax.set_xlabel("K (spread cost coefficient)")
    ax.set_ylabel("Mean priority spread (# ULDs used for priority)")
    ax.set_title("Spread vs K -- Clusterer A (prior-art) vs Clusterer B (new)")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(int(k)) for k in k_values])
    ax.set_ylim(0, max(summary["old_spread"].max(), summary["new_spread"].max()) + 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    plot()
