"""Final cost comparison: same assignment (from the prior-art
TransformerClusterer, transformer_rl_v2_K.pt) on all 100 synthetic_test
instances (5 K groups of 20, from metadata_with_K.csv), packed with three
different packers -- our trained RL policy, py3dbp (priority-first), and a
largest-volume-first greedy heuristic -- isolating the effect of the packer
alone, holding the assignment fixed. Also plots mean cost vs K for all three.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "model_b", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "rl_packer", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from assignment_env import evaluate_assignment
from baseline_greedy import GreedyPolicy
from data import load_split
from old_clusterer_bridge import load_old_clusterer, greedy_decode
from placement_policy import PlacementPolicy
from py3dbp_assignment_eval import evaluate_assignment_py3dbp

PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")


def run():
    test_instances = load_split("test")

    clusterer = load_old_clusterer()

    placement_policy = PlacementPolicy(hidden=96)
    placement_policy.load_state_dict(torch.load(PLACEMENT_CKPT))
    placement_policy.eval()

    greedy_policy = GreedyPolicy()

    rows = []
    for inst in test_instances:
        assignment = greedy_decode(clusterer, inst.packages, inst.ulds)

        rl_result = evaluate_assignment(inst, assignment, placement_policy)
        py3d_result = evaluate_assignment_py3dbp(inst, assignment)
        greedy_result = evaluate_assignment(inst, assignment, greedy_policy)

        rows.append(dict(
            instance=inst.instance_id, K=inst.K,
            rl_cost=rl_result["cost"], rl_spread=rl_result["spread"],
            rl_delay=rl_result["delay_cost"], rl_prio_dropped=len(rl_result["priority_dropped"]),
            rl_mean_util=float(np.mean(rl_result["utilization"])),
            py3d_cost=py3d_result["cost"], py3d_spread=py3d_result["spread"],
            py3d_delay=py3d_result["delay_cost"], py3d_prio_dropped=len(py3d_result["priority_dropped"]),
            py3d_mean_util=float(np.mean(py3d_result["utilization"])),
            greedy_cost=greedy_result["cost"], greedy_spread=greedy_result["spread"],
            greedy_delay=greedy_result["delay_cost"], greedy_prio_dropped=len(greedy_result["priority_dropped"]),
            greedy_mean_util=float(np.mean(greedy_result["utilization"])),
        ))

    df = pd.DataFrame(rows)
    summary = df.groupby("K")[["rl_cost", "py3d_cost", "greedy_cost",
                                "rl_mean_util", "py3d_mean_util", "greedy_mean_util"]].mean()
    print(summary)
    print()
    print("overall mean cost -- RL packer:     ", df["rl_cost"].mean())
    print("overall mean cost -- py3dbp packer:  ", df["py3d_cost"].mean())
    print("overall mean cost -- greedy packer:  ", df["greedy_cost"].mean())
    print("total priority dropped -- RL:", df["rl_prio_dropped"].sum(),
          " py3dbp:", df["py3d_prio_dropped"].sum(), " greedy:", df["greedy_prio_dropped"].sum())

    out_path = os.path.join(_THIS_DIR, "..", "results", "compare_packers_final.csv")
    df.to_csv(out_path, index=False)
    print(f"\nfull per-instance results saved to {out_path}")

    plot_cost_vs_k(summary)
    return df


def plot_cost_vs_k(summary: pd.DataFrame,
                    out_path: str = os.path.join(_THIS_DIR, "..", "results", "cost_vs_k.png")):
    """One panel per K, each auto-zoomed to its own cost range -- a single
    shared y-axis would let the K=5000 scale (~32k) visually flatten the
    smaller, but still meaningful, differences at K=100 (~15k)."""
    k_values = summary.index.to_numpy()
    methods = [("rl_cost", "RL packer (ours)", "#1f77b4"),
               ("py3d_cost", "py3dbp packer", "#d62728"),
               ("greedy_cost", "Greedy packer", "#2ca02c")]

    fig, axes = plt.subplots(1, len(k_values), figsize=(3.1 * len(k_values), 5), sharey=False)
    for ax, k in zip(axes, k_values):
        costs = [summary.loc[k, col] for col, _, _ in methods]
        colors = [c for _, _, c in methods]
        labels = [lbl for _, lbl, _ in methods]
        bars = ax.bar(labels, costs, color=colors, width=0.6)

        lo, hi = min(costs), max(costs)
        pad = max((hi - lo) * 0.6, hi * 0.01)
        ax.set_ylim(lo - pad, hi + pad)

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
    print(f"\nplot saved to {out_path}")


if __name__ == "__main__":
    run()
