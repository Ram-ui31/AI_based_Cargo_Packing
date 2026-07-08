"""End-to-end evaluation using the proven prior-art TransformerClusterer
(Desktop/clustering_v2/transformer_rl_v2_K.pt) for assignment, combined with
OUR trained Phase A placement policy for the actual geometric packing.

Each test instance is evaluated once, at its own real (fixed) K, from
good_data/synthetic_test/metadata_with_K.csv.
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "model_b", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "rl_packer", "src"))

import numpy as np
import pandas as pd
import torch

from assignment_env import evaluate_assignment
from data import load_split
from old_clusterer_bridge import load_old_clusterer, greedy_decode
from placement_policy import PlacementPolicy

PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")


def run(n_test_instances: int | None = None):
    test_instances = load_split("test")
    if n_test_instances:
        test_instances = test_instances[:n_test_instances]

    placement_policy = PlacementPolicy(hidden=96)
    placement_policy.load_state_dict(torch.load(PLACEMENT_CKPT))
    placement_policy.eval()

    old_clusterer = load_old_clusterer()

    rows = []
    for inst in test_instances:
        assignment = greedy_decode(old_clusterer, inst.packages, inst.ulds)
        result = evaluate_assignment(inst, assignment, placement_policy)
        rows.append(dict(
            instance=inst.instance_id, K=inst.K, cost=result["cost"], spread=result["spread"],
            delay_cost=result["delay_cost"], n_left_behind=len(result["left_behind"]),
            n_packages=len(inst.packages), n_priority=result["n_priority"],
            priority_dropped=len(result["priority_dropped"]),
            mean_uld_utilization=float(np.mean(result["utilization"])),
            n_none_by_clusterer=result["n_none_by_clusterer"],
            n_dropped_by_packer=result["n_dropped_by_packer"],
            n_assigned_by_clusterer=result["n_assigned_by_clusterer"],
        ))

    df = pd.DataFrame(rows)
    df["frac_left_behind"] = df["n_left_behind"] / df["n_packages"]
    df["frac_none_by_clusterer"] = df["n_none_by_clusterer"] / df["n_packages"]
    df["frac_dropped_by_packer"] = df["n_dropped_by_packer"] / df["n_packages"]
    summary = df.groupby("K")[["cost", "spread", "delay_cost", "frac_left_behind", "mean_uld_utilization"]].mean()
    print(summary)
    print("\noverall mean fraction left behind:", df["frac_left_behind"].mean())
    print("  of which, left behind because clusterer said NONE:", df["frac_none_by_clusterer"].mean())
    print("  of which, assigned to a ULD but packer couldn't fit it:", df["frac_dropped_by_packer"].mean())
    print("total priority packages dropped across all runs:", df["priority_dropped"].sum())
    out_path = os.path.join(_THIS_DIR, "..", "results", "eval_results_hybrid.csv")
    df.to_csv(out_path, index=False)
    print(f"\nfull per-instance results (incl. each instance's K) saved to {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-test-instances", type=int, default=None)
    args = p.parse_args()
    run(n_test_instances=args.n_test_instances)
