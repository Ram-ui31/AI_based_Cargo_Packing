"""End-to-end evaluation: assignment policy (Phase B, greedy) + frozen
placement policy (Phase A, greedy) on synthetic_test, cost broken down by K.
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

from assignment_env import sample_assignment, evaluate_assignment
from assignment_policy import AssignmentPolicy
from data import load_split
from placement_policy import PlacementPolicy

ASSIGNMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "model_b", "assignment_policy.pt")
PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")


def run(n_test_instances: int | None = None):
    """Evaluate each test instance once, at its own real (fixed) K -- not
    every instance re-tried at all 5 K values, which would just be testing
    on a hypothetical K the instance doesn't actually carry."""
    test_instances = load_split("test")
    if n_test_instances:
        test_instances = test_instances[:n_test_instances]

    placement_policy = PlacementPolicy(hidden=96)
    placement_policy.load_state_dict(torch.load(PLACEMENT_CKPT))
    placement_policy.eval()

    assignment_policy = AssignmentPolicy()
    assignment_policy.load_state_dict(torch.load(ASSIGNMENT_CKPT))
    assignment_policy.eval()

    rows = []
    for inst in test_instances:
        assignment = sample_assignment(assignment_policy, inst, greedy=True)
        result = evaluate_assignment(inst, assignment, placement_policy)
        rows.append(dict(
            instance=inst.instance_id, K=inst.K, cost=result["cost"], spread=result["spread"],
            delay_cost=result["delay_cost"], n_left_behind=len(result["left_behind"]),
            priority_dropped=len(result["priority_dropped"]), n_priority=result["n_priority"],
            mean_uld_utilization=float(np.mean(result["utilization"])),
        ))

    df = pd.DataFrame(rows)
    summary = df.groupby("K")["cost"].agg(["mean", "std", "count"])
    print(summary)
    print("\ntotal priority packages dropped across all runs:", df["priority_dropped"].sum())
    out_path = os.path.join(_THIS_DIR, "..", "results", "eval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nfull per-instance results saved to {out_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-test-instances", type=int, default=None)
    args = p.parse_args()
    run(n_test_instances=args.n_test_instances)
