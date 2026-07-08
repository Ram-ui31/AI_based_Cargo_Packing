"""Regenerate new_vs_old_clusterer_costs.csv comparing:
  - Clusterer B (new): our from-scratch assignment policy (assignment_policy.pt),
    K-aware at inference, hinge-loss + soft-spread-loss trained.
  - Clusterer A (prior-art): the original TransformerClusterer, now retrained
    with a real differentiable spread_loss added to its own training pipeline
    (transformer_rl_v2_K_spreadloss.pt), same architecture as before.

Both use OUR frozen Phase A placement policy for actual geometric packing, so
the comparison isolates assignment quality. Each test instance evaluated once
at its own real (fixed) K from good_data/synthetic_test/metadata_with_K.csv.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "model_b", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "rl_packer", "src"))

import numpy as np
import pandas as pd
import torch

from assignment_env import evaluate_assignment, sample_assignment
from assignment_policy import AssignmentPolicy
from data import load_split
from old_clusterer_bridge import load_old_clusterer, greedy_decode
from placement_policy import PlacementPolicy

PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")
NEW_CLUSTERER_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "model_b", "assignment_policy.pt")
OLD_CLUSTERER_CKPT = os.path.expanduser(
    "~/Desktop/clustering_v2/transformer_rl_v2_K_spreadloss.pt")


def run(n_test_instances: int | None = None):
    test_instances = load_split("test")
    if n_test_instances:
        test_instances = test_instances[:n_test_instances]

    placement_policy = PlacementPolicy(hidden=96)
    placement_policy.load_state_dict(torch.load(PLACEMENT_CKPT))
    placement_policy.eval()

    new_policy = AssignmentPolicy()
    new_policy.load_state_dict(torch.load(NEW_CLUSTERER_CKPT))
    new_policy.eval()

    old_clusterer = load_old_clusterer(checkpoint_path=OLD_CLUSTERER_CKPT)

    rows = []
    for inst in test_instances:
        new_assignment = sample_assignment(new_policy, inst, greedy=True)
        new_result = evaluate_assignment(inst, new_assignment, placement_policy)

        old_assignment = greedy_decode(old_clusterer, inst.packages, inst.ulds)
        old_result = evaluate_assignment(inst, old_assignment, placement_policy)

        rows.append(dict(
            instance=inst.instance_id, K=inst.K,
            new_cost=new_result["cost"], old_cost=old_result["cost"],
            new_spread=new_result["spread"], old_spread=old_result["spread"],
            new_delay=new_result["delay_cost"], old_delay=old_result["delay_cost"],
            new_priority_dropped=len(new_result["priority_dropped"]),
            old_priority_dropped=len(old_result["priority_dropped"]),
        ))

    df = pd.DataFrame(rows)
    summary = df.groupby("K")[["new_cost", "old_cost", "new_spread", "old_spread",
                                "new_delay", "old_delay"]].mean()
    print(summary)
    print(f"\noverall new (spread loss): {df['new_cost'].mean():.2f}  "
          f"overall old (spread loss retrain): {df['old_cost'].mean():.2f}")
    print(f"new wins: {(df['new_cost'] < df['old_cost']).sum()} / {len(df)}")
    print(f"priority dropped -- new: {df['new_priority_dropped'].sum()}  "
          f"old: {df['old_priority_dropped'].sum()}")

    out_path = os.path.join(_THIS_DIR, "..", "results", "new_vs_old_clusterer_costs.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved to {out_path}")
    return df


if __name__ == "__main__":
    run()
