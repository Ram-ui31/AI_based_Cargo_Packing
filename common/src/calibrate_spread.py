"""Clean calibration of target-spread-vs-cost per K: load-balanced forced
distribution across exactly N target ULDs (not round-robin, which can
overflow and trigger confounding fallback logic), with a feasibility
pre-check so infeasible (spread, instance) combinations are skipped rather
than silently distorted by a force-fallback."""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "model_b", "src"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "rl_packer", "src"))

import numpy as np
import pandas as pd
import torch

from assignment_policy import dim_fits

PLACEMENT_CKPT = os.path.join(_THIS_DIR, "..", "..", "..", "uld_heightmap_rl", "checkpoints", "rl_packer", "placement_policy.pt")


def greedy_assign_balanced_spread(packages_df, ulds_df, target_spread: int):
    n_ulds = len(ulds_df)
    uld_rows = list(ulds_df.itertuples())
    uld_weight_limit = [float(u.Weight_Limit) for u in uld_rows]
    uld_volume = [int(u.Length * u.Width * u.Height) for u in uld_rows]
    running_weight = [0.0] * n_ulds
    running_volume = [0] * n_ulds

    order_by_vol = sorted(range(n_ulds), key=lambda i: -uld_volume[i])
    target_ulds = order_by_vol[:target_spread]

    pkgs = packages_df.copy()
    pkgs["volume"] = pkgs.Length * pkgs.Width * pkgs.Height
    prio = pkgs[pkgs.Type == "Priority"].sort_values("volume", ascending=False)
    econ = pkgs[pkgs.Type == "Economy"].sort_values("volume", ascending=False)

    assignment = {}
    feasible = True
    for row in prio.itertuples():
        # among target ULDs this fits in, pick the one with the LEAST relative
        # fill (weight/limit) -- load-balances across all N rather than
        # greedily filling one first, so actual spread matches target_spread
        # whenever capacity allows it at all.
        candidates = [
            i for i in target_ulds
            if dim_fits(row.Length, row.Width, row.Height, uld_rows[i].Length, uld_rows[i].Width, uld_rows[i].Height)
            and running_weight[i] + row.Weight <= uld_weight_limit[i]
            and running_volume[i] + row.volume <= uld_volume[i]
        ]
        if not candidates:
            feasible = False
            break
        i = min(candidates, key=lambda i: running_weight[i] / uld_weight_limit[i])
        running_weight[i] += row.Weight
        running_volume[i] += row.volume
        assignment[row.Package_ID] = i

    if not feasible:
        return None  # this (instance, target_spread) combo isn't achievable -- skip, don't distort

    def try_assign_econ(row):
        order = sorted(range(n_ulds), key=lambda i: -(uld_volume[i] - running_volume[i]))
        for i in order:
            u = uld_rows[i]
            if not dim_fits(row.Length, row.Width, row.Height, u.Length, u.Width, u.Height):
                continue
            if running_weight[i] + row.Weight > uld_weight_limit[i]:
                continue
            if running_volume[i] + row.volume > uld_volume[i]:
                continue
            running_weight[i] += row.Weight
            running_volume[i] += row.volume
            return i
        return None

    for row in econ.itertuples():
        assignment[row.Package_ID] = try_assign_econ(row)
    return assignment


def calibrate(n_instances: int = 40, k_values=(100, 500, 1000, 3000, 5000), max_spread_compared: int = 4):
    """Paired, within-instance comparison: for each instance, only compare
    spread values that are ALL feasible for that instance (up to
    max_spread_compared), so the average for spread=1 isn't silently
    computed over an easier subset than spread=2's average."""
    from data import load_split
    from assignment_env import evaluate_assignment
    from placement_policy import PlacementPolicy

    pp = PlacementPolicy(hidden=96)
    pp.load_state_dict(torch.load(PLACEMENT_CKPT))
    pp.eval()

    instances = load_split("train")[500:500 + n_instances]

    results = {}
    for K in k_values:
        by_spread = {s: [] for s in range(1, max_spread_compared + 1)}
        n_paired = 0
        for inst in instances:
            inst.K = K
            n_ulds = len(inst.ulds)
            max_s = min(max_spread_compared, n_ulds)
            per_instance_costs = {}
            for target_spread in range(1, max_s + 1):
                assignment = greedy_assign_balanced_spread(inst.packages, inst.ulds, target_spread)
                if assignment is None:
                    per_instance_costs = None  # this instance can't support all spreads up to max_s -- exclude entirely
                    break
                result = evaluate_assignment(inst, assignment, pp)
                per_instance_costs[target_spread] = result["cost"]
            if per_instance_costs is None:
                continue
            n_paired += 1
            for s, c in per_instance_costs.items():
                by_spread[s].append(c)

        avg_by_spread = {s: float(np.mean(c)) for s, c in by_spread.items() if c}
        best_spread = min(avg_by_spread, key=avg_by_spread.get) if avg_by_spread else None
        results[K] = dict(avg_by_spread=avg_by_spread, n_paired=n_paired, best_spread=best_spread)
        print(f"K={K}: n_paired_instances={n_paired}, avg_cost_by_spread={avg_by_spread} -> best={best_spread}")
    return results


if __name__ == "__main__":
    calibrate()
