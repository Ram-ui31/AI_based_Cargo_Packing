"""Largest-volume-first greedy baseline for the placement env, used only as a
sanity-check reference for the learned policy -- not part of training."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl_packer", "src"))

from placement_env import PlacementEnv


def greedy_pack(uld_row, packages_df) -> float:
    env = PlacementEnv(uld_row, packages_df)
    while not env.done:
        cands = env.candidates()
        if not cands:
            env.close_out()
            break
        # largest-volume package first; among ties, lowest z then lowest x,y
        best = max(cands, key=lambda c: (c.volume, -c.z, -c.x, -c.y))
        env.step(best)
    return env.hm.utilization()


class GreedyPolicy:
    """Drop-in replacement for PlacementPolicy's .select() interface (same
    largest-volume-first / lowest-corner rule as greedy_pack), so it can be
    passed straight into assignment_env.evaluate_assignment() and reuse the
    exact same priority-first/fallback/economy logic as the RL and py3dbp
    comparisons -- only the per-step placement rule differs."""

    def select(self, cands, hm, pool, greedy: bool = True, rng=None):
        idx = max(range(len(cands)), key=lambda i: (cands[i].volume, -cands[i].z, -cands[i].x, -cands[i].y))
        return idx, None, None
