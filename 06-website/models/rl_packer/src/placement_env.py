"""Single-ULD placement episode using true 3D placement (geometry.py):
z is a genuine choice via pivot points, never derived/forced. Reward is the
normalized volume of whatever gets placed, so a full episode's return
telescopes to final volume utilization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geometry import Heightmap

MAX_PIVOTS = 400


@dataclass
class Candidate:
    pool_idx: int  # row-label index into self.pool
    pivot_idx: int
    orient_idx: int
    dx: int
    dy: int
    dz: int
    x: int
    y: int
    z: int
    volume: int
    weight: float
    is_priority: bool


class PlacementEnv:
    def __init__(self, uld_row: pd.Series | None, packages_df: pd.DataFrame, max_pivots: int = MAX_PIVOTS,
                 hm: Heightmap | None = None):
        if hm is not None:
            self.hm = hm  # continue placing into an already-partially-filled ULD
        else:
            self.hm = Heightmap(
                length=int(uld_row["Length"]),
                width=int(uld_row["Width"]),
                height=int(uld_row["Height"]),
                weight_limit=float(uld_row["Weight_Limit"]),
            )
        self.pool = packages_df.reset_index(drop=True).copy()
        self.max_pivots = max_pivots
        self.placed_ids: list[str] = []
        self.left_behind_ids: list[str] = []
        self.done = len(self.pool) == 0

    def candidates(self) -> list[Candidate]:
        """Enumerate all currently valid (package, pivot point, orientation) triples."""
        if len(self.pool) == 0:
            return []
        pivots = self.hm.pivot_points(cap=self.max_pivots)
        cands: list[Candidate] = []
        for pool_idx, row in self.pool.iterrows():
            l, w, h, weight = int(row["Length"]), int(row["Width"]), int(row["Height"]), float(row["Weight"])
            is_priority = row["Type"] == "Priority"
            seen_dims = set()
            for orient_idx in range(6):
                dx, dy, dz = self.hm.orientation_dims(l, w, h, orient_idx)
                if (dx, dy, dz) in seen_dims:
                    continue  # skip duplicate orientation when two dims are equal
                seen_dims.add((dx, dy, dz))
                for pivot_idx, (x, y, z) in enumerate(pivots):
                    if self.hm.fits(dx, dy, dz, x, y, z, weight):
                        cands.append(Candidate(pool_idx, pivot_idx, orient_idx, dx, dy, dz, x, y, z,
                                                dx * dy * dz, weight, is_priority))
        return cands

    def step(self, cand: Candidate) -> float:
        row = self.pool.loc[cand.pool_idx]
        self.hm.place(row["Package_ID"], cand.dx, cand.dy, cand.dz, cand.x, cand.y, cand.z, cand.weight)
        self.placed_ids.append(row["Package_ID"])
        self.pool = self.pool.drop(cand.pool_idx)
        reward = cand.volume / self.hm.volume
        if len(self.pool) == 0:
            self.done = True
        return reward

    def close_out(self) -> None:
        """Call once no candidate placements remain; whatever's left is left-behind."""
        self.left_behind_ids = list(self.pool["Package_ID"])
        self.done = True

    def run_episode(self, policy, rng: np.random.Generator | None = None, greedy: bool = False):
        """Roll out a full episode using `policy.select(candidates, hm) -> (index, logprob, entropy)`.

        Returns (rewards, logprobs, entropies, final_utilization).
        """
        rewards, logprobs, entropies = [], [], []
        while not self.done:
            cands = self.candidates()
            if not cands:
                self.close_out()
                break
            idx, logprob, entropy = policy.select(cands, self.hm, self.pool, greedy=greedy, rng=rng)
            r = self.step(cands[idx])
            rewards.append(r)
            logprobs.append(logprob)
            entropies.append(entropy)
        return rewards, logprobs, entropies, self.hm.utilization()
