"""Actor-critic policy that scores (package, extreme-point, orientation) candidates.

Variable-size action space per step (pool shrinks, extreme-point set changes),
so instead of a fixed masked-discrete head, every valid candidate is featurized
and scored by a shared MLP; softmax over the *valid* candidates only is the
masking mechanism (nothing invalid is ever in the list).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from geometry import Heightmap
from placement_env import Candidate

CAND_FEAT_DIM = 12
GLOBAL_FEAT_DIM = 10


def featurize_candidates(cands: list[Candidate], hm: Heightmap) -> np.ndarray:
    L, W, H, wlim = hm.length, hm.width, hm.height, hm.weight_limit
    vol = hm.volume
    feats = np.empty((len(cands), CAND_FEAT_DIM), dtype=np.float32)
    for i, c in enumerate(cands):
        feats[i] = [
            c.dx / L, c.dy / W, c.dz / H,
            c.weight / wlim if wlim else 0.0,
            c.volume / vol if vol else 0.0,
            c.x / L, c.y / W, c.z / H,
            (L - (c.x + c.dx)) / L,
            (W - (c.y + c.dy)) / W,
            (H - (c.z + c.dz)) / H,
            1.0 if c.is_priority else 0.0,
        ]
    return feats


def featurize_global(hm: Heightmap, pool) -> np.ndarray:
    vol, wlim = hm.volume, hm.weight_limit
    remaining_vol = int((pool["Length"] * pool["Width"] * pool["Height"]).sum()) if len(pool) else 0
    remaining_weight = float(pool["Weight"].sum()) if len(pool) else 0.0
    L, W, H = hm.length, hm.width, hm.height if hm.height else 1
    n = len(hm.placements)
    max_x1 = max((p.x1 for p in hm.placements), default=0)
    max_y1 = max((p.y1 for p in hm.placements), default=0)
    max_z1 = max((p.z1 for p in hm.placements), default=0)
    mean_z = float(np.mean([p.z0 for p in hm.placements])) if n else 0.0
    return np.array([
        min(len(pool) / 60.0, 1.0),
        remaining_vol / vol if vol else 0.0,
        remaining_weight / wlim if wlim else 0.0,
        hm.weight_used / wlim if wlim else 0.0,
        hm.utilization(),
        min(n / 100.0, 1.0),
        max_x1 / L if L else 0.0,
        max_y1 / W if W else 0.0,
        max_z1 / H,
        mean_z / H,
    ], dtype=np.float32)


class PlacementPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(CAND_FEAT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(GLOBAL_FEAT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def logits(self, cand_feats: torch.Tensor) -> torch.Tensor:
        return self.actor(cand_feats).squeeze(-1)

    def value(self, global_feats: torch.Tensor) -> torch.Tensor:
        return self.critic(global_feats).squeeze(-1)

    def select(self, cands: list[Candidate], hm: Heightmap, pool, greedy: bool = False,
               rng: np.random.Generator | None = None):
        feats = torch.from_numpy(featurize_candidates(cands, hm))
        logits = self.logits(feats)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            idx = int(torch.argmax(logits).item())
        else:
            idx = int(dist.sample().item())
        logprob = dist.log_prob(torch.tensor(idx))
        entropy = dist.entropy()
        return idx, logprob, entropy
