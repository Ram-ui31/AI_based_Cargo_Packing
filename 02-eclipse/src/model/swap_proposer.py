"""
swap_proposer.py -- SwapProposer: a small model that predicts the REAL
cost delta of swapping two specific Economy packages' positions in the
beam search's ordering, trained on (move, real_delta) pairs the beam
search itself generates as a byproduct (see ga_cargo_packing/scripts/
beam_search_economy.py's beam_moves_*.jsonl logging).

Why this is a different learning target than PackageSetRanker (the earlier
GNN attempt) or RL/GRPO: those asked a model to output GLOBAL per-package
scores that get discretized into a full ordering -- a smooth proxy for a
provably jagged, discontinuous problem (confirmed all night: tiny ranking
changes cause large non-monotonic real-cost swings). This model instead
predicts the outcome of one CONCRETE, ATOMIC action (swap package A and
package B) -- exactly the granularity the beam search already operates
at, so there's no smoothness mismatch to fight. It doesn't replace the
beam search; it makes the beam search's random candidate generation into
GUIDED candidate generation -- screen many cheap candidate swaps with this
model, real-evaluate only the most promising few, so the same real-
evaluation budget explores far more effectively.

Input per swap candidate: features of package A, features of package B,
their normalized positions in the current order, and the same
GLOBAL context features used elsewhere (n_ulds, K, total remaining
capacity). Output: predicted real cost delta (child_cost - parent_cost)
-- negative means the model expects this swap to IMPROVE cost.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PACKAGE_FEATURE_DIM = 9  # matches features.py's build_package_features
GLOBAL_FEATURE_DIM = 4


class SwapProposer(nn.Module):
    def __init__(self, hidden=48, dropout=0.2):
        super().__init__()
        in_dim = PACKAGE_FEATURE_DIM * 2 + GLOBAL_FEATURE_DIM + 2  # +2 for normalized pos_a, pos_b
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_a, feat_b, global_feat, pos_a, pos_b):
        """
        feat_a, feat_b   : (B, PACKAGE_FEATURE_DIM)
        global_feat      : (B, GLOBAL_FEATURE_DIM)
        pos_a, pos_b     : (B, 1) normalized positions in [0, 1]
        Returns: (B,) predicted real cost delta (lower/more negative = more promising swap).
        """
        x = torch.cat([feat_a, feat_b, global_feat, pos_a, pos_b], dim=-1)
        return self.mlp(x).squeeze(-1)
