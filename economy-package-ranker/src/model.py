"""
model.py -- PackageSetRanker: a permutation-invariant, set-attention
scorer over the full candidate Economy package set.

Why set-attention (not an isolated per-package formula): every classical
scoring formula tried (value_density, value_density_pow*, wpow*,
joint_pow*) scores each package independently of what else is being
selected -- structurally unable to represent "this package is worth
including GIVEN what else is competing for the same space." Self-attention
over the whole candidate set lets each package's score depend on the
OTHER packages present -- e.g. a package can look "expensive" in isolation
but be a great choice once the model sees that few other packages compete
for similar volume/weight, or vice versa.

Per-package input features (all cheap, no real geometry needed at
inference time -- geometry only appears in the LABELS this was trained on):
    length, width, height, volume, weight, delay_cost,
    value_density (delay_cost/volume), volume_frac (volume / avg ULD volume),
    weight_frac (weight / avg ULD weight_limit)
Plus a small set of GLOBAL context scalars (broadcast to every package):
    n_ulds, total_remaining_volume, total_remaining_weight, K value.

Output: one real-valued score per package -- sort descending, greedy
first-fit exactly like the existing econ_sort_key mechanism, so this is a
drop-in replacement for the sort key, not a new packing algorithm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


PACKAGE_FEATURE_DIM = 9
GLOBAL_FEATURE_DIM = 4


class PackageSetRanker(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.pkg_proj = nn.Linear(PACKAGE_FEATURE_DIM, d_model)
        self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1),
        )

    def forward(self, pkg_feats, global_feats, key_padding_mask=None):
        """
        pkg_feats         : (B, N, PACKAGE_FEATURE_DIM)
        global_feats      : (B, GLOBAL_FEATURE_DIM)
        key_padding_mask  : (B, N) bool, True = PAD (ignored), for batches
                            with variable N via padding.
        Returns: (B, N) real-valued scores (higher = more worth including).
        """
        x = self.pkg_proj(pkg_feats)                       # (B, N, d)
        g = self.global_proj(global_feats).unsqueeze(1)     # (B, 1, d)
        x = x + g                                            # broadcast global context into every package token
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        scores = self.score_head(h).squeeze(-1)             # (B, N)
        return scores
