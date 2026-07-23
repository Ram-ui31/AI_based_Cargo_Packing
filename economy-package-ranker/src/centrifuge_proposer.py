"""
centrifuge_proposer.py -- CentrifugeEvictProposer: predicts the REAL net
delay-cost gain of a "evict this Economy package -> compact its
container -> refill from the unplaced pool" move, without running the
actual geometry (Heightmap.compact() + greedy refill) at inference time.

Why this needs set-attention over TWO different sets (not a flat MLP like
SwapProposer): whether evicting a specific package is profitable depends
on (a) what else is already in the container -- compaction can only help
if the OTHER placed boxes leave a consolidatable void once this one is
gone, and (b) what's in the unplaced pool -- there's no gain unless
SOMETHING currently unplaced actually fits the freed, compacted space.
Both are genuinely set-valued, order-independent contexts (container
contents and the unplaced pool), unlike SwapProposer's move which only
ever involves two fixed named packages. This mirrors PackageSetRanker's
set-attention design, applied to two sets instead of one.

Per-package feature vector (7 dims, cheap, no real geometry involved):
    length, width, height, volume, weight, delay_cost, value_density
(value_density = delay_cost / volume**1.5, matching the sort key used
throughout model-training-pipeline's greedy refill).

Output: one real-valued predicted net_gain per (container, evict_pkg,
unplaced_pool) example -- positive means the model expects eviction to
be profitable. Sign gives the win/loss classification; magnitude gives
a ranking so only the top-K predicted candidates need the real
compact+refill check at inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PKG_FEATURE_DIM = 7
ULD_FEATURE_DIM = 4     # length, width, height, weight_limit
GLOBAL_FEATURE_DIM = 3  # k_value, n_container_pkgs, n_unplaced_pool


def pkg_to_feat_vec(pkg):
    """pkg: dict with length, width, height, weight, delay_cost (matches
    generate_centrifuge_data.py's pkg_record). Returns 7-float list."""
    vol = pkg['length'] * pkg['width'] * pkg['height']
    value_density = pkg['delay_cost'] / max(vol, 1.0) ** 1.5
    return [pkg['length'], pkg['width'], pkg['height'], vol, pkg['weight'], pkg['delay_cost'], value_density]


class SetEncoder(nn.Module):
    """Permutation-invariant set encoder: project -> transformer encoder ->
    masked mean pool. Used identically for container contents and the
    unplaced pool (same architecture, separate weights -- the two sets
    play different roles so shouldn't share parameters)."""

    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(PKG_FEATURE_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, feats, key_padding_mask=None):
        """
        feats             : (B, N, PKG_FEATURE_DIM)
        key_padding_mask  : (B, N) bool, True = PAD
        Returns: (B, d_model) pooled embedding.
        """
        x = self.proj(feats)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)  # (B, N, 1)
            summed = (h * valid).sum(dim=1)
            count = valid.sum(dim=1).clamp(min=1.0)
            return summed / count
        return h.mean(dim=1)


class CentrifugeEvictProposer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.container_encoder = SetEncoder(d_model, n_heads, n_layers, dropout)
        self.pool_encoder = SetEncoder(d_model, n_heads, n_layers, dropout)
        self.evict_proj = nn.Sequential(nn.Linear(PKG_FEATURE_DIM, d_model), nn.ReLU())
        self.uld_proj = nn.Sequential(nn.Linear(ULD_FEATURE_DIM, d_model), nn.ReLU())
        self.global_proj = nn.Sequential(nn.Linear(GLOBAL_FEATURE_DIM, d_model), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, container_feats, container_mask, pool_feats, pool_mask,
                evict_feats, uld_feats, global_feats):
        """
        container_feats : (B, Nc, PKG_FEATURE_DIM)   container_mask: (B, Nc) bool, True=PAD
        pool_feats       : (B, Np, PKG_FEATURE_DIM)   pool_mask: (B, Np) bool, True=PAD
        evict_feats      : (B, PKG_FEATURE_DIM)
        uld_feats        : (B, ULD_FEATURE_DIM)
        global_feats     : (B, GLOBAL_FEATURE_DIM)
        Returns: (B,) predicted net_gain.
        """
        c = self.container_encoder(container_feats, container_mask)
        p = self.pool_encoder(pool_feats, pool_mask)
        e = self.evict_proj(evict_feats)
        u = self.uld_proj(uld_feats)
        g = self.global_proj(global_feats)
        x = torch.cat([c, p, e, u, g], dim=-1)
        return self.head(x).squeeze(-1)
