"""Package -> ULD (or leave-behind) assignment network.

Fresh implementation: a small Transformer encoder over ULD tokens + package
tokens (padded to fixed max sizes -- 6 ULDs, 384 packages comfortably covers
every instance in good_data), with a per-package classification head over
{ULD slots, NONE}.

K IS fed as an explicit input feature (log-normalized, alongside the
K-independent "tightness" scalars) -- revisiting the prior-art rationale
("tightness generalizes across K, raw K would just memorize a scale") after
finding that a K-blind assignment policy can't adapt its spread-vs-delay
tradeoff per instance: it converges to one roughly fixed spread regardless
of K, which is exactly wrong (high K should push toward fewer ULDs for
priority; low K should favor spreading out to fit more economy packages).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

MAX_N_ULDS = 6
MAX_N_PKGS = 384
N_CLASSES = MAX_N_ULDS + 1  # last class = leave-behind ("NONE")
NONE_CLASS = MAX_N_ULDS

ULD_FEAT_DIM = 5
PKG_FEAT_DIM = 8
CONTEXT_DIM = 3  # wt_tightness, vol_tightness, log-normalized K

# normalization constants from good_data_generation.ipynb's sampling ranges
ULD_L_MAX, ULD_W_MAX, ULD_H_MAX, ULD_WT_MAX = 240.0, 320.0, 240.0, 5000.0
PKG_DIM_MAX, PKG_WT_MAX, DELAY_MAX = 110.0, 271.0, 180.0
K_LOG_MIN, K_LOG_MAX = math.log10(100), math.log10(5000)  # good_data's K range


def normalize_k(K: float) -> float:
    return (math.log10(max(K, 1.0)) - K_LOG_MIN) / (K_LOG_MAX - K_LOG_MIN)


def uld_features(ulds_df) -> np.ndarray:
    feats = np.zeros((MAX_N_ULDS, ULD_FEAT_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(ulds_df.iterrows()):
        if i >= MAX_N_ULDS:
            break
        l, w, h, wl = row["Length"], row["Width"], row["Height"], row["Weight_Limit"]
        vol = l * w * h
        feats[i] = [l / ULD_L_MAX, w / ULD_W_MAX, h / ULD_H_MAX, wl / ULD_WT_MAX,
                    vol / (ULD_L_MAX * ULD_W_MAX * ULD_H_MAX)]
    return feats


def package_features(packages_df) -> np.ndarray:
    n = min(len(packages_df), MAX_N_PKGS)
    feats = np.zeros((MAX_N_PKGS, PKG_FEAT_DIM), dtype=np.float32)
    rows = packages_df.iloc[:n]
    vol = rows["Length"] * rows["Width"] * rows["Height"]
    is_prio = (rows["Type"] == "Priority").astype(np.float32)
    delay = rows["Delay_Cost"].astype(np.float32)
    cost_density = delay / vol.clip(lower=1.0)
    feats[:n, 0] = rows["Length"] / PKG_DIM_MAX
    feats[:n, 1] = rows["Width"] / PKG_DIM_MAX
    feats[:n, 2] = rows["Height"] / PKG_DIM_MAX
    feats[:n, 3] = rows["Weight"] / PKG_WT_MAX
    feats[:n, 4] = vol / (PKG_DIM_MAX ** 3)
    feats[:n, 5] = is_prio
    feats[:n, 6] = delay / DELAY_MAX
    feats[:n, 7] = cost_density.clip(upper=5.0) / 5.0
    return feats


def compute_tightness(packages_df, ulds_df) -> np.ndarray:
    pkg_vol = float((packages_df["Length"] * packages_df["Width"] * packages_df["Height"]).sum())
    pkg_wt = float(packages_df["Weight"].sum())
    uld_vol = float((ulds_df["Length"] * ulds_df["Width"] * ulds_df["Height"]).sum())
    uld_wt = float(ulds_df["Weight_Limit"].sum())
    return np.array([
        pkg_vol / uld_vol if uld_vol else 0.0,
        pkg_wt / uld_wt if uld_wt else 0.0,
    ], dtype=np.float32)


def compute_context(packages_df, ulds_df, K: float) -> np.ndarray:
    tightness = compute_tightness(packages_df, ulds_df)
    return np.concatenate([tightness, [normalize_k(K)]]).astype(np.float32)


def dim_fits(pkg_l, pkg_w, pkg_h, uld_l, uld_w, uld_h) -> bool:
    """Sorted-axis fit check: does the package fit in the ULD in *some* orientation."""
    p = sorted([pkg_l, pkg_w, pkg_h])
    u = sorted([uld_l, uld_w, uld_h])
    return p[0] <= u[0] and p[1] <= u[1] and p[2] <= u[2]


def compute_masks(packages_df, ulds_df) -> np.ndarray:
    """(n_packages, N_CLASSES) bool mask: True = allowed."""
    n_pkgs = min(len(packages_df), MAX_N_PKGS)
    n_ulds = min(len(ulds_df), MAX_N_ULDS)
    mask = np.zeros((MAX_N_PKGS, N_CLASSES), dtype=bool)
    uld_rows = list(ulds_df.iloc[:n_ulds].itertuples())
    for i, row in enumerate(packages_df.iloc[:n_pkgs].itertuples()):
        for j, u in enumerate(uld_rows):
            if dim_fits(row.Length, row.Width, row.Height, u.Length, u.Width, u.Height):
                mask[i, j] = True
        is_priority = row.Type == "Priority"
        mask[i, NONE_CLASS] = not is_priority
    return mask


class AssignmentPolicy(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.uld_proj = nn.Linear(ULD_FEAT_DIM, d_model)
        self.pkg_proj = nn.Linear(PKG_FEAT_DIM, d_model)
        self.context_proj = nn.Linear(CONTEXT_DIM, d_model)
        self.type_embed = nn.Embedding(2, d_model)  # 0=ULD, 1=package
        self.pos_embed = nn.Embedding(MAX_N_ULDS + MAX_N_PKGS, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # K is concatenated directly here too (not just mixed into the shared
        # encoder via context_proj) -- found that a K signal only entering
        # through one generic additive term to every token gets diluted by
        # the time it reaches the classification head: the policy converged
        # to a fixed "always minimize spread" strategy regardless of K,
        # rather than genuinely conditioning on it. This gives K a short,
        # direct, low-noise path straight into the final decision.
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, d_model), nn.ReLU(),
            nn.Linear(d_model, N_CLASSES),
        )

    def forward(self, uld_feats, pkg_feats, context, key_padding_mask):
        """All inputs are batched (B, ...). Returns per-package logits (B, MAX_N_PKGS, N_CLASSES)."""
        B = uld_feats.shape[0]
        uld_tok = self.uld_proj(uld_feats) + self.type_embed(torch.zeros(1, dtype=torch.long))
        pkg_tok = self.pkg_proj(pkg_feats) + self.type_embed(torch.ones(1, dtype=torch.long))
        tok = torch.cat([uld_tok, pkg_tok], dim=1)
        pos_ids = torch.arange(MAX_N_ULDS + MAX_N_PKGS).unsqueeze(0).expand(B, -1)
        tok = tok + self.pos_embed(pos_ids)
        tok = tok + self.context_proj(context).unsqueeze(1)

        enc = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        pkg_enc = enc[:, MAX_N_ULDS:, :]
        k_raw = context[:, -1:].unsqueeze(1).expand(-1, MAX_N_PKGS, -1)  # (B, MAX_N_PKGS, 1)
        return self.head(torch.cat([pkg_enc, k_raw], dim=-1))
