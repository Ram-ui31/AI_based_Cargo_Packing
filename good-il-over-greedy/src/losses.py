import torch
import torch.nn.functional as F

from .config import MAX_N_ULDS, IGNORE_INDEX

# ─────────────────────────────────────────────────────────────────────────────
# CAPACITY-VIOLATION PENALTY
#
# Plain cross-entropy only rewards matching the labeller's chosen ULD index.
# It never tells the model "and by the way, that ULD only had 40kg of
# headroom left" — so nothing in the loss objects when the model's predicted
# assignment would blow through a ULD's weight or volume limit. Hard masks
# in TransformerClusterer._apply_hard_masks() only cover physical dimension
# fit, sequence range, and the priority->NONE rule; weight/volume are NOT
# hard-masked in the supervised forward() pass used during IL training
# (model.sample_actions(), used for RL, does apply weight/volume masking).
#
# This auxiliary penalty is computed directly from the model's soft (softmax)
# output distribution, per ULD, per batch instance:
#   1. For each package i and each ULD j, p[i, j] = P(model assigns i -> j).
#   2. expected_weight[j] = sum_i p[i, j] * pkg_weight[i]
#      expected_volume[j] = sum_i p[i, j] * pkg_volume[i]
#   3. overflow_weight[j] = relu(expected_weight[j] - uld_weight_limit[j])
#      overflow_volume[j] = relu(expected_volume[j] - uld_volume_limit[j])
#   4. penalty = mean over ULDs and batch of (overflow_weight + overflow_volume),
#      each normalised by that ULD's own limit so the penalty is unitless and
#      comparable in scale to the cross-entropy term regardless of how large
#      ULDs/packages are in absolute kg / cm^3.
#
# Using the softmax probabilities (rather than a hard argmax) keeps this
# differentiable, so gradients flow back through the classification head and
# the model is pushed, every batch, to lower its assigned probability mass on
# any ULD it's currently overfilling — for every ULD, not just one. This is
# on top of (not a replacement for) the cross-entropy imitation loss:
#
#   total_loss = ce_loss
#              + LAMBDA_WEIGHT_PENALTY * weight_overflow_penalty
#              + LAMBDA_VOLUME_PENALTY * volume_overflow_penalty
# ─────────────────────────────────────────────────────────────────────────────

def capacity_violation_penalty(logits, pkg_feats, uld_feats, n_ulds_batch, labels):
    """
    Differentiable penalty for predicted assignments that would overflow a
    ULD's weight or volume limit, computed per-ULD and averaged over the batch.

    logits       : (B, MAX_N_PKGS, N_ULD_CLASSES) raw model output
    pkg_feats    : (B, MAX_N_PKGS, PKG_FEAT_DIM)  normalised package features
                   (index 3 = weight / MAX_PKG_WEIGHT, index 4 = volume / MAX_PKG_DIM**3)
    uld_feats    : (B, MAX_N_ULDS, ULD_FEAT_DIM)  normalised ULD features
                   (index 3 = weight_limit / MAX_ULD_WEIGHT, index 4 = volume / MAX_ULD_DIM**3)
    n_ulds_batch : (B,) actual number of ULDs per instance
    labels       : (B, MAX_N_PKGS) ground-truth labels, used only to build the
                   "real package" mask (label != IGNORE_INDEX), so padding
                   packages contribute zero to the penalty.

    Returns: (weight_penalty, volume_penalty) — two scalars, each averaged
    over real ULDs across the batch.
    """
    B = logits.shape[0]
    device = logits.device

    probs = F.softmax(logits, dim=-1)                    # (B, MAX_N_PKGS, N_ULD_CLASSES)
    uld_probs = probs[:, :, :MAX_N_ULDS]                  # (B, MAX_N_PKGS, MAX_N_ULDS) — exclude NONE

    real_pkg_mask = (labels != IGNORE_INDEX).float()      # (B, MAX_N_PKGS)
    uld_probs = uld_probs * real_pkg_mask.unsqueeze(-1)   # zero out padding packages

    pkg_weight = pkg_feats[:, :, 3]                        # (B, MAX_N_PKGS)  normalised
    pkg_volume = pkg_feats[:, :, 4]                        # (B, MAX_N_PKGS)  normalised

    # Expected (soft) weight/volume placed in each ULD under the model's
    # current predicted distribution.
    expected_weight = torch.einsum('bp,bpu->bu', pkg_weight, uld_probs)   # (B, MAX_N_ULDS)
    expected_volume = torch.einsum('bp,bpu->bu', pkg_volume, uld_probs)   # (B, MAX_N_ULDS)

    uld_weight_limit = uld_feats[:, :, 3].clamp(min=1e-6)   # (B, MAX_N_ULDS) normalised
    uld_volume_limit = uld_feats[:, :, 4].clamp(min=1e-6)   # (B, MAX_N_ULDS) normalised

    weight_overflow = F.relu(expected_weight - uld_weight_limit) / uld_weight_limit
    volume_overflow = F.relu(expected_volume - uld_volume_limit) / uld_volume_limit

    uld_range  = torch.arange(MAX_N_ULDS, device=device).unsqueeze(0).expand(B, -1)
    real_uld_mask = (uld_range < n_ulds_batch.unsqueeze(1)).float()       # (B, MAX_N_ULDS)

    n_real_ulds = real_uld_mask.sum().clamp(min=1.0)
    weight_penalty = (weight_overflow * real_uld_mask).sum() / n_real_ulds
    volume_penalty = (volume_overflow * real_uld_mask).sum() / n_real_ulds

    return weight_penalty, volume_penalty
