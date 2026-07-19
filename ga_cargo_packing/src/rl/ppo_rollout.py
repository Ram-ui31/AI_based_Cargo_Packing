"""
PPO-specific rollout helper: replay a FIXED action sequence (sampled under an
earlier/"old" policy) and recompute its log-probs/entropy under the CURRENT
model parameters, for PPO's importance-sampling ratio exp(new_lp - old_lp).

This does NOT modify src/rl/model.py -- it only calls model.forward() (a
public method) and reimplements the same masking loop model.sample_actions()
already uses, substituting "look up the fixed action" for "sample a new
action". Kept as a separate module (not touching train_rl.py either) so the
existing REINFORCE training pipeline and its checkpoints are unaffected by
this PPO experiment regardless of how it turns out.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import MAX_N_ULDS, MAX_N_PKGS


def replay_log_probs(model, uld_feats, pkg_feats, key_padding_mask,
                      n_ulds, dim_mask, priority_mask, tightness, k_feat,
                      n_pkgs, pkg_weights, uld_weight_limits, fixed_actions,
                      temperature=1.0, pkg_volumes=None, uld_volumes=None):
    """
    Mirrors model.sample_actions()'s per-package sequential masking loop
    exactly (same iteration order: natural dataframe order 0..n_pkgs-1, same
    running weight_used/volume_used state machine), but REPLAYS
    `fixed_actions[i]` instead of drawing a fresh sample at each step.

    Running weight/volume usage is fully determined by the fixed action
    sequence alone (it does not depend on which policy produced those
    actions), so masks reconstruct identically to what the original
    sampling pass saw -- this is what makes off-policy log-prob recomputation
    valid for PPO's ratio.

    If a fixed action's own logit reads as masked (-1e9) under the current
    weight/volume state, we force it to 0.0 before the softmax -- this
    mirrors sample_actions()'s priority-forced-availability override
    without needing to know whether the original draw was itself an
    override (the condition "this specific action must be makeable" is
    necessary and sufficient to reproduce a well-defined, non--inf log-prob
    for it, exactly as the original sampling pass guaranteed).

    Returns:
        log_probs   : (MAX_N_PKGS,) float, differentiable, log-prob of
                       fixed_actions[i] under CURRENT params for i < n_pkgs
        entropy     : scalar, differentiable, entropy of the current
                       per-package action distributions (same definition as
                       sample_actions()'s returned entropy)
        logits_batch: (MAX_N_PKGS, N_ULD_CLASSES) raw current-param logits,
                       for computing auxiliary losses (hinge/spread) on
    """
    logits_batch = model.forward(
        uld_feats, pkg_feats, key_padding_mask,
        torch.tensor([n_ulds], device=uld_feats.device),
        dim_mask, priority_mask, tightness, k_feat,
    ).squeeze(0)

    entropy = -(F.softmax(logits_batch[:n_pkgs], dim=-1) *
                F.log_softmax(logits_batch[:n_pkgs], dim=-1)).sum()

    log_probs   = torch.zeros(MAX_N_PKGS, device=uld_feats.device)
    weight_used = [0.0] * n_ulds
    volume_used = [0.0] * n_ulds
    track_vol   = pkg_volumes is not None and uld_volumes is not None

    for i in range(n_pkgs):
        action = int(fixed_actions[i])

        lg = logits_batch[i].clone()
        for j in range(n_ulds):
            if weight_used[j] + pkg_weights[i] > uld_weight_limits[j]:
                lg[j] = -1e9
        if track_vol:
            for j in range(n_ulds):
                if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                    lg[j] = -1e9

        if lg[action].item() <= -1e8:
            lg[action] = 0.0

        log_probs[i] = F.log_softmax(lg / temperature, dim=-1)[action]

        if action < n_ulds:
            weight_used[action] += pkg_weights[i]
            if track_vol:
                volume_used[action] += pkg_volumes[i]

    return log_probs, entropy, logits_batch
