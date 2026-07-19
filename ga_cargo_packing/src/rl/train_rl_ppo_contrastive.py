"""
PPO + per-K EMA baseline training loop, IDENTICAL to train_rl_ppo.py except
for one addition: a contrastive cross-K loss term.

Why: multi-K training data alone (see train_ga_rl_ppo_multik.py) removed the
confound where different K's in the training set correlated with different
instance features, but did NOT produce causal K-sensitivity in the trained
policy (verified: same-instance K-sweep test showed a perfectly flat spread
curve, 2.12 at every K). Root cause diagnosed: PPO's advantage is normalized
by a baseline tracked SEPARATELY per K bucket
(`baseline[k_value]`/`baseline_std[k_value]`), so the loss at every update
only ever asks "did this rollout do better or worse than typical FOR THIS
K?" -- nothing in the objective ever contrasts the SAME instance's outcome
AT DIFFERENT K values, so a K-independent policy can satisfy every K
bucket's baseline about equally well.

Fix: for every collected episode (instance, k_value), also do one extra
forward pass (no rollout/sampling, no packer call -- cheap) on the SAME
instance at a second, randomly-chosen K. Compute the differentiable
soft_spread_loss_ipr for both, and penalize whenever the HIGHER-K pass
doesn't show a smaller soft-spread than the LOWER-K pass, by a margin that
SCALES with how far apart the pair is in normalized K-space (see
`contrastive_margin_slope` below -- a fixed constant margin gives zero
further gradient the moment any sampled pair clears it, which in practice
meant training stopped pushing spread down once adjacent, closely-spaced
K pairs were satisfied, well before more widely-spaced pairs like K=100 vs
K=5000 reached anything like their true achievable separation):

    gap    = |normalize_k(higher_K) - normalize_k(lower_K)|
    margin = contrastive_margin_slope * gap
    contrastive = relu(soft_spread(higher_K) - soft_spread(lower_K) + margin)

This is the term that was structurally missing -- it directly rewards the
network for using K as a causal control variable on a per-instance basis,
rather than permitting (and, as it turned out, encouraging) a constant bias.

This is a SEPARATE training path from train_rl_ppo.py -- does not modify
that file, model.py, reward.py, or data_utils.py, only imports from them.
"""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MAX_N_ULDS, MAX_N_PKGS,
    RL_LR, RL_GRAD_CLIP, RL_ENTROPY_COEF,
    RL_HINGE_COEF, RL_HINGE_MARGIN, RL_SPREAD_COEF,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    RL_TEMPERATURE, MAX_SAFE_PKGS, MAX_SAFE_ULDS, DEVICE,
)
from .data_utils import build_tensors, needs_chunking, actions_to_assignment, normalize_k
from .reward import (
    compute_packing_cost, rl_capacity_violation_penalty,
    feasibility_hinge_loss, soft_spread_loss_ipr,
)
from .train_rl import rl_assign_argmax_safe, _consolidate_priority_by_capacity
from .ppo_rollout import replay_log_probs
from .packer import DEFAULT_PACKER
from . import train_rl_ppo as _ppo_mod
from .train_rl_ppo import _load_test_pool, _collect_rollout, _print_update
# NOTE: _load_train_pool is deliberately NOT imported by name here -- callers
# that want a custom training pool (e.g. scripts/train_ga_rl_ppo_contrastive.py's
# multi-K pool) monkey-patch `_ppo_mod._load_train_pool` before calling
# train_rl_ppo_contrastive(); a direct `from .train_rl_ppo import
# _load_train_pool` would bind an independent name here that such a patch
# could never reach, since Python's `from X import Y` copies the reference
# at import time rather than looking it up on the module each call.

K_VALUES = [100, 500, 1000, 3000, 5000]


def _economy_only_entropy(logits, priority_mask, n_pkgs):
    """Entropy bonus restricted to Economy positions (priority_mask==True,
    see soft_spread_loss's docstring for the True=Economy convention).

    Root cause found by directly inspecting a trained checkpoint: Priority
    package logits across ULDs were nearly flat (std ~0.02, near-uniform
    softmax) no matter how long training ran or how large contrastive_coef
    was set, because entropy_loss (mean abs ~14, by far the largest loss
    term) was computed over ALL n_pkgs positions and dominated the gradient
    on the shared output_head weights every single update -- directly
    fighting the concentration that spread_loss/contrastive_loss need to
    achieve (fewer priority-holding ULDs = confident, NOT uniform, per-
    package softmax). Economy's entropy bonus (avoiding premature collapse
    on the much larger, harder NONE-vs-ULD decision) is still useful and
    kept; Priority packages get zero entropy pressure so the much smaller
    spread-shaping terms actually have room to move their logits.
    """
    lg   = logits[:n_pkgs]
    keep = priority_mask[:n_pkgs]
    if keep.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    probs     = F.softmax(lg[keep], dim=-1)
    log_probs = F.log_softmax(lg[keep], dim=-1)
    return -(probs * log_probs).sum()


def _consolidation_imitation_loss(logits, priority_mask, n_pkgs, n_ulds,
                                   pkg_id_list, teacher_assignment, uld_id_to_idx):
    """
    Cross-entropy loss pulling the model's own Priority-package softmax
    toward `_consolidate_priority_by_capacity`'s near-optimal greedy FFD
    consolidation target -- direct supervised signal for a combinatorial
    sub-problem (which few ULDs should hold ALL Priority) that raw RL and
    the contrastive cross-K term alone haven't been enough to discover: even
    after fixing soft_spread_loss's saturation (soft_spread_loss_ipr) and
    stabilizing the contrastive margin (scaled by K-gap, coef=20,
    slope=2.0 -- which did fix the earlier reversal/instability problem),
    a same-instance K-sweep with the heuristic disabled still showed the
    model's own unmasked predictions trailing the heuristic by ~16% mean
    cost. RL's reward for spread is sparse and indirect (whole-episode
    cost); this loss instead directly tells the network what the answer
    is, package by package. Caller scales this by normalize_k(k_value) so
    it fully applies at high K (where the heuristic's answer is clearly
    best) and fades to ~0 at low K (where the earlier K=100 analysis found
    the model's OWN more-spread-but-less-delay tradeoff was actually
    better than forcing maximal consolidation).
    """
    is_priority = ~priority_mask[:n_pkgs]
    prio_positions = is_priority.nonzero(as_tuple=True)[0]
    if len(prio_positions) == 0:
        return torch.tensor(0.0, device=logits.device)
    targets, valid_positions = [], []
    for pos in prio_positions.tolist():
        pid = pkg_id_list[pos]
        uid = teacher_assignment.get(pid)
        if uid is not None and uid in uld_id_to_idx:
            valid_positions.append(pos)
            targets.append(uld_id_to_idx[uid])
    if not targets:
        return torch.tensor(0.0, device=logits.device)
    valid_positions_t = torch.tensor(valid_positions, device=logits.device)
    targets_t = torch.tensor(targets, device=logits.device, dtype=torch.long)
    selected_logits = logits[valid_positions_t][:, :n_ulds]
    return F.cross_entropy(selected_logits, targets_t)


def train_rl_ppo_contrastive(
    model,
    data_dir,
    n_updates              = 200,
    episodes_per_update    = 8,
    ppo_epochs             = 4,
    clip_eps               = 0.2,
    lr                     = RL_LR,
    grad_clip              = RL_GRAD_CLIP,
    entropy_coef           = RL_ENTROPY_COEF,
    hinge_coef             = RL_HINGE_COEF,
    hinge_margin           = RL_HINGE_MARGIN,
    spread_coef            = RL_SPREAD_COEF,
    contrastive_coef       = 8.0,
    contrastive_margin_slope = 3.0,
    imitation_coef         = 0.0,
    lambda_weight_penalty  = RL_LAMBDA_WEIGHT_PENALTY,
    lambda_volume_penalty  = RL_LAMBDA_VOLUME_PENALTY,
    val_every              = 10,
    val_instances           = 40,
    patience                = 12,
    save_path               = None,
    log_path                 = None,
    device                   = DEVICE,
    temperature              = RL_TEMPERATURE,
    max_instances            = None,
    packer                   = None,
    max_safe_pkgs            = None,
    max_safe_ulds            = None,
    k_values_map_dict        = None,
    priority_drop_penalty    = 0.0,
    initial_best_val_cost    = float('inf'),
    seed                     = 0,
):
    if packer is None:
        packer = DEFAULT_PACKER
    max_safe_pkgs = MAX_SAFE_PKGS if max_safe_pkgs is None else max_safe_pkgs
    max_safe_ulds = MAX_SAFE_ULDS if max_safe_ulds is None else max_safe_ulds
    if k_values_map_dict is None:
        k_values_map_dict = {}
    rng = random.Random(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    train_pool = _ppo_mod._load_train_pool(data_dir, k_values_map_dict, max_instances, max_safe_pkgs, max_safe_ulds)
    test_pool  = _load_test_pool(data_dir, k_values_map_dict, max_safe_pkgs, max_safe_ulds)
    train_tags = list(train_pool.keys())

    baseline: dict[float, float] = {}
    baseline_std: dict[float, float] = {}
    k_values_seen = sorted({k for _, _, k in train_pool.values()})
    print(f"Warming up per-K baseline for K values: {k_values_seen}")
    tags_by_k = {k: [t for t in train_tags if train_pool[t][2] == k] for k in k_values_seen}
    with torch.no_grad():
        for k in k_values_seen:
            sample_tags = rng.sample(tags_by_k[k], min(8, len(tags_by_k[k])))
            warm_costs = []
            for tag in sample_tags:
                ulds_df, pkgs_df, _ = train_pool[tag]
                asgn = rl_assign_argmax_safe(model, pkgs_df, ulds_df, device, k,
                                              max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds)
                placements, _ = packer.pack(asgn, pkgs_df, ulds_df)
                c, _, _, _, _, _ = compute_packing_cost(placements, pkgs_df, k)
                warm_costs.append(float(c))
            baseline[k] = float(np.mean(warm_costs))
            baseline_std[k] = float(np.std(warm_costs)) if len(warm_costs) > 1 else max(baseline[k] * 0.1, 1.0)
            baseline_std[k] = max(baseline_std[k], 1.0)
            print(f"  K={k}: baseline={baseline[k]:.1f} +/- {baseline_std[k]:.1f} (from {len(warm_costs)} rollouts)")

    history = []
    best_val_cost    = initial_best_val_cost
    patience_counter = 0
    stop_training    = False

    log_f = None
    writer = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_f = open(log_path, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(['update', 'mean_cost', 'mean_advantage', 'policy_loss', 'entropy_loss',
                          'capacity_loss', 'hinge_loss', 'spread_loss', 'contrastive_loss', 'imitation_loss',
                          'total_loss', 'priority_dropped', 'economy_dropped', 'val_rl_cost', 'best_val_cost'])

    for update in range(1, n_updates + 1):
        model.train()
        batch = []
        for _ in range(episodes_per_update):
            tag = rng.choice(train_tags)
            ulds_df, pkgs_df, k_value = train_pool[tag]
            ep = _collect_rollout(model, pkgs_df, ulds_df, k_value, device, temperature,
                                   packer, priority_drop_penalty)

            prev_baseline = baseline.get(k_value, ep['training_cost'])
            baseline[k_value] = (ep['training_cost'] if k_value not in baseline
                                  else 0.9 * prev_baseline + 0.1 * ep['training_cost'])
            dev = ep['training_cost'] - prev_baseline
            baseline_std[k_value] = (max(abs(dev), 1.0) if k_value not in baseline_std
                                      else max(0.9 * baseline_std[k_value] + 0.1 * abs(dev), 1.0))
            advantage = (prev_baseline - ep['training_cost']) / baseline_std[k_value]

            ep['advantage'] = advantage
            # Cross-K contrastive pairing: same instance, a different K.
            # Tensors only (no rollout/packer needed -- soft_spread_loss is
            # computed directly from raw logits, not an actual assignment).
            paired_k = rng.choice([k for k in K_VALUES if k != k_value])
            ep['paired_k'] = paired_k
            ep['paired_tensors'] = build_tensors(pkgs_df, ulds_df, device, paired_k)
            # Consolidation-imitation target -- static w.r.t. model weights,
            # computed once per episode rather than once per ppo_epoch.
            teacher_assignment, _, _ = _consolidate_priority_by_capacity(pkgs_df, ulds_df)
            ep['teacher_assignment'] = teacher_assignment
            ep['pkg_id_list']        = pkgs_df['Package_ID'].tolist()
            ep['uld_id_to_idx']      = {uid: i for i, uid in enumerate(ulds_df['ULD_ID'].tolist())}
            batch.append(ep)

        last_metrics = {}
        for _ in range(ppo_epochs):
            policy_losses, entropies, weight_pens, volume_pens, hinges, spreads, contrastives, imitations = (
                [], [], [], [], [], [], [], [])
            for ep in batch:
                t = ep['tensors']
                new_lp, entropy, logits = replay_log_probs(
                    model, t['uld_feats'], t['pkg_feats'], t['key_padding_mask'],
                    ep['n_ulds'], t['dim_mask'], t['priority_mask'], t['tightness'], t['k_feat'],
                    ep['n_pkgs'], ep['pkg_weights'], ep['uld_weight_limits'], ep['fixed_actions'],
                    temperature=temperature, pkg_volumes=ep['pkg_volumes'], uld_volumes=ep['uld_volumes'],
                )
                new_lp_sum = new_lp[:ep['n_pkgs']].sum()
                ratio = torch.exp(new_lp_sum - ep['old_lp_sum'])
                adv = float(ep['advantage'])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_losses.append(-torch.min(surr1, surr2))
                dim_mask_i      = t['dim_mask'].squeeze(0)
                priority_mask_i = t['priority_mask'].squeeze(0)
                entropies.append(_economy_only_entropy(logits, priority_mask_i, ep['n_pkgs']))

                w_pen, v_pen = rl_capacity_violation_penalty(
                    logits, ep['n_pkgs'], ep['n_ulds'], ep['pkg_weights'], ep['uld_weight_limits'],
                    pkg_volumes=ep['pkg_volumes'], uld_volumes=ep['uld_volumes'],
                )
                weight_pens.append(w_pen)
                volume_pens.append(v_pen)

                hinges.append(feasibility_hinge_loss(logits, dim_mask_i, priority_mask_i,
                                                      ep['n_pkgs'], ep['n_ulds'], margin=hinge_margin))
                soft_spread_primary = soft_spread_loss_ipr(logits, dim_mask_i, priority_mask_i,
                                                        ep['n_pkgs'], ep['n_ulds'])
                spreads.append(normalize_k(ep['k_value']) * soft_spread_primary)

                # ── Contrastive cross-K term ──────────────────────────────
                pt = ep['paired_tensors']
                paired_logits = model.forward(
                    pt['uld_feats'], pt['pkg_feats'], pt['key_padding_mask'],
                    torch.tensor([ep['n_ulds']], device=device),
                    pt['dim_mask'], pt['priority_mask'], pt['tightness'], pt['k_feat'],
                ).squeeze(0)
                paired_dim_mask_i      = pt['dim_mask'].squeeze(0)
                paired_priority_mask_i = pt['priority_mask'].squeeze(0)
                soft_spread_paired = soft_spread_loss_ipr(paired_logits, paired_dim_mask_i, paired_priority_mask_i,
                                                       ep['n_pkgs'], ep['n_ulds'])

                if ep['k_value'] < ep['paired_k']:
                    lower_spread, higher_spread = soft_spread_primary, soft_spread_paired
                else:
                    lower_spread, higher_spread = soft_spread_paired, soft_spread_primary
                # Margin scales with how far apart the pair is in normalized
                # K-space, instead of a fixed constant. A fixed margin gives
                # ZERO further gradient once any pair clears it (relu(x)=0 for
                # x<0) -- verified this made training stop pushing spread down
                # the moment adjacent-K pairs (e.g. K=3000 vs K=5000, a small
                # normalized gap) were satisfied, well before it reached the
                # true packing floor, and made K=500/1000/3000/5000 collapse
                # to near-identical spread instead of continuing to separate.
                # A distant pair (K=100 vs K=5000) now demands a proportionally
                # larger gap than a close pair (K=3000 vs K=5000), so gradient
                # pressure to differentiate further never fully vanishes across
                # the whole K range.
                gap_k = abs(normalize_k(ep['paired_k']) - normalize_k(ep['k_value']))
                scaled_margin = contrastive_margin_slope * gap_k
                contrastives.append(F.relu(higher_spread - lower_spread + scaled_margin))

                # ── Consolidation-imitation term ──────────────────────────
                # Scaled by normalize_k(k_value): fully applies at high K
                # (where the heuristic's near-optimal FFD consolidation is
                # clearly the right target), fades to ~0 at low K (where the
                # model's own more-spread-but-less-delay tradeoff was found
                # to genuinely beat forced consolidation).
                imit = _consolidation_imitation_loss(
                    logits, priority_mask_i, ep['n_pkgs'], ep['n_ulds'],
                    ep['pkg_id_list'], ep['teacher_assignment'], ep['uld_id_to_idx'],
                )
                imitations.append(normalize_k(ep['k_value']) * imit)

            policy_loss  = torch.stack(policy_losses).mean()
            entropy_loss = -entropy_coef * torch.stack(entropies).mean()
            capacity_loss = (lambda_weight_penalty * torch.stack(weight_pens).mean()
                             + lambda_volume_penalty * torch.stack(volume_pens).mean())
            hinge_loss       = hinge_coef * torch.stack(hinges).mean()
            spread_loss      = spread_coef * torch.stack(spreads).mean()
            contrastive_loss = contrastive_coef * torch.stack(contrastives).mean()
            imitation_loss   = imitation_coef * torch.stack(imitations).mean()
            total_loss   = (policy_loss + entropy_loss + capacity_loss + imitation_loss
                             + hinge_loss + spread_loss + contrastive_loss)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            last_metrics = dict(
                policy_loss=policy_loss.item(), entropy_loss=entropy_loss.item(),
                capacity_loss=capacity_loss.item(), hinge_loss=hinge_loss.item(),
                spread_loss=spread_loss.item(), contrastive_loss=contrastive_loss.item(),
                imitation_loss=imitation_loss.item(),
                total_loss=total_loss.item(),
            )

        mean_cost = float(np.mean([ep['cost'] for ep in batch]))
        mean_adv  = float(np.mean([ep['advantage'] for ep in batch]))
        mean_prio_dropped = float(np.mean([ep['n_priority_dropped'] for ep in batch]))
        mean_econ_dropped = float(np.mean([ep['n_economy_dropped'] for ep in batch]))

        metrics = {
            'update': update,
            'reward/mean_cost': mean_cost,
            'reward/mean_advantage': mean_adv,
            'loss/policy_loss': last_metrics['policy_loss'],
            'loss/entropy_loss': last_metrics['entropy_loss'],
            'loss/capacity_loss': last_metrics['capacity_loss'],
            'loss/hinge_loss': last_metrics['hinge_loss'],
            'loss/spread_loss': last_metrics['spread_loss'],
            'loss/contrastive_loss': last_metrics['contrastive_loss'],
            'loss/imitation_loss': last_metrics['imitation_loss'],
            'loss/total_loss': last_metrics['total_loss'],
            'pack/priority_dropped': mean_prio_dropped,
            'pack/economy_dropped': mean_econ_dropped,
        }

        is_eval = (update % val_every == 0) or (update == n_updates)
        if is_eval:
            model.eval()
            val_tags = rng.sample(list(test_pool.keys()), min(val_instances, len(test_pool)))
            val_costs, val_prio_dropped = [], 0
            with torch.no_grad():
                for tag in val_tags:
                    ulds_df, pkgs_df, k_value = test_pool[tag]
                    asgn = rl_assign_argmax_safe(model, pkgs_df, ulds_df, device, k_value,
                                                  max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds)
                    placements, _ = packer.pack(asgn, pkgs_df, ulds_df)
                    c, _, _, _, unplaced_prio, _ = compute_packing_cost(placements, pkgs_df, k_value)
                    val_costs.append(float(c) + priority_drop_penalty * len(unplaced_prio))
                    val_prio_dropped += len(unplaced_prio)
            val_rl_cost = float(np.mean(val_costs))
            metrics['val/rl_cost'] = val_rl_cost

            if val_rl_cost < best_val_cost:
                best_val_cost = val_rl_cost
                patience_counter = 0
                if save_path:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    torch.save({
                        'epoch': update,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rl_cost': val_rl_cost,
                        'val_rl_cost_penalized': best_val_cost,
                        'val_il_cost': float('nan'),
                    }, save_path)
                    print(f"    Checkpoint saved (best val_rl_cost = {best_val_cost:.1f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_training = True

            _print_update(update, n_updates, metrics, patience_counter, patience, is_eval)
            print(f"    contrastive_loss : {last_metrics['contrastive_loss']:>8.4f}    "
                  f"imitation_loss : {last_metrics['imitation_loss']:>8.4f}")
            if stop_training:
                print(f"\n  [Early Stop] No val improvement for {patience} checks.")

        if writer:
            writer.writerow([update, mean_cost, mean_adv, last_metrics['policy_loss'],
                              last_metrics['entropy_loss'], last_metrics['capacity_loss'],
                              last_metrics['hinge_loss'], last_metrics['spread_loss'],
                              last_metrics['contrastive_loss'], last_metrics['imitation_loss'],
                              last_metrics['total_loss'],
                              mean_prio_dropped, mean_econ_dropped,
                              metrics.get('val/rl_cost', ''), best_val_cost])
            log_f.flush()

        history.append(metrics)
        if stop_training:
            break

    if log_f:
        log_f.close()

    print(f"\nPPO training complete. Best val cost: {best_val_cost:.1f}")
    return pd.DataFrame(history)
