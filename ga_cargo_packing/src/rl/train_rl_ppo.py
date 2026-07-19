"""
PPO + per-K EMA baseline training loop -- ported from
cargoism/git/model_b(c)/training/train_assignment.py's mechanics, adapted to
this codebase's TransformerClusterer / build_tensors / RLPackerAdapter /
compute_packing_cost.

This is a SEPARATE training path from src/rl/train_rl.py's REINFORCE loop.
It does not modify train_rl.py, model.py, reward.py, or data_utils.py --
only imports from them. If this experiment doesn't beat the REINFORCE
result, the existing pipeline and its checkpoints are completely unaffected.

Key differences from train_rl.py's REINFORCE approach:
  - PPO clipped surrogate objective (multiple gradient steps per batch of
    collected rollouts, replaying the same actions under updated params and
    importance-weighting by the probability ratio) instead of single-step
    REINFORCE.
  - Advantage is normalized by an ONLINE per-K exponential-moving-average
    baseline AND per-K EMA std (updated continuously as training
    progresses), instead of a frozen pre-computed IL-baseline-per-instance.
  - Training proceeds in "updates" (each collecting a small batch of random
    rollouts across instances/K-values) rather than full epochs over the
    whole dataset -- matches model_b's own update-based loop structure.

Kept unchanged from the current best (REINFORCE) pipeline:
  - The dual K-injection architecture (model.py's k_proj + output_head
    K-concat), unchanged.
  - The feasibility hinge loss and K-scaled soft-spread loss (reward.py),
    now with the calibration fix (comparable-magnitude coefficients)
    discovered today.
  - The graft warm-start technique (zero-init k_proj/output_head-K-column
    onto the old strong K-blind IL checkpoint).
  - RLPackerAdapter as the packer, compute_packing_cost as the true cost.

Chunking (instances larger than the model's MAX_N_PKGS/MAX_N_ULDS) is NOT
supported here -- oversized instances are filtered out of the training pool
up front (same simplification model_b itself makes). This keeps the
implementation tractable; ~98% of instances fit without chunking.
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
    feasibility_hinge_loss, soft_spread_loss,
)
from .train_rl import rl_assign_argmax_safe
from .ppo_rollout import replay_log_probs
from .packer import DEFAULT_PACKER


def _print_update(update, n_updates, metrics, patience_counter, patience, is_eval):
    bar = "█" * int(30 * update / max(n_updates, 1))
    bar = f"[{bar:<30}] {update}/{n_updates}"
    print(f"\n{'─'*72}")
    print(f"  Update {update:>4d}  {bar}  patience {patience_counter}/{patience}")
    print(f"{'─'*72}")
    print(f"  REWARD")
    print(f"    mean_cost      : {metrics['reward/mean_cost']:>10.2f}")
    print(f"    mean_advantage : {metrics['reward/mean_advantage']:>+10.4f}")
    print(f"  LOSS")
    print(f"    policy_loss    : {metrics['loss/policy_loss']:>10.5f}")
    print(f"    entropy_loss   : {metrics['loss/entropy_loss']:>10.5f}")
    print(f"    capacity_loss  : {metrics['loss/capacity_loss']:>10.5f}")
    print(f"    hinge_loss     : {metrics['loss/hinge_loss']:>10.5f}")
    print(f"    spread_loss    : {metrics['loss/spread_loss']:>10.5f}")
    print(f"    total_loss     : {metrics['loss/total_loss']:>10.5f}")
    if is_eval:
        print(f"  VALIDATION")
        print(f"    val_rl_cost    : {metrics.get('val/rl_cost', float('nan')):>10.2f}")
    print(f"  PACKING")
    print(f"    priority_dropped : {metrics.get('pack/priority_dropped', 0):>8.2f}")
    print(f"    economy_dropped  : {metrics.get('pack/economy_dropped',  0):>8.2f}")
    print(f"{'─'*72}")


def _load_train_pool(data_dir, k_values_map_dict, max_instances, max_safe_pkgs, max_safe_ulds):
    train_meta = pd.read_csv(os.path.join(data_dir, 'synthetic_train', 'metadata.csv'))
    if max_instances is not None:
        train_meta = train_meta.head(max_instances).reset_index(drop=True)

    pool = {}
    n_skipped = 0
    for _, row in train_meta.iterrows():
        tag = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv')
        if not (os.path.exists(u_path) and os.path.exists(p_path)):
            continue
        ulds_df, pkgs_df = pd.read_csv(u_path), pd.read_csv(p_path)
        if needs_chunking(pkgs_df, ulds_df, max_safe_pkgs, max_safe_ulds):
            n_skipped += 1
            continue
        pool[tag] = (ulds_df, pkgs_df, k_values_map_dict.get(tag, 0))
    print(f"Train pool: {len(pool)} instances ({n_skipped} oversized instances skipped -- "
          f"chunking not supported in the PPO path).")
    return pool


def _load_test_pool(data_dir, k_values_map_dict, max_safe_pkgs, max_safe_ulds):
    test_meta = pd.read_csv(os.path.join(data_dir, 'synthetic_test', 'metadata.csv'))
    pool = {}
    for _, row in test_meta.iterrows():
        tag = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_packages.csv')
        if not (os.path.exists(u_path) and os.path.exists(p_path)):
            continue
        ulds_df, pkgs_df = pd.read_csv(u_path), pd.read_csv(p_path)
        if needs_chunking(pkgs_df, ulds_df, max_safe_pkgs, max_safe_ulds):
            continue
        pool[tag] = (ulds_df, pkgs_df, k_values_map_dict.get(tag, 0))
    return pool


def _collect_rollout(model, pkgs_df, ulds_df, k_value, device, temperature,
                      packer, priority_drop_penalty):
    """One stochastic rollout (no_grad) -- returns everything PPO needs to
    later replay it under updated params, plus its true cost for the
    per-K baseline update."""
    n_ulds, n_pkgs = len(ulds_df), len(pkgs_df)
    tensors = build_tensors(pkgs_df, ulds_df, device, k_value)
    pkg_weights       = pkgs_df['Weight'].tolist()
    uld_weight_limits = ulds_df['Weight_Limit'].tolist()
    pkg_volumes       = (pkgs_df['Length'] * pkgs_df['Width'] * pkgs_df['Height']).tolist()
    uld_volumes       = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist()

    with torch.no_grad():
        actions, log_probs, _, _, _, _ = model.sample_actions(
            tensors['uld_feats'], tensors['pkg_feats'], tensors['key_padding_mask'],
            n_ulds, tensors['dim_mask'], tensors['priority_mask'],
            tensors['tightness'], tensors['k_feat'],
            n_pkgs, pkg_weights, uld_weight_limits, temperature,
            pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
        )
    old_lp_sum = log_probs[:n_pkgs].sum().detach()

    assignment = actions_to_assignment(actions, n_pkgs, pkgs_df, ulds_df)
    placements, _ = packer.pack(assignment, pkgs_df, ulds_df)
    cost, delay_cost, spread_cost, n_prio_ulds, unplaced_prio, unplaced_eco = (
        compute_packing_cost(placements, pkgs_df, k_value)
    )
    training_cost = float(cost) + priority_drop_penalty * len(unplaced_prio)

    return dict(
        tensors=tensors, n_ulds=n_ulds, n_pkgs=n_pkgs,
        pkg_weights=pkg_weights, uld_weight_limits=uld_weight_limits,
        pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
        fixed_actions=actions[:n_pkgs].detach().cpu().tolist(),
        old_lp_sum=old_lp_sum, k_value=k_value,
        cost=float(cost), training_cost=training_cost,
        n_priority_dropped=len(unplaced_prio), n_economy_dropped=len(unplaced_eco),
    )


def train_rl_ppo(
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

    train_pool = _load_train_pool(data_dir, k_values_map_dict, max_instances, max_safe_pkgs, max_safe_ulds)
    test_pool  = _load_test_pool(data_dir, k_values_map_dict, max_safe_pkgs, max_safe_ulds)
    train_tags = list(train_pool.keys())

    # ── Warm up per-K EMA baseline + std from a few greedy rollouts per K ──────
    # An empty baseline dict means the first rollout at each K sets
    # baseline[K] = its own cost (zero advantage, no signal), and the next
    # few rollouts see a very noisy, low-sample-count target -- seeding from
    # real greedy-decode rollouts up front avoids that early instability
    # (mirrors model_b's train_assignment.py).
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
                          'capacity_loss', 'hinge_loss', 'spread_loss', 'total_loss',
                          'priority_dropped', 'economy_dropped', 'val_rl_cost', 'best_val_cost'])

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
            batch.append(ep)

        last_metrics = {}
        for _ in range(ppo_epochs):
            policy_losses, entropies, weight_pens, volume_pens, hinges, spreads = [], [], [], [], [], []
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
                entropies.append(entropy)

                w_pen, v_pen = rl_capacity_violation_penalty(
                    logits, ep['n_pkgs'], ep['n_ulds'], ep['pkg_weights'], ep['uld_weight_limits'],
                    pkg_volumes=ep['pkg_volumes'], uld_volumes=ep['uld_volumes'],
                )
                weight_pens.append(w_pen)
                volume_pens.append(v_pen)

                dim_mask_i      = t['dim_mask'].squeeze(0)
                priority_mask_i = t['priority_mask'].squeeze(0)
                hinges.append(feasibility_hinge_loss(logits, dim_mask_i, priority_mask_i,
                                                      ep['n_pkgs'], ep['n_ulds'], margin=hinge_margin))
                spreads.append(normalize_k(ep['k_value']) * soft_spread_loss(
                    logits, dim_mask_i, priority_mask_i, ep['n_pkgs'], ep['n_ulds']))

            policy_loss  = torch.stack(policy_losses).mean()
            entropy_loss = -entropy_coef * torch.stack(entropies).mean()
            capacity_loss = (lambda_weight_penalty * torch.stack(weight_pens).mean()
                             + lambda_volume_penalty * torch.stack(volume_pens).mean())
            hinge_loss   = hinge_coef * torch.stack(hinges).mean()
            spread_loss  = spread_coef * torch.stack(spreads).mean()
            total_loss   = policy_loss + entropy_loss + capacity_loss + hinge_loss + spread_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            last_metrics = dict(
                policy_loss=policy_loss.item(), entropy_loss=entropy_loss.item(),
                capacity_loss=capacity_loss.item(), hinge_loss=hinge_loss.item(),
                spread_loss=spread_loss.item(), total_loss=total_loss.item(),
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
            if stop_training:
                print(f"\n  [Early Stop] No val improvement for {patience} checks.")

        if writer:
            writer.writerow([update, mean_cost, mean_adv, last_metrics['policy_loss'],
                              last_metrics['entropy_loss'], last_metrics['capacity_loss'],
                              last_metrics['hinge_loss'], last_metrics['spread_loss'],
                              last_metrics['total_loss'], mean_prio_dropped, mean_econ_dropped,
                              metrics.get('val/rl_cost', ''), best_val_cost])
            log_f.flush()

        history.append(metrics)
        if stop_training:
            break

    if log_f:
        log_f.close()

    print(f"\nPPO training complete. Best val cost: {best_val_cost:.1f}")
    return pd.DataFrame(history)
