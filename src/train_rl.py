import os
import json
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import (
    MAX_N_ULDS, MAX_N_PKGS, N_ULD_CLASSES,
    RL_LR, RL_EPOCHS, RL_GRAD_CLIP, RL_ENTROPY_COEF, RL_KL_COEF,
    RL_EVAL_EVERY, RL_PATIENCE, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    MAX_SAFE_PKGS, MAX_SAFE_ULDS, DEVICE,
)
from .data_utils import (
    build_tensors, chunk_dataframe, needs_chunking,
    il_sample_assignment, il_sample_assignment_safe, actions_to_assignment,
)
from .reward import compute_packing_cost, rl_capacity_violation_penalty
from .packer import DEFAULT_PACKER


# ── Chunk-safe stochastic policy rollout ─────────────────────────────────────

def rl_sample_actions_safe(model, packages_df, ulds_df, device,
                            temperature=1.0, max_pkgs=None, max_ulds=None):
    """
    Stochastic assignment for an instance of any size, chunking when needed.

    Log-probs and entropy are summed across chunks: the joint log-prob of
    independent per-package decisions is the sum of per-decision log-probs,
    so REINFORCE works correctly on the stitched result.

    Capacity limits carry over between package chunks by passing reduced ULD
    limits (limit minus already-used) into each subsequent chunk.

    Returns:
        assignment     : {Package_ID: ULD_ID | 'NONE'}
        log_prob_sum   : scalar tensor (differentiable)
        entropy_sum    : scalar tensor (differentiable)
        weight_used    : {ULD_ID: float}
        volume_used    : {ULD_ID: float}
        weight_penalty : scalar tensor (differentiable)
        volume_penalty : scalar tensor (differentiable)
    """
    device   = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    full_assignment    = {}
    weight_used_by_id  = {}
    volume_used_by_id  = {}
    log_prob_terms     = []
    entropy_terms      = []
    weight_pen_terms   = []
    volume_pen_terms   = []

    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks        = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break

        n_ulds_here        = len(uld_chunk)
        uld_ids_here       = uld_chunk['ULD_ID'].tolist()
        base_weight_limits = uld_chunk['Weight_Limit'].tolist()
        base_volumes       = (uld_chunk['Length'] * uld_chunk['Width'] * uld_chunk['Height']).tolist()
        weight_used_group  = [0.0] * n_ulds_here
        volume_used_group  = [0.0] * n_ulds_here

        for pkg_chunk in chunk_dataframe(remaining_pkgs_df, max_pkgs):
            n_pkgs_here      = len(pkg_chunk)
            tensors          = build_tensors(pkg_chunk, uld_chunk, device)
            pkg_weights_here = pkg_chunk['Weight'].tolist()
            pkg_volumes_here = (pkg_chunk['Length'] * pkg_chunk['Width'] * pkg_chunk['Height']).tolist()

            # Reduce limits by what earlier chunks in this ULD group already used
            remaining_weight_limits = [base_weight_limits[j] - weight_used_group[j]
                                       for j in range(n_ulds_here)]
            remaining_volumes       = [base_volumes[j] - volume_used_group[j]
                                       for j in range(n_ulds_here)]

            actions, log_probs, entropy, raw_logits, w_used, v_used = model.sample_actions(
                tensors['uld_feats'], tensors['pkg_feats'],
                tensors['key_padding_mask'], n_ulds_here,
                tensors['dim_mask'], tensors['priority_mask'],
                tensors['tightness'],
                n_pkgs_here, pkg_weights_here, remaining_weight_limits, temperature,
                pkg_volumes=pkg_volumes_here, uld_volumes=remaining_volumes,
            )

            w_pen, v_pen = rl_capacity_violation_penalty(
                raw_logits, n_pkgs_here, n_ulds_here,
                pkg_weights_here, remaining_weight_limits,
                pkg_volumes=pkg_volumes_here, uld_volumes=remaining_volumes,
            )
            weight_pen_terms.append((w_pen, n_ulds_here))
            volume_pen_terms.append((v_pen, n_ulds_here))

            full_assignment.update(actions_to_assignment(actions, n_pkgs_here, pkg_chunk, uld_chunk))
            log_prob_terms.append(log_probs[:n_pkgs_here].sum())
            entropy_terms.append(entropy)

            for j in range(n_ulds_here):
                weight_used_group[j] += w_used[j]
                volume_used_group[j] += v_used[j]

        for j, uid in enumerate(uld_ids_here):
            weight_used_by_id[uid] = weight_used_group[j]
            volume_used_by_id[uid] = volume_used_group[j]

        unresolved_ids = {pid for pid in remaining_pkgs_df['Package_ID']
                          if full_assignment.get(pid, 'NONE') == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(unresolved_ids)
        ].reset_index(drop=True)

    log_prob_sum = (torch.stack(log_prob_terms).sum() if log_prob_terms
                    else torch.tensor(0.0, device=device))
    entropy_sum  = (torch.stack(entropy_terms).sum() if entropy_terms
                    else torch.tensor(0.0, device=device))

    # Weighted mean across chunks so the result is "mean over all real ULDs"
    if weight_pen_terms:
        total_ulds     = sum(n for _, n in weight_pen_terms)
        weight_penalty = sum(p * n for p, n in weight_pen_terms) / max(total_ulds, 1)
        volume_penalty = sum(p * n for p, n in volume_pen_terms) / max(total_ulds, 1)
    else:
        weight_penalty = torch.tensor(0.0, device=device)
        volume_penalty = torch.tensor(0.0, device=device)

    return (full_assignment, log_prob_sum, entropy_sum,
            weight_used_by_id, volume_used_by_id,
            weight_penalty, volume_penalty)


# ── Chunk-safe deterministic inference ───────────────────────────────────────

def rl_assign_argmax_safe(model, packages_df, ulds_df, device,
                           max_pkgs=None, max_ulds=None):
    """
    Deterministic (argmax) assignment for an instance of any size.
    Used for validation and inference; mirrors the chunking of rl_sample_actions_safe().
    """
    device   = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    full_assignment   = {}
    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks        = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    model.eval()
    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break

        n_ulds_here        = len(uld_chunk)
        uld_ids_here       = uld_chunk['ULD_ID'].tolist()
        base_weight_limits = uld_chunk['Weight_Limit'].tolist()
        base_volumes       = (uld_chunk['Length'] * uld_chunk['Width'] * uld_chunk['Height']).tolist()
        weight_used_group  = [0.0] * n_ulds_here
        volume_used_group  = [0.0] * n_ulds_here

        for pkg_chunk in chunk_dataframe(remaining_pkgs_df, max_pkgs):
            n_pkgs_here = len(pkg_chunk)
            tensors     = build_tensors(pkg_chunk, uld_chunk, device)

            with torch.no_grad():
                logits = model.forward(
                    tensors['uld_feats'], tensors['pkg_feats'],
                    tensors['key_padding_mask'],
                    torch.tensor([n_ulds_here], device=device),
                    tensors['dim_mask'], tensors['priority_mask'],
                    tensors['tightness'],
                ).squeeze(0)[:n_pkgs_here]

            pkg_records = pkg_chunk.to_dict('records')
            for i in range(n_pkgs_here):
                pkg         = pkg_records[i]
                pid         = pkg['Package_ID']
                pw_         = pkg['Weight']
                pvol        = pkg['Length'] * pkg['Width'] * pkg['Height']
                is_priority = (str(pkg['Type']).upper() == 'PRIORITY')

                lg = logits[i].clone()
                for j in range(n_ulds_here):
                    if weight_used_group[j] + pw_ > base_weight_limits[j] + 1e-6:
                        lg[j] = -1e9
                    if volume_used_group[j] + pvol > base_volumes[j] + 1e-6:
                        lg[j] = -1e9

                all_blocked = all(lg[j].item() <= -1e8 for j in range(n_ulds_here))
                if is_priority and all_blocked:
                    def _relative_overflow(j):
                        wt_over  = max(0.0, (weight_used_group[j] + pw_ - base_weight_limits[j])
                                       / max(base_weight_limits[j], 1e-6))
                        vol_over = max(0.0, (volume_used_group[j] + pvol - base_volumes[j])
                                       / max(base_volumes[j], 1e-6))
                        return max(wt_over, vol_over)
                    action = min(range(n_ulds_here), key=_relative_overflow)
                elif all_blocked:
                    action = MAX_N_ULDS
                else:
                    action = int(torch.argmax(lg).item())

                if action == MAX_N_ULDS or action >= n_ulds_here:
                    full_assignment[pid] = 'NONE'
                else:
                    full_assignment[pid]          = uld_ids_here[action]
                    weight_used_group[action]     += pw_
                    volume_used_group[action]     += pvol

        unresolved_ids = {pid for pid in remaining_pkgs_df['Package_ID']
                          if full_assignment.get(pid, 'NONE') == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(unresolved_ids)
        ].reset_index(drop=True)

    return full_assignment


# ── Training loop ─────────────────────────────────────────────────────────────

def _print_metrics(epoch, n_epochs, metrics, patience_counter, patience, is_eval):
    bar = "█" * int(30 * epoch / max(n_epochs, 1))
    bar = f"[{bar:<30}] {epoch}/{n_epochs}"
    pc  = f"  patience {patience_counter}/{patience}"
    print(f"\n{'─'*72}")
    print(f"  Epoch {epoch:>3d}  {bar}{pc}")
    print(f"{'─'*72}")
    print(f"  REWARD")
    print(f"    mean_cost      : {metrics['reward/mean_cost']:>10.2f}")
    print(f"    il_baseline    : {metrics['reward/il_baseline']:>10.2f}")
    print(f"    advantage      : {metrics['reward/advantage']:>+10.4f}")
    print(f"    success_rate   : {metrics['reward/success_rate']:>9.1%}")
    print(f"  LOSS  (lr={metrics.get('lr', 0):.2e})")
    print(f"    policy_loss    : {metrics['loss/policy_loss']:>10.5f}")
    print(f"    entropy_loss   : {metrics['loss/entropy_loss']:>10.5f}")
    print(f"    value_loss     : {metrics['loss/value_loss']:>10.5f}")
    print(f"    capacity_loss  : {metrics['loss/capacity_loss']:>10.5f}")
    print(f"    total_loss     : {metrics['loss/total_loss']:>10.5f}")
    print(f"  CAPACITY (raw, pre-mask logits)")
    print(f"    weight_pen     : {metrics.get('cap/weight_pen', 0):>10.5f}")
    print(f"    volume_pen     : {metrics.get('cap/volume_pen', 0):>10.5f}")
    if is_eval:
        print(f"  VALIDATION")
        print(f"    val_rl_cost    : {metrics.get('val/rl_cost', float('nan')):>10.2f}")
        print(f"    val_il_cost    : {metrics.get('val/il_cost', float('nan')):>10.2f}")
        print(f"    val_cost_diff  : {metrics.get('val/cost_diff', float('nan')):>+10.2f}")
        print(f"    val_better_pct : {metrics.get('val/better_pct', float('nan')):>9.1%}")
    print(f"  PACKING")
    print(f"    priority_dropped : {metrics.get('pack/priority_dropped', 0):>8.2f}")
    print(f"    economy_dropped  : {metrics.get('pack/economy_dropped',  0):>8.2f}")
    print(f"    wt_used_pct      : {metrics.get('pack/wt_used_pct',      0):>8.1%}")
    print(f"    vol_used_pct     : {metrics.get('pack/vol_used_pct',      0):>8.1%}")
    print(f"{'─'*72}")


def train_rl(
    model,
    il_model,
    data_dir,
    n_epochs              = RL_EPOCHS,
    lr                    = RL_LR,
    grad_clip             = RL_GRAD_CLIP,
    entropy_coef          = RL_ENTROPY_COEF,
    kl_coef               = RL_KL_COEF,
    eval_every            = RL_EVAL_EVERY,
    patience              = RL_PATIENCE,
    save_path             = None,
    log_path              = None,
    il_baseline_cache_path = None,
    device                = DEVICE,
    temperature           = RL_TEMPERATURE,
    max_instances         = None,
    n_il_samples          = 4,
    packer                = None,
    max_safe_pkgs         = None,
    max_safe_ulds         = None,
    lambda_weight_penalty = RL_LAMBDA_WEIGHT_PENALTY,
    lambda_volume_penalty = RL_LAMBDA_VOLUME_PENALTY,
    k_values_map_dict     = None,
):
    if packer is None:
        packer = DEFAULT_PACKER
    max_safe_pkgs = MAX_SAFE_PKGS if max_safe_pkgs is None else max_safe_pkgs
    max_safe_ulds = MAX_SAFE_ULDS if max_safe_ulds is None else max_safe_ulds
    if k_values_map_dict is None:
        k_values_map_dict = {}

    model    = model.to(device)
    il_model = il_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    train_meta = pd.read_csv(os.path.join(data_dir, 'synthetic_train', 'metadata.csv'))
    test_meta  = pd.read_csv(os.path.join(data_dir, 'synthetic_test',  'metadata.csv'))
    if max_instances is not None:
        train_meta = train_meta.head(max_instances).reset_index(drop=True)

    print("Pre-loading training CSVs...", end=" ", flush=True)
    train_cache, test_cache = {}, {}
    for _, row in train_meta.iterrows():
        tag    = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv')
        if os.path.exists(u_path) and os.path.exists(p_path):
            train_cache[tag] = (pd.read_csv(u_path), pd.read_csv(p_path))
    for _, row in test_meta.iterrows():
        tag    = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_packages.csv')
        if os.path.exists(u_path) and os.path.exists(p_path):
            test_cache[tag] = (pd.read_csv(u_path), pd.read_csv(p_path))
    print(f"loaded {len(train_cache)} train / {len(test_cache)} test instances.")

    oversized = [t for t, (u, p) in train_cache.items()
                 if needs_chunking(p, u, max_safe_pkgs, max_safe_ulds)]
    if oversized:
        print(f"  {len(oversized)} train instances exceed capacity — will be chunked.")

    # ── Pre-compute IL baselines ──────────────────────────────────────────────
    il_baseline_cache = {}
    il_logits_cache   = {}
    tensors_cache     = {}

    if il_baseline_cache_path and os.path.exists(il_baseline_cache_path):
        print(f"Loading IL baselines from {il_baseline_cache_path}...", end=" ", flush=True)
        try:
            with open(il_baseline_cache_path, 'rb') as f:
                loaded = pickle.load(f)
            il_baseline_cache = loaded['il_baseline_cache']
            il_logits_cache   = loaded['il_logits_cache']
            tensors_cache     = loaded['tensors_cache']
            print("Done.")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing.")
            il_baseline_cache, il_logits_cache, tensors_cache = {}, {}, {}
    else:
        print("Pre-computing IL baselines (one-time)...", flush=True)
        for tag, (ulds_df, pkgs_df) in tqdm(train_cache.items(), desc="  IL baseline"):
            n_ulds  = len(ulds_df)
            n_pkgs  = len(pkgs_df)
            k_value = k_values_map_dict.get(tag, 0)

            if not needs_chunking(pkgs_df, ulds_df, max_safe_pkgs, max_safe_ulds):
                tensors = build_tensors(pkgs_df, ulds_df, device)
                tensors_cache[tag] = {k: v.cpu() if hasattr(v, 'cpu') else v
                                      for k, v in tensors.items()}
                il_asgns, il_logits, _, _ = il_sample_assignment(
                    il_model, tensors, n_ulds, n_pkgs,
                    pkgs_df, ulds_df, n_samples=n_il_samples, temperature=1.0,
                )
                il_logits_cache[tag] = il_logits.cpu()
            else:
                il_asgns = il_sample_assignment_safe(
                    il_model, pkgs_df, ulds_df, device,
                    n_samples=n_il_samples, temperature=1.0,
                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                )

            il_costs = []
            for asgn in il_asgns:
                pl, _ = packer.pack(asgn, pkgs_df, ulds_df)
                c, _, _, _, _, _ = compute_packing_cost(pl, pkgs_df, k_value)
                il_costs.append(float(c))
            il_baseline_cache[tag] = float(np.mean(il_costs))

        print(f"  Done. Mean IL baseline: {np.mean(list(il_baseline_cache.values())):.1f}")

        if il_baseline_cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(il_baseline_cache_path)), exist_ok=True)
                with open(il_baseline_cache_path, 'wb') as f:
                    pickle.dump({'il_baseline_cache': il_baseline_cache,
                                 'il_logits_cache':   il_logits_cache,
                                 'tensors_cache':     tensors_cache}, f)
                print(f"IL baselines saved to {il_baseline_cache_path}")
            except Exception as e:
                print(f"Error saving IL baseline cache: {e}")

    history          = []
    best_val_cost    = float('inf')
    patience_counter = 0
    stop_training    = False
    epoch_bar        = tqdm(range(n_epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        acc = {k: [] for k in [
            'policy_loss', 'entropy_loss', 'value_loss', 'capacity_loss', 'total_loss',
            'weight_pen', 'volume_pen', 'rl_cost', 'il_baseline', 'advantage', 'success',
            'n_priority_dropped', 'n_economy_dropped', 'wt_used_pct', 'vol_used_pct', 'lr',
        ]}

        idxs = list(train_cache.keys())
        random.shuffle(idxs)

        for tag in idxs:
            ulds_df, pkgs_df = train_cache[tag]
            n_ulds, n_pkgs   = len(ulds_df), len(pkgs_df)
            il_baseline      = il_baseline_cache[tag]
            in_capacity      = tag in tensors_cache
            k_value          = k_values_map_dict.get(tag, 0)

            if in_capacity:
                tensors = {k: v.to(device) if hasattr(v, 'to') else v
                           for k, v in tensors_cache[tag].items()}
                pkg_weights       = pkgs_df['Weight'].tolist()
                uld_weight_limits = ulds_df['Weight_Limit'].tolist()
                pkg_volumes       = (pkgs_df['Length'] * pkgs_df['Width'] * pkgs_df['Height']).tolist()
                uld_volumes       = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist()
                il_logits         = il_logits_cache[tag].to(device)

                actions, log_probs, entropy, rl_logits, wt_used, vol_used = model.sample_actions(
                    tensors['uld_feats'], tensors['pkg_feats'],
                    tensors['key_padding_mask'], n_ulds,
                    tensors['dim_mask'], tensors['priority_mask'],
                    tensors['tightness'],
                    n_pkgs, pkg_weights, uld_weight_limits, temperature,
                    pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
                )
                rl_assignment = actions_to_assignment(actions, n_pkgs, pkgs_df, ulds_df)
                lp_sum        = log_probs[:n_pkgs].sum()
                entropy_term  = entropy

                weight_pen, volume_pen = rl_capacity_violation_penalty(
                    rl_logits, n_pkgs, n_ulds,
                    pkg_weights, uld_weight_limits,
                    pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
                )

                rl_log_probs_all = F.log_softmax(rl_logits[:n_pkgs], dim=-1)
                il_probs         = F.softmax(il_logits[:n_pkgs].detach(), dim=-1)
                kl_term = (F.kl_div(rl_log_probs_all, il_probs, reduction='batchmean')
                           if kl_coef != 0.0 else torch.tensor(0.0, device=device))

                mean_wt_pct  = float(sum(wt_used[j] / max(uld_weight_limits[j], 1)
                                         for j in range(n_ulds)) / n_ulds)
                mean_vol_pct = float(sum(vol_used[j] / max(uld_volumes[j], 1)
                                         for j in range(n_ulds)) / n_ulds)
            else:
                (rl_assignment, lp_sum, entropy_term, wt_used_dict, vol_used_dict,
                 weight_pen, volume_pen) = rl_sample_actions_safe(
                    model, pkgs_df, ulds_df, device,
                    temperature=temperature,
                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                )
                kl_term            = torch.tensor(0.0, device=device)
                uld_wt_map         = dict(zip(ulds_df['ULD_ID'], ulds_df['Weight_Limit']))
                uld_vol_map        = dict(zip(ulds_df['ULD_ID'],
                                              ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']))
                mean_wt_pct  = float(np.mean([wt_used_dict.get(uid, 0.0) / max(uld_wt_map[uid], 1)
                                              for uid in ulds_df['ULD_ID']])) if len(ulds_df) else 0.0
                mean_vol_pct = float(np.mean([vol_used_dict.get(uid, 0.0) / max(uld_vol_map[uid], 1)
                                              for uid in ulds_df['ULD_ID']])) if len(ulds_df) else 0.0

            rl_placements, _ = packer.pack(rl_assignment, pkgs_df, ulds_df)
            rl_cost, _, _, _, unplaced_prio, unplaced_eco = compute_packing_cost(
                rl_placements, pkgs_df, k_value)

            advantage    = il_baseline - float(rl_cost)
            policy_loss  = -advantage * lp_sum
            entropy_loss = -entropy_coef * entropy_term
            value_loss   = kl_coef * kl_term
            capacity_loss = (lambda_weight_penalty * weight_pen
                             + lambda_volume_penalty * volume_pen)
            total_loss   = policy_loss + entropy_loss + value_loss + capacity_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            acc['policy_loss'].append(policy_loss.item())
            acc['entropy_loss'].append(entropy_loss.item())
            acc['value_loss'].append(value_loss.item() if torch.is_tensor(value_loss) else float(value_loss))
            acc['capacity_loss'].append(capacity_loss.item() if torch.is_tensor(capacity_loss) else float(capacity_loss))
            acc['weight_pen'].append(weight_pen.item() if torch.is_tensor(weight_pen) else float(weight_pen))
            acc['volume_pen'].append(volume_pen.item() if torch.is_tensor(volume_pen) else float(volume_pen))
            acc['total_loss'].append(total_loss.item())
            acc['rl_cost'].append(float(rl_cost))
            acc['il_baseline'].append(il_baseline)
            acc['advantage'].append(advantage)
            acc['success'].append(1.0 if advantage > 0 else 0.0)
            acc['n_priority_dropped'].append(len(unplaced_prio))
            acc['n_economy_dropped'].append(len(unplaced_eco))
            acc['wt_used_pct'].append(mean_wt_pct)
            acc['vol_used_pct'].append(mean_vol_pct)
            acc['lr'].append(optimizer.param_groups[0]['lr'])

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0

        metrics = {
            'epoch':                 epoch + 1,
            'reward/mean_cost':      _mean(acc['rl_cost']),
            'reward/il_baseline':    _mean(acc['il_baseline']),
            'reward/advantage':      _mean(acc['advantage']),
            'reward/success_rate':   _mean(acc['success']),
            'loss/policy_loss':      _mean(acc['policy_loss']),
            'loss/entropy_loss':     _mean(acc['entropy_loss']),
            'loss/value_loss':       _mean(acc['value_loss']),
            'loss/capacity_loss':    _mean(acc['capacity_loss']),
            'loss/total_loss':       _mean(acc['total_loss']),
            'cap/weight_pen':        _mean(acc['weight_pen']),
            'cap/volume_pen':        _mean(acc['volume_pen']),
            'pack/priority_dropped': _mean(acc['n_priority_dropped']),
            'pack/economy_dropped':  _mean(acc['n_economy_dropped']),
            'pack/wt_used_pct':      _mean(acc['wt_used_pct']),
            'pack/vol_used_pct':     _mean(acc['vol_used_pct']),
            'lr':                    _mean(acc['lr']),
        }

        # ── Validation ────────────────────────────────────────────────────────
        is_eval = (epoch + 1) % eval_every == 0
        if is_eval:
            model.eval()
            rl_costs_eval, il_costs_eval = [], []

            for tag, (u, p) in test_cache.items():
                n_u, n_p = len(u), len(p)
                k_value  = k_values_map_dict.get(tag, 0)

                if not needs_chunking(p, u, max_safe_pkgs, max_safe_ulds):
                    t = build_tensors(p, u, device)
                    with torch.no_grad():
                        logits = model.forward(
                            t['uld_feats'], t['pkg_feats'],
                            t['key_padding_mask'],
                            torch.tensor([n_u], device=device),
                            t['dim_mask'], t['priority_mask'],
                            t['tightness'],
                        ).squeeze(0)
                        acts = logits[:n_p].argmax(dim=-1)
                    rl_a    = actions_to_assignment(acts, n_p, p, u)
                    il_asgns, _, _, _ = il_sample_assignment(
                        il_model, t, n_u, n_p, p, u,
                        n_samples=n_il_samples, temperature=1.0,
                    )
                else:
                    rl_a    = rl_assign_argmax_safe(model, p, u, device,
                                                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds)
                    il_asgns = il_sample_assignment_safe(
                        il_model, p, u, device,
                        n_samples=n_il_samples, temperature=1.0,
                        max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                    )

                rl_p, _ = packer.pack(rl_a, p, u)
                rl_c, _, _, _, _, _ = compute_packing_cost(rl_p, p, k_value)

                il_sc = []
                for ia in il_asgns:
                    ip, _ = packer.pack(ia, p, u)
                    ic, _, _, _, _, _ = compute_packing_cost(ip, p, k_value)
                    il_sc.append(float(ic))

                rl_costs_eval.append(float(rl_c))
                il_costs_eval.append(float(np.mean(il_sc)))

            val_rl     = _mean(rl_costs_eval)
            val_il     = _mean(il_costs_eval)
            better_pct = (sum(r < il for r, il in zip(rl_costs_eval, il_costs_eval))
                          / max(len(rl_costs_eval), 1))
            metrics.update({
                'val/rl_cost':    val_rl,
                'val/il_cost':    val_il,
                'val/cost_diff':  val_rl - val_il,
                'val/better_pct': better_pct,
            })

            if val_rl < best_val_cost:
                best_val_cost    = val_rl
                patience_counter = 0
                metrics['checkpoint'] = True
                if save_path:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    torch.save({
                        'epoch':                epoch + 1,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rl_cost':          best_val_cost,
                        'val_il_cost':          val_il,
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_training = True

            # Save clustering JSON for this eval epoch
            if save_path:
                epoch_dir = os.path.join(os.path.dirname(save_path), f'outputs/epoch_{epoch+1:03d}')
                os.makedirs(epoch_dir, exist_ok=True)
                clustering_agg = {}
                for val_tag, (val_u, val_p) in test_cache.items():
                    val_n_u, val_n_p = len(val_u), len(val_p)
                    if not needs_chunking(val_p, val_u, max_safe_pkgs, max_safe_ulds):
                        val_t = build_tensors(val_p, val_u, device)
                        with torch.no_grad():
                            val_logits = model.forward(
                                val_t['uld_feats'], val_t['pkg_feats'],
                                val_t['key_padding_mask'],
                                torch.tensor([val_n_u], device=device),
                                val_t['dim_mask'], val_t['priority_mask'],
                                val_t['tightness'],
                            ).squeeze(0)
                            val_acts = val_logits[:val_n_p].argmax(dim=-1)
                        clustering_agg[val_tag] = actions_to_assignment(
                            val_acts, val_n_p, val_p, val_u)
                    else:
                        clustering_agg[val_tag] = rl_assign_argmax_safe(
                            model, val_p, val_u, device,
                            max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                        )
                with open(os.path.join(epoch_dir, 'clustering.json'), 'w') as f:
                    json.dump(clustering_agg, f, indent=2)

        if is_eval:
            _print_metrics(epoch + 1, n_epochs, metrics, patience_counter, patience, is_eval)
            if metrics.get('checkpoint'):
                print(f"    Checkpoint saved  (best val_rl_cost = {best_val_cost:.1f})")
            if stop_training:
                print(f"\n  [Early Stop] No val improvement for {patience} epochs.")

        epoch_bar.set_postfix(
            rl=f"{metrics['reward/mean_cost']:.0f}",
            il=f"{metrics['reward/il_baseline']:.0f}",
            succ=f"{metrics['reward/success_rate']:.0%}",
            p_loss=f"{metrics['loss/policy_loss']:.3f}",
            cap=f"{metrics['loss/capacity_loss']:.3f}",
        )

        history.append(metrics)
        if stop_training:
            break

    epoch_bar.close()

    df = pd.DataFrame(history)
    if log_path:
        df.to_csv(log_path, index=False)
        print(f"\nLog saved -> {log_path}")
    print(f"\nRL training complete.  Best val cost: {best_val_cost:.1f}")
    return df
