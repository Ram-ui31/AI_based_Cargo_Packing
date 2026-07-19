"""
rl_finetune_grpo.py -- GRPO-style (Group Relative Policy Optimization)
fine-tuning of PackageSetRanker against the REAL, FULL production
CombinedPacker, replacing rl_finetune.py's vanilla REINFORCE (single
sample per step vs. a slow-moving EMA baseline).

Why GRPO fits this problem unusually well: GRPO normally shines when
sampling multiple responses per VARYING prompt and scoring them relative
to each other. Here there is only ONE "prompt" -- the same 400-package
instance, every single step -- which is actually the ideal case for it:
instead of one noisy sample compared against a laggy exponential-average
baseline (rl_finetune.py's approach), sample a GROUP of G orderings from
the CURRENT policy per round, real-evaluate all of them, and score each
one's advantage relative to THAT GROUP'S OWN mean/std -- a fresh, unbiased,
much lower-variance baseline every round, with no separate value network
needed (unlike PPO).

Each round:
    1. One forward pass -> scores (shared across the whole group).
    2. Sample G independent orders via Gumbel-max (different noise draws,
       same underlying scores) -- G real "rollouts" of the current policy.
    3. Real-evaluate each of the G orders (greedy first-fit + full
       CombinedPacker) -- the only ground truth used anywhere in this loop.
    4. reward_i = -cost_i. advantage_i = (reward_i - group_mean) / (group_std + eps)
       -- the GRPO normalization, using THIS round's own group as the
       reference point instead of history from possibly-stale past policies.
    5. loss = -mean_i [advantage_i * log_prob_i] - entropy_coef * entropy.
       One backward pass per round (averaged over the group), Adam step.

Warm-starts from checkpoints/best_ever.pt (30,787, the pre-RL best) fresh
-- NOT from rl_finetune.py's rl_best_ever.pt, since that checkpoint's
weights reflect wherever REINFORCE's single-sample noise happened to drift
them, not necessarily a better starting point than the clean pre-RL best.

Usage:
    python src/rl_finetune_grpo.py --rounds 100 --group-size 6
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ga_cargo_packing')
sys.path.insert(0, GA_CARGO_ROOT)

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

from model import PackageSetRanker
from features import build_package_features, build_global_features, normalize_features
from rl_finetune import parse_input_csv, sample_plackett_luce, greedy_first_fit

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
WARM_START_CKPT = os.path.join(CKPT_DIR, 'best_ever.pt')
GRPO_BEST_CKPT = os.path.join(CKPT_DIR, 'grpo_best_ever.pt')
GRPO_BEST_META = os.path.join(CKPT_DIR, 'grpo_best_ever_meta.json')
GRPO_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'grpo_log.jsonl')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--group-size', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--entropy-coef', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warm-start-ckpt', type=str, default=WARM_START_CKPT,
                    help='defaults to the pre-RL best_ever.pt; pass grpo_best_ever.pt (or another '
                         'run\'s checkpoint) to continue from a previous GRPO run\'s progress instead.')
    args = p.parse_args()
    torch.manual_seed(args.seed)

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)

    ckpt = torch.load(args.warm_start_ckpt, map_location='cpu', weights_only=False)
    feats_np, _, _ = normalize_features(feats_np, ckpt['feat_mean'], ckpt['feat_std'])
    global_feats_np = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats_np, _, _ = normalize_features(global_feats_np.reshape(1, -1), ckpt['gmean'], ckpt['gstd'])

    model = PackageSetRanker(**ckpt['arch'])
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Warm-started GRPO policy from {args.warm_start_ckpt}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full_assignment = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full_assignment.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    pkg_feats_t = torch.tensor(feats_np, dtype=torch.float32)
    global_feats_t = torch.tensor(global_feats_np, dtype=torch.float32).squeeze(0)

    best_ever_cost = float('inf')
    if os.path.exists(GRPO_BEST_META):
        with open(GRPO_BEST_META) as f:
            best_ever_cost = json.load(f)['cost']
        print(f'Resuming: GRPO best-ever so far = {best_ever_cost:,.0f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    G = args.group_size
    for round_idx in range(args.rounds):
        t0 = time.time()
        model.train()
        scores = model(pkg_feats_t.unsqueeze(0), global_feats_t.unsqueeze(0)).squeeze(0)  # (n_items,) shared this round

        orders, log_probs, costs = [], [], []
        for g in range(G):
            order, log_prob = sample_plackett_luce(scores)
            ranked_pids = economy_df.loc[order.detach().numpy(), 'Package_ID'].tolist()
            assignment = greedy_first_fit(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment)
            placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
            cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
                placements, pkgs_df, k_value)
            orders.append(order)
            log_probs.append(log_prob)
            costs.append(cost)

        eval_dt = time.time() - t0
        rewards = torch.tensor([-c for c in costs], dtype=torch.float32)
        group_mean = rewards.mean()
        group_std = rewards.std(unbiased=False)
        advantages = (rewards - group_mean) / (group_std + 1e-6)

        log_probs_t = torch.stack(log_probs)
        probs = F.softmax(scores, dim=0)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()

        loss = -(advantages.detach() * log_probs_t).mean() - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        round_best_cost = min(costs)
        is_best = round_best_cost < best_ever_cost
        if is_best:
            best_ever_cost = round_best_cost
            torch.save({
                'model_state_dict': model.state_dict(), 'arch': ckpt['arch'],
                'feat_mean': ckpt['feat_mean'], 'feat_std': ckpt['feat_std'],
                'gmean': ckpt['gmean'], 'gstd': ckpt['gstd'],
                'avg_uld_volume': avg_uld_volume, 'avg_uld_weight': avg_uld_weight,
            }, GRPO_BEST_CKPT)
            with open(GRPO_BEST_META, 'w') as f:
                json.dump({'cost': round_best_cost, 'round': round_idx}, f)

        with open(GRPO_LOG_PATH, 'a') as f:
            f.write(json.dumps({'round': round_idx, 'costs': costs, 'group_mean_cost': -group_mean.item(),
                                 'group_std': group_std.item(), 'entropy': entropy.item()}) + '\n')

        print(f'[round {round_idx:4d}] costs={[f"{c:,.0f}" for c in costs]}  '
              f'group_mean={-group_mean.item():,.0f}  group_std={group_std.item():,.0f}  '
              f'entropy={entropy.item():.2f}  ({eval_dt:.1f}s)  '
              f'{"** NEW BEST: " + f"{round_best_cost:,.0f}" + " **" if is_best else ""}')

    print(f'\nDone. GRPO best-ever real cost: {best_ever_cost:,.0f}  '
          f'(pre-RL best: 30,475, competitor target: 29,203)')


if __name__ == '__main__':
    main()
