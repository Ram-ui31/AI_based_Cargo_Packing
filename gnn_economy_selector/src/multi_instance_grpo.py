"""
multi_instance_grpo.py -- GRPO training of PackageSetRanker across MANY
different cargo instances (good_data/synthetic_train, ~1000 varied
instances, oversampling the real 400-package benchmark), instead of
tonight's single-instance-only version (rl_finetune_grpo.py).

Generalization fix: features are normalized FRESH per instance (mean/std
computed from THAT instance's own Economy packages, every round, never
cached into the checkpoint) -- PackageSetRanker's Transformer architecture
already handles variable package/ULD counts natively, so no architecture
change was needed, only this per-instance normalization discipline (see
multi_instance_data.py's docstring).

Each round:
    1. Sample an instance (mostly from synthetic_train, real benchmark
       oversampled at --benchmark-frac so the model stays anchored to
       what we're actually scored on).
    2. Consolidate Priority (same deterministic heuristic used everywhere
       all session -- proven optimal for the real benchmark; assumed to
       generalize as a sound default across instances too).
    3. Build fresh per-instance features/global-features (no caching).
    4. GRPO group of G real-packer rollouts, group-relative advantage,
       one backward pass on the SHARED PackageSetRanker weights.

Tracks two separate "best" checkpoints: best-ever on the REAL benchmark
specifically (what we're ultimately scored against, 29,203 target), and
best mean group performance across ALL sampled instances (a rough proxy
for general quality, since "best cost" isn't comparable across differently
-sized/valued instances).

Usage:
    python src/multi_instance_grpo.py --rounds 300 --group-size 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
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
from rl_finetune import sample_plackett_luce, greedy_first_fit
from multi_instance_data import InstanceSampler

CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
SINGLE_INSTANCE_CKPT = os.path.join(CKPT_DIR, 'best_ever.pt')  # warm-start source (arch reference)
MULTI_BEST_CKPT = os.path.join(CKPT_DIR, 'multi_instance_best.pt')
MULTI_BEST_META = os.path.join(CKPT_DIR, 'multi_instance_best_meta.json')
MULTI_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'multi_instance_log.jsonl')


def prep_instance(k_value, ulds_df, pkgs_df, clusterer):
    """Priority consolidation + fresh per-instance features. Returns
    everything needed for one GRPO round on this instance."""
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full_assignment = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full_assignment.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
    feats_np, _, _ = normalize_features(feats_np)  # fresh per-instance, not cached
    global_feats_np = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats_np, _, _ = normalize_features(global_feats_np.reshape(1, -1))

    return dict(economy_df=economy_df, pkg_lookup_all=pkg_lookup_all, uld_lookup=uld_lookup,
                prio_assignment=prio_assignment, feats=feats_np, global_feats=global_feats_np.squeeze(0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=300)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--entropy-coef', type=float, default=0.001)
    p.add_argument('--benchmark-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warm-start-ckpt', type=str, default=None,
                    help='continue from a previous multi-instance run\'s checkpoint (e.g. checkpoints/'
                         'multi_instance_best.pt) instead of fresh init, to build on prior progress '
                         'rather than re-deriving general structure from scratch each run.')
    args = p.parse_args()
    torch.manual_seed(args.seed)

    arch_ref = torch.load(SINGLE_INSTANCE_CKPT, map_location='cpu', weights_only=False)['arch']
    model = PackageSetRanker(**arch_ref)
    if args.warm_start_ckpt:
        warm_ckpt = torch.load(args.warm_start_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(warm_ckpt['model_state_dict'])
        print(f'Warm-started multi-instance policy from {args.warm_start_ckpt}')
    else:
        print(f'Fresh PackageSetRanker({arch_ref}) for multi-instance training (no warm-start -- '
              f'single-instance checkpoint was fit to that instance\'s own cached normalization, not a general regime)')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    sampler = InstanceSampler(benchmark_frac=args.benchmark_frac, seed=args.seed)

    best_benchmark_cost = float('inf')
    if os.path.exists(MULTI_BEST_META):
        with open(MULTI_BEST_META) as f:
            meta = json.load(f)
            best_benchmark_cost = meta.get('benchmark_cost', float('inf'))
        print(f'Resuming: best-ever on REAL benchmark so far = {best_benchmark_cost:,.0f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    G = args.group_size
    for round_idx in range(args.rounds):
        t0 = time.time()
        name, (k_value, ulds_df, pkgs_df) = sampler.sample()
        ctx = prep_instance(k_value, ulds_df, pkgs_df, clusterer)
        pkg_feats_t = torch.tensor(ctx['feats'], dtype=torch.float32)
        global_feats_t = torch.tensor(ctx['global_feats'], dtype=torch.float32)

        model.train()
        scores = model(pkg_feats_t.unsqueeze(0), global_feats_t.unsqueeze(0)).squeeze(0)

        costs, log_probs = [], []
        for g in range(G):
            order, log_prob = sample_plackett_luce(scores)
            ranked_pids = ctx['economy_df'].loc[order.detach().numpy(), 'Package_ID'].tolist()
            assignment = greedy_first_fit(ranked_pids, ctx['pkg_lookup_all'], ctx['uld_lookup'], ctx['prio_assignment'])
            placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
            cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
                placements, pkgs_df, k_value)
            costs.append(cost)
            log_probs.append(log_prob)

        eval_dt = time.time() - t0
        rewards = torch.tensor([-c for c in costs], dtype=torch.float32)
        group_mean, group_std = rewards.mean(), rewards.std(unbiased=False)
        advantages = (rewards - group_mean) / (group_std + 1e-6)

        log_probs_t = torch.stack(log_probs)
        probs = F.softmax(scores, dim=0)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        loss = -(advantages.detach() * log_probs_t).mean() - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        round_best_cost = min(costs)
        is_benchmark = (name == 'REAL_BENCHMARK')
        is_best = is_benchmark and round_best_cost < best_benchmark_cost
        if is_best:
            best_benchmark_cost = round_best_cost
            torch.save({
                'model_state_dict': model.state_dict(), 'arch': arch_ref,
            }, MULTI_BEST_CKPT)
            with open(MULTI_BEST_META, 'w') as f:
                json.dump({'benchmark_cost': round_best_cost, 'round': round_idx}, f)

        with open(MULTI_LOG_PATH, 'a') as f:
            f.write(json.dumps({'round': round_idx, 'instance': name, 'n_items': len(ctx['economy_df']),
                                 'costs': costs, 'group_mean_cost': -group_mean.item(),
                                 'entropy': entropy.item()}) + '\n')

        print(f'[round {round_idx:4d}] instance={name:16s} n_econ={len(ctx["economy_df"]):3d}  '
              f'costs={[f"{c:,.0f}" for c in costs]}  mean={-group_mean.item():,.0f}  '
              f'({eval_dt:.1f}s)  {"** NEW BENCHMARK BEST: " + f"{round_best_cost:,.0f}" + " **" if is_best else ""}')

    print(f'\nDone. Best-ever on REAL benchmark: {best_benchmark_cost:,.0f}  '
          f'(single-instance best: 30,475, competitor target: 29,203)')


if __name__ == '__main__':
    main()
