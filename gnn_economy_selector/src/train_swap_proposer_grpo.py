"""
train_swap_proposer_grpo.py -- IL-warm-start + GRPO fine-tune for
SwapProposer.

The supervised (pairwise-ranking) SwapProposer can only ever imitate
patterns present in its training data -- moves tried near an already-
converged solution by the vanilla local search. It has no mechanism to
discover a genuinely new good move the heuristic search never happened to
try. This script addresses that gap directly: warm-start from the
existing supervised checkpoint, then fine-tune with an on-policy GRPO
update, so the model can learn from swaps IT ITSELF chooses to try (real-
evaluated), not just a static pre-collected dataset.

Why GRPO specifically (not REINFORCE or PPO):
  - REINFORCE's single-sample updates are too high-variance here -- each
    real evaluation costs ~15-25s and has genuine MPS floating-point
    non-determinism on top, so we can't afford enough samples per update
    to average out the noise without a baseline.
  - PPO needs a separate critic (value network), an extra network that
    needs its own data and can misestimate with the limited real-eval
    budget available here; its main advantage (reusing one batch of
    rollouts across multiple gradient epochs) matters most when rollouts
    are cheap -- here the opposite is true, real evaluations are the
    expensive bottleneck.
  - GRPO needs no critic: for a given parent state, sample a GROUP of
    candidate swaps, real-evaluate them, and normalize advantage as
    (reward - group_mean) / group_std within that group. This also maps
    directly onto infrastructure already built for the guided search
    (which already generates a pool of candidates per parent) -- GRPO's
    "sample a group, compare within it" is the same operation, repurposed
    to actually train instead of just rank at inference time.

Usage:
    python src/train_swap_proposer_grpo.py --rounds 30 --group-size 8
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

from swap_proposer import SwapProposer
from features import build_package_features, build_global_features, normalize_features

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints', 'rl_ppo_contrastive_v7', 'transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl',
                                    'checkpoints', 'rl_packer', 'placement_policy_density.pt')
RESULTS_DIR = os.path.join(GA_CARGO_ROOT, 'results')
SWAP_PROPOSER_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'swap_proposer.pt')
CKPT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'swap_proposer_grpo.pt')


def parse_input_csv(path):
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f]
    k_value = float(lines[0].strip())
    uld_rows, pkg_rows = [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if parts[0].startswith('U'):
            uid, length, width, height, weight_limit = parts
            uld_rows.append({'ULD_ID': uid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight_Limit': float(weight_limit)})
        else:
            pid, length, width, height, weight, ptype, delay = parts
            delay_cost = 0.0 if delay.strip() == '-' else float(delay)
            pkg_rows.append({'Package_ID': pid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight': float(weight), 'Type': ptype,
                              'Delay_Cost': delay_cost})
    return k_value, pd.DataFrame(uld_rows), pd.DataFrame(pkg_rows)


def greedy_first_fit(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment):
    weight_used = {u: 0.0 for u in uld_lookup}
    volume_used = {u: 0.0 for u in uld_lookup}
    for pid, uid in prio_assignment.items():
        weight_used[uid] += pkg_lookup_all[pid]['Weight']
        volume_used[uid] += pkg_lookup_all[pid]['Length'] * pkg_lookup_all[pid]['Width'] * pkg_lookup_all[pid]['Height']
    assignment = dict(prio_assignment)
    for pid in ranked_pids:
        p = pkg_lookup_all[pid]
        pw, pv = p['Weight'], p['Length'] * p['Width'] * p['Height']
        placed = False
        for uid in uld_lookup:
            cap_w = uld_lookup[uid]['Weight_Limit']
            cap_v = uld_lookup[uid]['Length'] * uld_lookup[uid]['Width'] * uld_lookup[uid]['Height']
            if weight_used[uid] + pw <= cap_w + 1e-6 and volume_used[uid] + pv <= cap_v + 1e-6:
                weight_used[uid] += pw
                volume_used[uid] += pv
                assignment[pid] = uid
                placed = True
                break
        if not placed:
            assignment[pid] = 'NONE'
    return assignment


def swap_candidate(order, i, j):
    order = list(order)
    order[i], order[j] = order[j], order[i]
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--group-size', type=int, default=8,
                    help='number of candidate swaps sampled and real-evaluated per round (the GRPO group)')
    p.add_argument('--pool-multiplier', type=int, default=6,
                    help='cheap candidate pool size = group_size * pool_multiplier, sampled down to '
                         'group_size via the policy\'s own softmax distribution (not argmax) so the '
                         'model can explore, not just exploit its current ranking')
    p.add_argument('--lr', type=float, default=2e-4, help='small LR -- fine-tuning, not training from scratch')
    p.add_argument('--temperature', type=float, default=0.5,
                    help='divides logits before Gumbel-max sampling. <1 sharpens toward the model\'s '
                         'current top picks (more exploitation), >1 flattens toward uniform (more '
                         'exploration). Tuned down from an implicit 1.0 after the first run found '
                         '0 improvements in 12 rounds -- with deltas mostly in the hundreds and logits '
                         'that close together, temp=1.0 sampling was close to random, wasting real-eval '
                         'budget on candidates the model already knew were poor.')
    p.add_argument('--parent-states', type=str, default='ems_v1,ems_v2,guided_ems_v2',
                    help='comma-separated run-names to load beam_state_<name>.json from as DIVERSE '
                         'starting contexts -- fixed after the first run showed 0 improvements in 12 '
                         'rounds fine-tuning around a single repeated context (same neighborhood every '
                         'round is an extremely narrow training signal).')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    economy_pids = list(economy_df['Package_ID'])
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pid_to_i = {pid: i for i, pid in enumerate(economy_pids)}
    n_items = len(economy_pids)

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
        ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ('dblf_ems', HeuristicPacker(strategy='dblf', origin_source='ems')),
    ])

    def real_cost(order):
        assignment, _ = greedy_first_fit(order, pkg_lookup_all, uld_lookup, prio_assignment), None
        placements, _ = packer.pack(assignment, pkgs_df, ulds_df)
        cost, *_ = compute_packing_cost(placements, pkgs_df, k_value)
        return cost

    # Warm start: DIVERSE parent contexts, not one repeated point -- a
    # single fixed context gave the model the same neighborhood 25 times
    # in a row, an extremely narrow training signal. Each run-name's saved
    # beam becomes one independent context the round loop rotates through.
    parents = []
    for run_name in args.parent_states.split(','):
        with open(os.path.join(RESULTS_DIR, f'beam_state_{run_name}.json')) as f:
            state = json.load(f)
        best_entry = min(state['beam'], key=lambda b: b['cost'])
        order = list(best_entry['order'])
        cost = real_cost(order)
        parents.append({'name': run_name, 'order': order, 'cost': cost})
        print(f'Parent context "{run_name}": real cost = {cost:,.0f}')

    # Warm-start the policy from the supervised checkpoint.
    proposer_ckpt = torch.load(SWAP_PROPOSER_CKPT, map_location='cpu', weights_only=False)
    model = SwapProposer()
    model.load_state_dict(proposer_ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
    feats_np, _, _ = normalize_features(feats_np, proposer_ckpt['feat_mean'], proposer_ckpt['feat_std'])
    global_feats_np = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats_np, _, _ = normalize_features(global_feats_np.reshape(1, -1), proposer_ckpt['gmean'], proposer_ckpt['gstd'])
    feats_t = torch.tensor(feats_np, dtype=torch.float32)
    global_feat_single = torch.tensor(global_feats_np, dtype=torch.float32)

    def find_cutoff(order):
        assignment = greedy_first_fit(order, pkg_lookup_all, uld_lookup, prio_assignment)
        for idx, pid in enumerate(order):
            if assignment[pid] == 'NONE':
                return idx
        return len(order)

    history = []
    best_ever_cost = min(p['cost'] for p in parents)
    best_ever_order = list(next(p for p in parents if p['cost'] == best_ever_cost)['order'])
    LOG_PATH = os.path.join(RESULTS_DIR, 'swap_proposer_grpo_log.jsonl')

    for round_idx in range(args.rounds):
        t0 = time.time()
        parent = parents[round_idx % len(parents)]
        parent_order, parent_cost = parent['order'], parent['cost']
        n = len(parent_order)
        cutoff = find_cutoff(parent_order)

        # Generate a cheap candidate pool (same distribution as the guided search).
        pool = []
        for _ in range(args.group_size * args.pool_multiplier):
            if rng.random() < 0.6 and 0 < cutoff < n:
                lo, hi = max(0, cutoff - 15), min(n, cutoff + 15)
                i = rng.integers(lo, cutoff) if cutoff > lo else rng.integers(0, n)
                j = rng.integers(cutoff, hi) if hi > cutoff else rng.integers(0, n)
            else:
                i, j = rng.integers(0, n), rng.integers(0, n)
            if i == j:
                continue
            pool.append((int(i), int(j)))

        idx_a = torch.tensor([pid_to_i[parent_order[i]] for i, j in pool])
        idx_b = torch.tensor([pid_to_i[parent_order[j]] for i, j in pool])
        pos_a = torch.tensor([[i / n] for i, j in pool], dtype=torch.float32)
        pos_b = torch.tensor([[j / n] for i, j in pool], dtype=torch.float32)
        gfeat = global_feat_single.repeat(len(pool), 1)

        # ON-POLICY sampling: softmax over -predicted_delta (higher = more
        # promising), sample group_size WITHOUT replacement via Gumbel-max
        # -- exploration, not argmax, so the model can discover moves its
        # current ranking underrates, not just confirm what it already believes.
        model.eval()
        with torch.no_grad():
            pred_delta = model(feats_t[idx_a], feats_t[idx_b], gfeat, pos_a, pos_b)
        logits = -pred_delta / args.temperature
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        perturbed = logits + gumbel_noise
        group_size = min(args.group_size, len(pool))
        sampled_idx = torch.topk(perturbed, group_size).indices.tolist()
        sampled_pairs = [pool[k] for k in sampled_idx]

        # Real-evaluate the sampled group.
        rewards = []
        for i, j in sampled_pairs:
            child_order = swap_candidate(parent_order, i, j)
            child_cost = real_cost(child_order)
            rewards.append(-(child_cost - parent_cost))  # higher = better (cost reduction)
            if child_cost < parent['cost']:
                parent['cost'] = child_cost
                parent['order'] = child_order
            if child_cost < best_ever_cost:
                best_ever_cost = child_cost
                best_ever_order = list(child_order)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        # GRPO policy update: log-prob of each sampled candidate under the
        # model's current softmax distribution over the FULL pool (not just
        # the sampled group) -- standard importance-respecting REINFORCE
        # target, weighted by group-relative advantage, no critic needed.
        model.train()
        pred_delta_train = model(feats_t[idx_a], feats_t[idx_b], gfeat, pos_a, pos_b)
        # Same temperature as the sampling distribution above -- the
        # log-prob used for the policy gradient must match the
        # distribution candidates were actually drawn from, or the
        # gradient estimate is biased.
        log_probs_full = F.log_softmax(-pred_delta_train / args.temperature, dim=0)
        log_probs_sampled = log_probs_full[sampled_idx]
        policy_loss = -(advantage.detach() * log_probs_sampled).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        dt = time.time() - t0
        mean_reward = rewards_t.mean().item()
        best_in_group = max(rewards)
        history.append({'round': round_idx, 'policy_loss': policy_loss.item(),
                         'mean_reward': mean_reward, 'best_in_group_reward': best_in_group,
                         'best_ever_cost': best_ever_cost})
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(history[-1]) + '\n')

        print(f'[round {round_idx:3d}] policy_loss={policy_loss.item():.4f}  '
              f'mean_reward={mean_reward:+.1f}  best_in_group={best_in_group:+.1f}  '
              f'best_ever_cost={best_ever_cost:,.0f}  ({dt:.1f}s)')

    for parent in parents:
        print(f'Final cost for context "{parent["name"]}": {parent["cost"]:,.0f}')

    os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feat_mean': proposer_ckpt['feat_mean'], 'feat_std': proposer_ckpt['feat_std'],
        'gmean': proposer_ckpt['gmean'], 'gstd': proposer_ckpt['gstd'],
        'warm_start_costs': {p['name']: p['cost'] for p in parents}, 'best_ever_cost': best_ever_cost,
    }, CKPT_OUT)
    print(f'\nSaved GRPO-finetuned checkpoint to {CKPT_OUT}')
    print(f'Best-ever real cost found during fine-tuning: {best_ever_cost:,.0f}')


if __name__ == '__main__':
    main()
