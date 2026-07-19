"""
rl_finetune.py -- genuine reward-seeking policy-gradient (REINFORCE) fine-
tuning of PackageSetRanker against the REAL, FULL production CombinedPacker
-- not a proxy, and not supervised imitation of fixed demonstrations
(confirmed tonight: imitation is bounded above by whatever's already the
best label in the training set, and even perfectly imitating a single
label doesn't perfectly reproduce its real cost due to a translation gap
between continuous scores and the discrete greedy-sort outcome).

Policy: PackageSetRanker's per-package scores define a Plackett-Luce
distribution over full orderings of the Economy candidate set -- sampling
via the Gumbel-max trick (add i.i.d. Gumbel(0,1) noise to each score, sort
descending) gives an exact sample from that distribution, and the
Plackett-Luce log-probability of the sampled order is differentiable
w.r.t. the model's own (unperturbed) scores. Each step:
    1. Forward pass -> scores (deterministic, differentiable).
    2. Sample an order via Gumbel-max (stochastic, not differentiable --
       treated as the fixed action for REINFORCE, matching how sampling
       works in every policy-gradient method).
    3. Greedy first-fit that order into ULDs (same deterministic mechanism
       as every econ_sort_key), merge with the fixed Priority assignment,
       and evaluate via the REAL, FULL CombinedPacker (~15s per step, per
       tonight's timing discovery -- fast enough for real policy gradient
       at meaningful scale).
    4. Reward = -cost. Advantage = reward - EMA baseline (variance
       reduction, standard REINFORCE practice).
    5. Loss = -advantage.detach() * log_prob(sampled order | scores).
       Backprop, Adam step.

Warm-starts from checkpoints/best_ever.pt (the best real-validated result
across supervised + bootstrap phases, 30,787) rather than random init --
a good starting policy reduces variance and gives RL a running start
instead of re-deriving everything from scratch.

Usage:
    python src/rl_finetune.py --steps 300
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

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
WARM_START_CKPT = os.path.join(CKPT_DIR, 'best_ever.pt')
RL_BEST_CKPT = os.path.join(CKPT_DIR, 'rl_best_ever.pt')
RL_BEST_META = os.path.join(CKPT_DIR, 'rl_best_ever_meta.json')
RL_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'rl_finetune_log.jsonl')


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
            uld_rows.append({
                'ULD_ID': uid, 'Length': float(length), 'Width': float(width),
                'Height': float(height), 'Weight_Limit': float(weight_limit),
            })
        else:
            pid, length, width, height, weight, ptype, delay = parts
            delay_cost = 0.0 if delay.strip() == '-' else float(delay)
            pkg_rows.append({
                'Package_ID': pid, 'Length': float(length), 'Width': float(width),
                'Height': float(height), 'Weight': float(weight), 'Type': ptype,
                'Delay_Cost': delay_cost,
            })
    return k_value, pd.DataFrame(uld_rows), pd.DataFrame(pkg_rows)


def sample_plackett_luce(scores):
    """Gumbel-max sampling of a full permutation from Plackett-Luce(scores).
    Returns (order: LongTensor of indices, log_prob: scalar differentiable
    w.r.t. scores for the SAMPLED order)."""
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
    perturbed = scores + gumbel
    order = torch.argsort(perturbed, descending=True)

    ordered_scores = scores[order]
    # log P(order) = sum_k [ s_{order[k]} - logsumexp(s_{order[k:]}) ]
    reverse_logcumsumexp = torch.flip(
        torch.logcumsumexp(torch.flip(ordered_scores, dims=[0]), dim=0), dims=[0])
    log_prob = (ordered_scores - reverse_logcumsumexp).sum()
    return order, log_prob


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--baseline-decay', type=float, default=0.9)
    p.add_argument('--entropy-coef', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)

    ckpt = torch.load(WARM_START_CKPT, map_location='cpu', weights_only=False)
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
    print(f'Warm-started policy from {WARM_START_CKPT}')

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
    if os.path.exists(RL_BEST_META):
        with open(RL_BEST_META) as f:
            best_ever_cost = json.load(f)['cost']
        print(f'Resuming: RL best-ever so far = {best_ever_cost:,.0f}')
    baseline = None

    os.makedirs(CKPT_DIR, exist_ok=True)
    for step in range(args.steps):
        t0 = time.time()
        model.train()
        scores = model(pkg_feats_t.unsqueeze(0), global_feats_t.unsqueeze(0)).squeeze(0)  # (n_items,)

        order, log_prob = sample_plackett_luce(scores)
        ranked_pids = economy_df.loc[order.detach().numpy(), 'Package_ID'].tolist()

        assignment = greedy_first_fit(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment)
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        eval_dt = time.time() - t0

        reward = -cost
        if baseline is None:
            baseline = reward
        advantage = reward - baseline
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * reward

        probs = F.softmax(scores, dim=0)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()

        loss = -advantage * log_prob - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        is_best = cost < best_ever_cost
        if is_best:
            best_ever_cost = cost
            torch.save({
                'model_state_dict': model.state_dict(), 'arch': ckpt['arch'],
                'feat_mean': ckpt['feat_mean'], 'feat_std': ckpt['feat_std'],
                'gmean': ckpt['gmean'], 'gstd': ckpt['gstd'],
                'avg_uld_volume': avg_uld_volume, 'avg_uld_weight': avg_uld_weight,
            }, RL_BEST_CKPT)
            with open(RL_BEST_META, 'w') as f:
                json.dump({'cost': cost, 'step': step}, f)

        with open(RL_LOG_PATH, 'a') as f:
            f.write(json.dumps({'step': step, 'cost': cost, 'baseline': -baseline,
                                 'advantage': advantage, 'entropy': entropy.item()}) + '\n')

        print(f'[step {step:4d}] cost={cost:,.0f}  baseline={-baseline:,.0f}  advantage={advantage:+,.0f}  '
              f'entropy={entropy.item():.2f}  ({eval_dt:.1f}s)  {"** NEW BEST **" if is_best else ""}')

    print(f'\nDone. RL best-ever real cost: {best_ever_cost:,.0f}  '
          f'(pre-RL best: 30,475, competitor target: 29,203)')


if __name__ == '__main__':
    main()
