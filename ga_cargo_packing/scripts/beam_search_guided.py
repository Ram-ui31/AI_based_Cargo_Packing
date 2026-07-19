"""
beam_search_guided.py -- same local beam search as beam_search_economy.py,
but candidate perturbations are SCREENED by a trained SwapProposer model
before spending a real evaluation on them: generate a larger POOL of cheap
candidate swaps, score all of them with the model (no real packing, just a
tiny MLP forward pass), and real-evaluate only the top-scoring
`children-per-parent` per round instead of purely random ones.

This is the "make the search itself smarter" follow-up to
beam_search_economy.py, once beam_moves_*.jsonl has enough data to train
gnn_economy_selector/src/train_swap_proposer.py's SwapProposer on.

Usage:
    python scripts/beam_search_guided.py --rounds 40 --beam-width 3 \
        --children-per-parent 6 --pool-multiplier 8 \
        --proposer-ckpt ../gnn_economy_selector/checkpoints/swap_proposer.pt \
        --run-name guided --resume-from-run default
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
GNN_ECON_SELECTOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gnn_economy_selector')
sys.path.insert(0, os.path.join(GNN_ECON_SELECTOR_ROOT, 'src'))

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

from swap_proposer import SwapProposer
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt'
DENSITY_PACKER_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


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


def greedy_first_fit(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment):
    weight_used = {u: 0.0 for u in uld_lookup}
    volume_used = {u: 0.0 for u in uld_lookup}
    for pid, uid in prio_assignment.items():
        weight_used[uid] += pkg_lookup_all[pid]['Weight']
        volume_used[uid] += pkg_lookup_all[pid]['Length'] * pkg_lookup_all[pid]['Width'] * pkg_lookup_all[pid]['Height']
    assignment = dict(prio_assignment)
    n_placed_econ = 0
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
                n_placed_econ += 1
                break
        if not placed:
            assignment[pid] = 'NONE'
    return assignment, n_placed_econ


def swap_candidate(order, i, j):
    order = list(order)
    pid_i, pid_j = order[i], order[j]
    order[i], order[j] = order[j], order[i]
    return order, {'pid_a': pid_i, 'pid_b': pid_j, 'pos_a': i, 'pos_b': j}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--beam-width', type=int, default=3)
    p.add_argument('--children-per-parent', type=int, default=6)
    p.add_argument('--pool-multiplier', type=int, default=8,
                    help='generate pool_multiplier x children_per_parent candidate swaps per parent, '
                         'score them all cheaply with the proposer, real-evaluate only the best children_per_parent')
    p.add_argument('--proposer-ckpt', type=str,
                    default=os.path.join(GNN_ECON_SELECTOR_ROOT, 'checkpoints', 'swap_proposer.pt'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-name', type=str, default='guided')
    p.add_argument('--resume-from-run', type=str, default=None,
                    help='seed the beam from an existing run\'s saved state (e.g. "default") instead '
                         'of starting fresh from pow1.5 -- continues that search, but now guided.')
    args = p.parse_args()
    rng = random.Random(args.seed)
    BEAM_STATE_PATH = os.path.join(RESULTS_DIR, f'beam_state_{args.run_name}.json')
    BEAM_LOG_PATH = os.path.join(RESULTS_DIR, f'beam_log_{args.run_name}.jsonl')

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pid_to_i = {pid: i for i, pid in enumerate(economy_df['Package_ID'])}
    n_items = len(economy_df)

    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full_assignment = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full_assignment.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    # EMS (Empty-Maximal-Space) origin-source candidates added after
    # confirming they beat pivot_points()-based candidates by 200-2000
    # points on this instance (scripts/test_ems_packer.py) -- the shared
    # candidate-generation bottleneck every prior strategy inherited.
    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
        ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ('dblf_ems', HeuristicPacker(strategy='dblf', origin_source='ems')),
    ])

    def real_cost(order):
        assignment, n_placed_econ = greedy_first_fit(order, pkg_lookup_all, uld_lookup, prio_assignment)
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        return cost, n_placed_econ

    # Proposer model + features (fresh per-instance, matching how it was trained).
    proposer_ckpt = torch.load(args.proposer_ckpt, map_location='cpu', weights_only=False)
    proposer = SwapProposer()
    proposer.load_state_dict(proposer_ckpt['model_state_dict'])
    proposer.eval()
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

    # Seed: resume from another run's saved beam if requested, else fresh pow1.5.
    if args.resume_from_run:
        src_state_path = os.path.join(RESULTS_DIR, f'beam_state_{args.resume_from_run}.json')
        with open(src_state_path) as f:
            src_state = json.load(f)
        beam = [(entry['cost'], entry['order']) for entry in src_state['beam']]
        print(f'Seeded guided search from run "{args.resume_from_run}"\'s beam, best={min(b[0] for b in beam):,.0f}')
    else:
        econ_sorted = economy_df.assign(
            _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
        ).assign(
            _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5),
        ).sort_values('_vdp', ascending=False)
        init_order = econ_sorted['Package_ID'].tolist()
        cost0, n0 = real_cost(init_order)
        beam = [(cost0, init_order)]
        print(f'Initial order (pow1.5): cost={cost0:,.0f}  econ_placed={n0}')

    if os.path.exists(BEAM_STATE_PATH):
        with open(BEAM_STATE_PATH) as f:
            state = json.load(f)
        beam = [(entry['cost'], entry['order']) for entry in state['beam']]
        start_round = state['round']
        print(f'Resuming guided run "{args.run_name}" from round {start_round}, best={min(b[0] for b in beam):,.0f}')
    else:
        start_round = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_ever_cost = min(b[0] for b in beam)

    for round_idx in range(start_round, start_round + args.rounds):
        t0 = time.time()
        candidates = list(beam)
        n_screened = 0
        for cost, order in beam:
            n = len(order)
            cutoff = None
            assignment, _ = greedy_first_fit(order, pkg_lookup_all, uld_lookup, prio_assignment)
            for idx, pid in enumerate(order):
                if assignment[pid] == 'NONE':
                    cutoff = idx
                    break
            if cutoff is None:
                cutoff = n

            # Generate a larger pool of candidate swaps cheaply (no real eval yet).
            pool_size = args.children_per_parent * args.pool_multiplier
            pool = []
            for _ in range(pool_size):
                if rng.random() < 0.6 and 0 < cutoff < n:
                    lo = max(0, cutoff - 15)
                    hi = min(n, cutoff + 15)
                    i = rng.randrange(lo, cutoff) if cutoff > lo else rng.randrange(0, n)
                    j = rng.randrange(cutoff, hi) if hi > cutoff else rng.randrange(0, n)
                else:
                    i, j = rng.randrange(n), rng.randrange(n)
                if i == j:
                    continue
                pool.append((i, j))
            n_screened += len(pool)

            # Score the whole pool with the proposer in one batch (cheap).
            idx_a = torch.tensor([pid_to_i[order[i]] for i, j in pool])
            idx_b = torch.tensor([pid_to_i[order[j]] for i, j in pool])
            pos_a = torch.tensor([[i / n] for i, j in pool], dtype=torch.float32)
            pos_b = torch.tensor([[j / n] for i, j in pool], dtype=torch.float32)
            gfeat = global_feat_single.repeat(len(pool), 1)
            with torch.no_grad():
                pred_delta = proposer(feats_t[idx_a], feats_t[idx_b], gfeat, pos_a, pos_b)
            ranked = sorted(range(len(pool)), key=lambda k: pred_delta[k].item())
            top_pairs = [pool[k] for k in ranked[:args.children_per_parent]]

            for i, j in top_pairs:
                child_order, move_meta = swap_candidate(order, i, j)
                child_cost, _ = real_cost(child_order)
                candidates.append((child_cost, child_order))

            # One unscreened block-shuffle per parent: a stronger, multi-
            # package move the pairwise proposer can't score (it only takes
            # two packages), for escaping local optima that are stable
            # under any single swap -- see beam_search_economy.py's
            # identical rationale.
            if 0 < cutoff < n:
                blo, bhi = max(0, cutoff - 20), min(n, cutoff + 20)
            else:
                blo, bhi = 0, n
            block_order = list(order)
            window = block_order[blo:bhi]
            rng.shuffle(window)
            block_order[blo:bhi] = window
            block_cost, _ = real_cost(block_order)
            candidates.append((block_cost, block_order))

        candidates.sort(key=lambda t: t[0])
        # De-dupe by order -- see beam_search_economy.py's identical fix for why:
        # a beam that collapses to identical members stops exploring anything new.
        beam = []
        seen_orders = set()
        for cost, order in candidates:
            key = tuple(order)
            if key in seen_orders:
                continue
            seen_orders.add(key)
            beam.append((cost, order))
            if len(beam) == args.beam_width:
                break
        dt = time.time() - t0

        gen_best = beam[0][0]
        is_new_best = gen_best < best_ever_cost
        if is_new_best:
            best_ever_cost = gen_best

        with open(BEAM_LOG_PATH, 'a') as f:
            f.write(json.dumps({'round': round_idx, 'beam_costs': [b[0] for b in beam],
                                 'n_candidates': len(candidates), 'n_screened': n_screened}) + '\n')
        with open(BEAM_STATE_PATH, 'w') as f:
            json.dump({'round': round_idx + 1,
                       'beam': [{'cost': c, 'order': o} for c, o in beam]}, f)

        print(f'[round {round_idx:4d}] beam_costs={[f"{c:,.0f}" for c, _ in beam]}  '
              f'n_screened={n_screened}  n_real_evals={len(candidates)-len(beam)}  ({dt:.1f}s)  '
              f'{"** NEW BEST **" if is_new_best else ""}')

    print(f'\nDone. Guided beam search best-ever real cost: {best_ever_cost:,.0f}  '
          f'(competitor target: 29,203)')


if __name__ == '__main__':
    main()
