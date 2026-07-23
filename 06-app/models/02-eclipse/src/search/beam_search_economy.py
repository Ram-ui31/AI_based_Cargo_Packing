"""
beam_search_economy.py -- local beam search over the Economy package
ORDERING (the same representation every econ_sort_key formula, the GNN,
and RL all ultimately produce: a full permutation of the 297 candidate
packages, greedily first-fit into ULDs), searching directly with the REAL
packer as the only evaluator -- no gradient (unlike RL/GRPO, which is
local search around ONE starting point via gradient descent) and no
nominal/proxy fitness (unlike the earlier GA attempts, which failed
specifically because their proxies didn't match real outcomes).

Why this is different from everything else tried: RL/GRPO can only nudge
a neural network's weights in the direction of the local gradient -- it
cannot make a large, structural jump like "swap these two specific
packages' priority" the way this search directly can. GA used a fitness
PROXY (nominal capacity) to stay fast, which was proven to not correlate
well with real placement; this pays the ~15s-per-real-evaluation cost
directly, now known to be cheap enough (confirmed in tonight's RL runs)
for a meaningfully-sized search.

Method: start from the current best full ordering (pow1.5's induced
order, 30,475). Maintain a BEAM of the B best orderings found so far (by
real cost). Each round, generate perturbations of every beam member:
    - boundary swaps: swap a package just inside the include/exclude
      cutoff with one just outside it (most likely to matter, since the
      cutoff is where marginal inclusion decisions are made)
    - random swaps: swap two uniformly random positions (broader
      exploration, catches structural effects the boundary-local swaps
      would miss)
    - single relocations: move one random package to a random new
      position in the order
Real-evaluate every perturbation, keep the top-B among (beam + all
children) for the next round (elitist, so it can only match or improve
on the previous round's beam).

Usage:
    python scripts/beam_search_economy.py --rounds 40 --beam-width 3 --children-per-parent 6
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt'
DENSITY_PACKER_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..',
    'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
GNN_ECON_SELECTOR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'economy-package-ranker')


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


def perturb(order, cutoff, rng, mode):
    """Returns (new_order, move_meta) -- move_meta records exactly what
    changed (mode, package IDs involved, their original positions) so
    every trial -- not just survivors -- can be logged as training data
    for a future swap-proposer model (which packages, which move type,
    did it help)."""
    order = list(order)
    n = len(order)
    if mode == 'boundary_swap' and 0 < cutoff < n:
        lo = max(0, cutoff - 15)
        hi = min(n, cutoff + 15)
        i = rng.randrange(lo, cutoff) if cutoff > lo else rng.randrange(0, n)
        j = rng.randrange(cutoff, hi) if hi > cutoff else rng.randrange(0, n)
        pid_i, pid_j = order[i], order[j]
        order[i], order[j] = order[j], order[i]
        move_meta = {'mode': mode, 'pid_a': pid_i, 'pid_b': pid_j, 'pos_a': i, 'pos_b': j}
    elif mode == 'random_swap':
        i, j = rng.randrange(n), rng.randrange(n)
        pid_i, pid_j = order[i], order[j]
        order[i], order[j] = order[j], order[i]
        move_meta = {'mode': mode, 'pid_a': pid_i, 'pid_b': pid_j, 'pos_a': i, 'pos_b': j}
    elif mode == 'relocate':
        i = rng.randrange(n)
        item = order.pop(i)
        j = rng.randrange(len(order) + 1)
        order.insert(j, item)
        move_meta = {'mode': mode, 'pid_a': item, 'pos_a': i, 'pos_b': j}
    elif mode == 'block_shuffle':
        # A stronger, multi-package move: shuffle a whole contiguous window
        # around the placement cutoff. Single pairwise swaps/relocates can
        # only escape a local optimum one step at a time; this lets the
        # search jump to a materially different neighborhood in one move,
        # for when the beam has converged to distinct orders that all share
        # the same real cost (no single-swap improvement left to find).
        if 0 < cutoff < n:
            lo = max(0, cutoff - 20)
            hi = min(n, cutoff + 20)
        else:
            lo, hi = 0, n
        window = order[lo:hi]
        rng.shuffle(window)
        order[lo:hi] = window
        move_meta = {'mode': mode, 'lo': lo, 'hi': hi}
    else:
        move_meta = {'mode': mode}
    return order, move_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--beam-width', type=int, default=3)
    p.add_argument('--children-per-parent', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-name', type=str, default='default',
                    help='distinct state/log file names, so parallel runs never clobber each other')
    p.add_argument('--seed-order-from', type=str, default='pow1.5', choices=['pow1.5', 'multi_instance_gnn'],
                    help='initial beam seed: the pow1.5 formula ranking (default), or the multi-instance '
                         'GNN\'s own induced ranking (a genuinely different local optimum, 30,608) -- '
                         'seeding the beam with a second, different attractor to explore around.')
    args = p.parse_args()
    rng = random.Random(args.seed)
    BEAM_STATE_PATH = os.path.join(RESULTS_DIR, f'beam_state_{args.run_name}.json')
    BEAM_LOG_PATH = os.path.join(RESULTS_DIR, f'beam_log_{args.run_name}.jsonl')

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

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

    if args.seed_order_from == 'multi_instance_gnn':
        sys.path.insert(0, os.path.join(GNN_ECON_SELECTOR_ROOT, 'src'))
        from model import PackageSetRanker
        from features import build_package_features, build_global_features, normalize_features
        gnn_ckpt = torch.load(os.path.join(GNN_ECON_SELECTOR_ROOT, 'checkpoints', 'multi_instance_FINAL_30608.pt'),
                               map_location='cpu', weights_only=False)
        gnn_model = PackageSetRanker(**gnn_ckpt['arch'])
        gnn_model.load_state_dict(gnn_ckpt['model_state_dict'])
        gnn_model.eval()
        avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
        avg_uld_weight = ulds_df['Weight_Limit'].mean()
        feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
        feats_np, _, _ = normalize_features(feats_np)  # fresh per-instance, matching multi-instance training
        global_feats_np = build_global_features(
            n_ulds=len(ulds_df),
            total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
            total_remaining_weight=ulds_df['Weight_Limit'].sum(),
            k_value=k_value,
        )
        global_feats_np, _, _ = normalize_features(global_feats_np.reshape(1, -1))
        with torch.no_grad():
            scores = gnn_model(
                torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0),
                torch.tensor(global_feats_np, dtype=torch.float32),
            ).squeeze(0).numpy()
        order_idx = (-scores).argsort()
        init_order = economy_df.loc[order_idx, 'Package_ID'].tolist()
        print(f'Seeding beam from multi-instance GNN\'s own induced ranking (its real cost: 30,608)')
    else:
        # pow1.5's induced ranking (the classical best, 30,475).
        econ_sorted = economy_df.assign(
            _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
        ).assign(
            _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5),
        ).sort_values('_vdp', ascending=False)
        init_order = econ_sorted['Package_ID'].tolist()

    if os.path.exists(BEAM_STATE_PATH):
        with open(BEAM_STATE_PATH) as f:
            state = json.load(f)
        beam = [(entry['cost'], entry['order']) for entry in state['beam']]
        start_round = state['round']
        print(f'Resuming beam search from round {start_round}, best={min(b[0] for b in beam):,.0f}')
    else:
        cost0, n0 = real_cost(init_order)
        beam = [(cost0, init_order)]
        start_round = 0
        print(f'Initial order (pow1.5): cost={cost0:,.0f}  econ_placed={n0}')

    os.makedirs(os.path.dirname(BEAM_STATE_PATH), exist_ok=True)
    best_ever_cost = min(b[0] for b in beam)

    for round_idx in range(start_round, start_round + args.rounds):
        t0 = time.time()
        candidates = list(beam)
        move_log_entries = []
        for parent_idx, (cost, order) in enumerate(beam):
            cutoff = None
            assignment, _ = greedy_first_fit(order, pkg_lookup_all, uld_lookup, prio_assignment)
            for idx, pid in enumerate(order):
                if assignment[pid] == 'NONE':
                    cutoff = idx
                    break
            if cutoff is None:
                cutoff = len(order)
            for _ in range(args.children_per_parent):
                mode = rng.choice(['boundary_swap', 'boundary_swap', 'random_swap', 'relocate', 'block_shuffle'])
                child_order, move_meta = perturb(order, cutoff, rng, mode)
                child_cost, _ = real_cost(child_order)
                candidates.append((child_cost, child_order))
                # Training data for a future swap-proposer model: every
                # TRIED move, not just survivors, with its real delta.
                # group_id ties together all children of the SAME parent in
                # the SAME round, across all run files -- the unit a ranking
                # loss needs (comparable deltas share the same parent_cost).
                move_log_entries.append({**move_meta, 'parent_idx': parent_idx, 'parent_cost': cost,
                                          'child_cost': child_cost, 'delta': child_cost - cost,
                                          'group_id': f'{args.run_name}_{round_idx}_{parent_idx}'})

        candidates.sort(key=lambda t: t[0])
        # De-dupe by order: once all beam_width slots converge to the SAME
        # ordering, every next round perturbs beam_width identical copies of
        # it -- a "beam" search that has silently collapsed into redundant
        # single-point local search with no diversity left to explore from.
        # Keeping only the best copy of each distinct order forces the beam
        # to keep exploring beam_width genuinely different neighborhoods.
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

        with open(os.path.join(RESULTS_DIR, f'beam_moves_{args.run_name}.jsonl'), 'a') as f:
            for entry in move_log_entries:
                f.write(json.dumps({'round': round_idx, **entry}) + '\n')

        gen_best = beam[0][0]
        is_new_best = gen_best < best_ever_cost
        if is_new_best:
            best_ever_cost = gen_best

        with open(BEAM_LOG_PATH, 'a') as f:
            f.write(json.dumps({'round': round_idx, 'beam_costs': [b[0] for b in beam],
                                 'n_candidates': len(candidates)}) + '\n')
        with open(BEAM_STATE_PATH, 'w') as f:
            json.dump({'round': round_idx + 1,
                       'beam': [{'cost': c, 'order': o} for c, o in beam]}, f)

        print(f'[round {round_idx:4d}] beam_costs={[f"{c:,.0f}" for c, _ in beam]}  '
              f'n_candidates={len(candidates)}  ({dt:.1f}s)  '
              f'{"** NEW BEST **" if is_new_best else ""}')

    print(f'\nDone. Beam search best-ever real cost: {best_ever_cost:,.0f}  '
          f'(pre-search best: 30,475, generalized RL best: 30,608)')


if __name__ == '__main__':
    main()
