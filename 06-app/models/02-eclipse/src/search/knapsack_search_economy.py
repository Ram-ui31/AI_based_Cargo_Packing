"""
knapsack_search_economy.py -- local BEAM search directly on the Economy
package -> ULD ASSIGNMENT, not on an ordering fed through greedy_first_fit.

Why: every order-based search this session (classical formulas, RL/GRPO,
beam search over orderings) plateaued, and the beam search's own log
proved why -- three GENUINELY DIFFERENT orderings converged to the
IDENTICAL real cost. Order is only an indirect, lossy encoding of what
actually determines cost: which packages get placed, and where.
greedy_first_fit collapses a huge space of orderings onto a much smaller
space of assignments, so once a beam finds a locally-good assignment,
nearby order-perturbations mostly just re-derive the same assignment.
(HeuristicPacker._greedy_pack_into is itself order-independent -- it
re-scores every remaining candidate at each step -- so the packer's own
placement quality was never the bottleneck; the ORDER's only real job was
indirectly picking a SET via greedy_first_fit's cutoff.)

This script searches that smaller space directly: the state IS a
pid -> ULD-or-NONE assignment. Moves (swap/relocate/two_for_one) mutate
it directly, so no move is wasted re-deriving something already tried.

IMPORTANT lesson from the first version of this script: a cheap
volume+weight-only proxy is NOT a safe stand-in for real 3D feasibility --
box shape/orientation constraints reject plenty of moves the proxy
thinks fit, and batching many proxy-only moves before any real check let
that error compound catastrophically (real cost went 29,564 -> 33,000+
in 3 epochs). Fixed here by real-evaluating every single candidate
immediately with the actual packer (exactly as beam_search_economy.py
does for order-perturbations) -- no move survives into the beam without
having been checked against reality first.

Moves:
  - swap: evict one assigned package, insert one unassigned package into
    the freed capacity.
  - relocate: move one assigned package to a different ULD with room.
  - two_for_one: evict one assigned package, insert two unassigned
    packages into the freed capacity -- targets the classic knapsack
    fragmentation failure a single swap can't reach.
  - every candidate also gets an opportunistic refill pass (greedily
    insert any still-unassigned package into remaining slack) before
    being real-evaluated, so no real-eval is spent on an obviously
    incomplete assignment.

Usage:
    python scripts/knapsack_search_economy.py --rounds 40 --beam-width 3 --children-per-parent 6
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
    p.add_argument('--rounds', type=int, default=40)
    p.add_argument('--beam-width', type=int, default=3)
    p.add_argument('--children-per-parent', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-name', type=str, default='knapsack_v1')
    p.add_argument('--seed-from-run', type=str, default=None,
                    help='seed the initial assignment from an existing order-based beam run\'s best '
                         '(e.g. "guided_v3" or "gnn_seed") via greedy_first_fit.')
    args = p.parse_args()
    rng = random.Random(args.seed)
    STATE_PATH = os.path.join(RESULTS_DIR, f'knapsack_state_{args.run_name}.json')
    LOG_PATH = os.path.join(RESULTS_DIR, f'knapsack_log_{args.run_name}.jsonl')

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    economy_pids = list(economy_df['Package_ID'])
    economy_pid_set = set(economy_pids)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    uld_ids = list(uld_lookup.keys())

    def vol(pid):
        p = pkg_lookup_all[pid]
        return p['Length'] * p['Width'] * p['Height']

    def wt(pid):
        return pkg_lookup_all[pid]['Weight']

    def delay(pid):
        return pkg_lookup_all[pid]['Delay_Cost']

    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full_assignment0 = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                 econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full_assignment0.items()
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

    def real_eval(econ_assignment):
        """Real-evaluate via the actual packer. Returns (cost, resynced
        econ_assignment reflecting what the packer ACTUALLY placed)."""
        full_assignment = {**prio_assignment, **econ_assignment}
        placements, total_unfit = packer.pack(full_assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        real_econ_assignment = {}
        for pl in placements:
            pid, uid = pl['Package_ID'], pl['ULD_ID']
            if pid in economy_pid_set:
                real_econ_assignment[pid] = uid
        return cost, real_econ_assignment, len(unplaced_prio)

    def capacities_from_assignment(econ_assignment):
        remaining_vol = {u: uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height']
                          for u in uld_ids}
        remaining_wt = {u: uld_lookup[u]['Weight_Limit'] for u in uld_ids}
        for pid, uid in prio_assignment.items():
            remaining_vol[uid] -= vol(pid)
            remaining_wt[uid] -= wt(pid)
        for pid, uid in econ_assignment.items():
            if uid != 'NONE':
                remaining_vol[uid] -= vol(pid)
                remaining_wt[uid] -= wt(pid)
        return remaining_vol, remaining_wt

    def refill(econ_assignment, remaining_vol, remaining_wt):
        unassigned = [pid for pid in economy_pids if econ_assignment.get(pid, 'NONE') == 'NONE']
        unassigned.sort(key=lambda pid: delay(pid) / max(vol(pid), 1.0) ** 1.5, reverse=True)
        for pid in unassigned:
            pv, pw = vol(pid), wt(pid)
            for uid in uld_ids:
                if remaining_vol[uid] + 1e-6 >= pv and remaining_wt[uid] + 1e-6 >= pw:
                    econ_assignment[pid] = uid
                    remaining_vol[uid] -= pv
                    remaining_wt[uid] -= pw
                    break

    def mutate(econ_assignment, remaining_vol, remaining_wt, mode):
        """Applies ONE move to a COPY of econ_assignment/capacities, then
        an opportunistic refill pass. Returns the mutated child (does not
        touch its arguments)."""
        child = dict(econ_assignment)
        rv, rw = dict(remaining_vol), dict(remaining_wt)
        assigned = [pid for pid in economy_pids if child.get(pid, 'NONE') != 'NONE']
        unassigned = [pid for pid in economy_pids if child.get(pid, 'NONE') == 'NONE']

        if mode == 'swap' and assigned and unassigned:
            a = rng.choice(assigned)
            b = rng.choice(unassigned)
            u = child[a]
            if rv[u] + vol(a) + 1e-6 >= vol(b) and rw[u] + wt(a) + 1e-6 >= wt(b):
                child[a] = 'NONE'
                child[b] = u
                rv[u] += vol(a) - vol(b)
                rw[u] += wt(a) - wt(b)
        elif mode == 'relocate' and assigned and len(uld_ids) > 1:
            a = rng.choice(assigned)
            u1 = child[a]
            u2 = rng.choice([u for u in uld_ids if u != u1])
            if rv[u2] + 1e-6 >= vol(a) and rw[u2] + 1e-6 >= wt(a):
                child[a] = u2
                rv[u1] += vol(a)
                rw[u1] += wt(a)
                rv[u2] -= vol(a)
                rw[u2] -= wt(a)
        elif mode == 'two_for_one' and assigned and len(unassigned) >= 2:
            a = rng.choice(assigned)
            b, c = rng.sample(unassigned, 2)
            u = child[a]
            need_v, need_w = vol(b) + vol(c), wt(b) + wt(c)
            if rv[u] + vol(a) + 1e-6 >= need_v and rw[u] + wt(a) + 1e-6 >= need_w:
                child[a] = 'NONE'
                child[b] = u
                child[c] = u
                rv[u] += vol(a) - need_v
                rw[u] += wt(a) - need_w

        refill(child, rv, rw)
        return child

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        beam = [(entry['cost'], entry['assignment']) for entry in state['beam']]
        start_round = state['round']
        print(f'Resuming knapsack beam "{args.run_name}" from round {start_round}, '
              f'best={min(b[0] for b in beam):,.0f}')
    else:
        if args.seed_from_run:
            src_path = os.path.join(RESULTS_DIR, f'beam_state_{args.seed_from_run}.json')
            with open(src_path) as f:
                src_state = json.load(f)
            best_entry = min(src_state['beam'], key=lambda b: b['cost'])
            init_order = best_entry['order']
            print(f'Seeding from order-based run "{args.seed_from_run}"\'s best (real cost {best_entry["cost"]:,.0f})')
        else:
            econ_sorted = economy_df.assign(
                _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
            ).assign(
                _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5),
            ).sort_values('_vdp', ascending=False)
            init_order = econ_sorted['Package_ID'].tolist()
            print('Seeding from pow1.5 formula order')
        init_full = greedy_first_fit(init_order, pkg_lookup_all, uld_lookup, prio_assignment)
        init_assignment = {pid: init_full[pid] for pid in economy_pids}
        cost0, init_assignment, n_unplaced_prio0 = real_eval(init_assignment)
        beam = [(cost0, init_assignment)]
        start_round = 0
        print(f'Initial real cost: {cost0:,.0f}')

    best_ever_cost = min(b[0] for b in beam)

    for round_idx in range(start_round, start_round + args.rounds):
        t0 = time.time()
        candidates = list(beam)
        for cost, econ_assignment in beam:
            remaining_vol, remaining_wt = capacities_from_assignment(econ_assignment)
            for _ in range(args.children_per_parent):
                mode = rng.choice(['swap', 'swap', 'relocate', 'two_for_one'])
                child = mutate(econ_assignment, remaining_vol, remaining_wt, mode)
                child_cost, child, n_unplaced_prio = real_eval(child)
                candidates.append((child_cost, child))

        candidates.sort(key=lambda t: t[0])
        # De-dupe by assignment: keep only the best copy of each distinct
        # assignment, so the beam doesn't collapse into redundant copies of
        # the same solution (see beam_search_economy.py's identical fix).
        beam = []
        seen = set()
        for cost, assignment in candidates:
            key = tuple(sorted(assignment.items()))
            if key in seen:
                continue
            seen.add(key)
            beam.append((cost, assignment))
            if len(beam) == args.beam_width:
                break
        dt = time.time() - t0

        gen_best = beam[0][0]
        is_new_best = gen_best < best_ever_cost
        if is_new_best:
            best_ever_cost = gen_best

        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps({'round': round_idx, 'beam_costs': [b[0] for b in beam],
                                 'n_candidates': len(candidates)}) + '\n')
        with open(STATE_PATH, 'w') as f:
            json.dump({'round': round_idx + 1,
                       'beam': [{'cost': c, 'assignment': a} for c, a in beam]}, f)

        print(f'[round {round_idx:4d}] beam_costs={[f"{c:,.0f}" for c, _ in beam]}  '
              f'n_candidates={len(candidates)}  ({dt:.1f}s)  '
              f'{"** NEW BEST **" if is_new_best else ""}')

    print(f'\nDone. Knapsack beam search best-ever real cost: {best_ever_cost:,.0f}  '
          f'')


if __name__ == '__main__':
    main()
