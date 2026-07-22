"""
milp_ceiling.py -- solves the Economy package -> ULD assignment as an
actual multiple-knapsack MILP (volume + weight capacity only, ignoring
exact 3D shape), giving a THEORETICAL CEILING on how good any
order/assignment-based search could possibly do -- since real 3D
geometric packing can only be harder than this relaxation, never easier.

Then takes the MILP's optimal item selection and real-packs it with the
actual packer, to see how much of the gap between our best real result
and this ceiling is (a) a "which items" search problem (fixed by using
the MILP's selection directly) vs (b) a 3D packing-efficiency problem
(the real packer failing to realize even a volume/weight-optimal
selection, since box shapes don't tile as cleanly as raw volume implies).

Usage:
    python scripts/milp_ceiling.py --time-limit 200
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import milp, LinearConstraint, Bounds

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
            uld_rows.append({'ULD_ID': uid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight_Limit': float(weight_limit)})
        else:
            pid, length, width, height, weight, ptype, delay = parts
            delay_cost = 0.0 if delay.strip() == '-' else float(delay)
            pkg_rows.append({'Package_ID': pid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight': float(weight), 'Type': ptype,
                              'Delay_Cost': delay_cost})
    return k_value, pd.DataFrame(uld_rows), pd.DataFrame(pkg_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--time-limit', type=float, default=200.0)
    p.add_argument('--mip-gap', type=float, default=1e-4)
    args = p.parse_args()

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    uld_ids = list(uld_lookup.keys())

    clusterer = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full0 = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                      econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full0.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    remaining_vol = {u: uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height'] for u in uld_ids}
    remaining_wt = {u: uld_lookup[u]['Weight_Limit'] for u in uld_ids}
    for pid, uid in prio_assignment.items():
        pk = pkg_lookup_all[pid]
        remaining_vol[uid] -= pk['Length'] * pk['Width'] * pk['Height']
        remaining_wt[uid] -= pk['Weight']

    pids = list(economy_df['Package_ID'])
    n_items, n_ulds = len(pids), len(uld_ids)
    vols = np.array([pkg_lookup_all[pid]['Length'] * pkg_lookup_all[pid]['Width'] * pkg_lookup_all[pid]['Height']
                      for pid in pids])
    wts = np.array([pkg_lookup_all[pid]['Weight'] for pid in pids])
    delays = np.array([pkg_lookup_all[pid]['Delay_Cost'] for pid in pids])
    cap_vol = np.array([remaining_vol[u] for u in uld_ids])
    cap_wt = np.array([remaining_wt[u] for u in uld_ids])

    n_vars = n_items * n_ulds
    c = np.repeat(-delays, n_ulds)  # c[i*n_ulds+u] = -delays[i]

    A_item = np.zeros((n_items, n_vars))
    for i in range(n_items):
        A_item[i, i * n_ulds:(i + 1) * n_ulds] = 1
    item_constraint = LinearConstraint(A_item, ub=np.ones(n_items))

    A_vol = np.zeros((n_ulds, n_vars))
    A_wt = np.zeros((n_ulds, n_vars))
    for u in range(n_ulds):
        for i in range(n_items):
            A_vol[u, i * n_ulds + u] = vols[i]
            A_wt[u, i * n_ulds + u] = wts[i]
    vol_constraint = LinearConstraint(A_vol, ub=cap_vol)
    wt_constraint = LinearConstraint(A_wt, ub=cap_wt)

    print(f'{n_items} economy items, {n_ulds} ULDs, {n_vars} binary variables')
    print(f'Total economy volume={vols.sum():,.0f} vs remaining ULD volume={cap_vol.sum():,.0f} '
          f'({vols.sum() / cap_vol.sum():.2f}x overbooked)')
    print('\nSolving MILP (volume+weight relaxation of true 3D geometry)...')
    t0 = time.time()
    res = milp(c, constraints=[item_constraint, vol_constraint, wt_constraint],
               integrality=np.ones(n_vars), bounds=Bounds(0, 1),
               options={'time_limit': args.time_limit, 'mip_rel_gap': args.mip_gap, 'disp': False})
    dt = time.time() - t0
    print(f'Solved in {dt:.1f}s: status={res.status}  message={res.message}')

    x = res.x.reshape(n_items, n_ulds)
    milp_assignment = {}
    for i, pid in enumerate(pids):
        chosen = np.where(x[i] > 0.5)[0]
        milp_assignment[pid] = uld_ids[chosen[0]] if len(chosen) > 0 else 'NONE'

    best_delay_placed = -res.fun
    total_delay = delays.sum()
    print(f'MILP total delay_cost placed: {best_delay_placed:,.1f} / {total_delay:,.1f} total')
    print(f'Volume+weight-only ceiling on unplaced-economy delay cost: {total_delay - best_delay_placed:,.1f}')

    with open(os.path.join(RESULTS_DIR, 'milp_assignment.pkl'), 'wb') as f:
        pickle.dump({'milp_assignment': milp_assignment, 'prio_assignment': prio_assignment,
                     'milp_delay_placed': best_delay_placed, 'milp_status': res.status,
                     'milp_message': res.message}, f)

    print('\n--- Now real-evaluating the MILP\'s selection with the actual 3D packer ---')
    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])
    full_assignment = {**prio_assignment, **milp_assignment}
    placements, total_unfit = packer.pack(full_assignment, pkgs_df, ulds_df)
    cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
        placements, pkgs_df, k_value)
    print(f'\nReal cost of MILP-selected assignment: {cost:,.0f}  '
          f'(delay_cost={delay_cost:,.0f}, spread_cost={spread_cost:,.0f}, '
          f'unplaced_prio={len(unplaced_prio)}, unplaced_econ={len(unplaced_eco)})')
    print(f'Theoretical ceiling')


if __name__ == '__main__':
    main()
