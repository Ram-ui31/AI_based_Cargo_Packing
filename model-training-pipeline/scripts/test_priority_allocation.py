"""
test_priority_allocation.py -- tests whether the SPECIFIC Priority-package
-to-ULD allocation (within the already-confirmed-best 3-ULD set, U3/U5/U6)
matters, holding that 3-ULD set fixed. Different from
test_priority_uld_combo.py, which only varied WHICH 3 ULDs collectively
hold Priority -- this varies HOW the 103 Priority packages are distributed
among those 3, given they're always sorted by descending volume for the
FFD trial.

Strategies tested, all real-evaluated through the full production
pipeline (Economy: value_density_pow1.5, Packing: CombinedPacker):
    'ffd_vol_desc'   -- current default: first-fit into the largest-volume
                         ULD (of the 3) that has room, packages tried
                         largest-first.
    'ffd_weight_desc'-- same FFD mechanics, but packages tried heaviest-first.
    'best_fit'       -- for each package (largest-first), assign to
                         whichever of the 3 ULDs would be left with the
                         LEAST remaining volume (tightest fit), not just
                         the first one (by size) that has room.
    'worst_fit'      -- assign to whichever ULD is left with the MOST
                         remaining volume (spreads Priority more evenly).
    'balanced_util'  -- assign to whichever ULD has the LOWEST current
                         utilization fraction (keeps utilization % even
                         across the 3 ULDs as Priority fills them).

Usage:
    python scripts/test_priority_allocation.py
"""
from __future__ import annotations
import os
import sys

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
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
BEST_COMBO = ['U5', 'U6', 'U3']  # volume-descending order, confirmed best 3-ULD set


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


def allocate_priority(strategy, prio_ids, pkg_lookup, uld_lookup, candidate_ulds):
    cap_w = {u: uld_lookup[u]['Weight_Limit'] for u in candidate_ulds}
    cap_v = {u: uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height'] for u in candidate_ulds}
    weight_used = {u: 0.0 for u in candidate_ulds}
    volume_used = {u: 0.0 for u in candidate_ulds}

    if strategy == 'ffd_vol_desc':
        order = sorted(prio_ids, key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']))
        try_order_fn = lambda pv, pw: candidate_ulds  # fixed volume-desc ULD order
    elif strategy == 'ffd_weight_desc':
        order = sorted(prio_ids, key=lambda p: -pkg_lookup[p]['Weight'])
        try_order_fn = lambda pv, pw: candidate_ulds
    elif strategy == 'best_fit':
        order = sorted(prio_ids, key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']))
        try_order_fn = lambda pv, pw: sorted(candidate_ulds, key=lambda u: cap_v[u] - volume_used[u])
    elif strategy == 'worst_fit':
        order = sorted(prio_ids, key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']))
        try_order_fn = lambda pv, pw: sorted(candidate_ulds, key=lambda u: -(cap_v[u] - volume_used[u]))
    elif strategy == 'balanced_util':
        order = sorted(prio_ids, key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']))
        try_order_fn = lambda pv, pw: sorted(candidate_ulds, key=lambda u: volume_used[u] / cap_v[u])
    else:
        raise ValueError(strategy)

    assignment = {}
    for pid in order:
        p = pkg_lookup[pid]
        pv = p['Length'] * p['Width'] * p['Height']
        pw = p['Weight']
        for u in try_order_fn(pv, pw):
            if weight_used[u] + pw <= cap_w[u] + 1e-6 and volume_used[u] + pv <= cap_v[u] + 1e-6:
                weight_used[u] += pw
                volume_used[u] += pv
                assignment[pid] = u
                break
        else:
            return None
    return assignment


def main():
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    prio_ids = pkgs_df[pkgs_df['Type'].str.upper() == 'PRIORITY']['Package_ID'].tolist()

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    orig_consolidate = tr._consolidate_priority_by_capacity

    for strategy in ['ffd_vol_desc', 'ffd_weight_desc', 'best_fit', 'worst_fit', 'balanced_util']:
        prio_assignment = allocate_priority(strategy, prio_ids, pkg_lookup, uld_lookup, BEST_COMBO)
        if prio_assignment is None:
            print(f'{strategy:16s}: INFEASIBLE (some priority package could not be placed nominally)')
            continue

        def _forced(packages_df, ulds_df_inner, _assignment=prio_assignment):
            weight_used = {u: 0.0 for u in BEST_COMBO}
            volume_used = {u: 0.0 for u in BEST_COMBO}
            for pid, uid in _assignment.items():
                weight_used[uid] += pkg_lookup[pid]['Weight']
                volume_used[uid] += pkg_lookup[pid]['Length'] * pkg_lookup[pid]['Width'] * pkg_lookup[pid]['Height']
            return dict(_assignment), weight_used, volume_used

        tr.PRIORITY_CONSOLIDATION_MIN_K = -1
        tr._consolidate_priority_by_capacity = _forced
        try:
            assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                                   econ_sort_key='value_density_pow1.5')
        finally:
            tr._consolidate_priority_by_capacity = orig_consolidate

        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'{strategy:16s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')

    print(f'\nCurrent best (ffd_vol_desc, i.e. the existing default): 30,475')
    print(f'Competitor target: 29,203')


if __name__ == '__main__':
    main()
