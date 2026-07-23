"""
report_uld_utilization.py -- for the current production pipeline (CombinedPacker:
RL density fine-tune + contact heuristic + DBLF heuristic, plus cross-ULD
rescue), reports per-ULD real volume and weight utilization percentage on
the real 400-package instance (~/Downloads/input.csv), at the current best
cost of 30,822.

Usage:
    python scripts/report_uld_utilization.py
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
from src.rl.adaptive_assign import rl_assign_argmax_adaptive

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt'
DENSITY_PACKER_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)


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


def main():
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    assignment, placements, cost, _total_unfit, chosen = rl_assign_argmax_adaptive(
        model, pkgs_df, ulds_df, DEVICE, k_value, packer)
    _, delay_cost, spread_cost, n_priority_ulds, unplaced_prio, unplaced_eco = (
        compute_packing_cost(placements, pkgs_df, k_value)
    )

    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')

    # Per-ULD placed volume / weight, straight from the final placements list
    # (works regardless of which candidate strategy CombinedPacker chose per
    # ULD, since placements already reflect the winning arrangement).
    vol_used = {uid: 0.0 for uid in uld_lookup}
    weight_used = {uid: 0.0 for uid in uld_lookup}
    n_placed = {uid: 0 for uid in uld_lookup}
    for p in placements:
        uid = p['ULD_ID']
        if uid == 'NONE':
            continue
        pid = p['Package_ID']
        pk = pkg_lookup[pid]
        vol_used[uid] += pk['Length'] * pk['Width'] * pk['Height']
        weight_used[uid] += pk['Weight']
        n_placed[uid] += 1

    print(f'Total cost: {cost:,.0f}  (spread={n_priority_ulds}, delay={delay_cost:,.0f}, '
          f'prio_drop={len(unplaced_prio)}, econ_drop={len(unplaced_eco)})\n')
    print(f'{"ULD":8s} {"Packages":9s} {"Vol Used":>12s} {"Vol Cap":>12s} {"Vol %":>7s}   '
          f'{"Wt Used":>10s} {"Wt Cap":>10s} {"Wt %":>7s}')
    tot_vol_used = tot_vol_cap = tot_wt_used = tot_wt_cap = 0.0
    for uid, row in uld_lookup.items():
        vol_cap = row['Length'] * row['Width'] * row['Height']
        wt_cap = row['Weight_Limit']
        vu, wu = vol_used[uid], weight_used[uid]
        print(f'{uid:8s} {n_placed[uid]:9d} {vu:12,.0f} {vol_cap:12,.0f} {100*vu/vol_cap:6.1f}%   '
              f'{wu:10,.0f} {wt_cap:10,.0f} {100*wu/wt_cap:6.1f}%')
        tot_vol_used += vu; tot_vol_cap += vol_cap
        tot_wt_used += wu; tot_wt_cap += wt_cap
    print('-' * 90)
    print(f'{"TOTAL":8s} {sum(n_placed.values()):9d} {tot_vol_used:12,.0f} {tot_vol_cap:12,.0f} '
          f'{100*tot_vol_used/tot_vol_cap:6.1f}%   {tot_wt_used:10,.0f} {tot_wt_cap:10,.0f} '
          f'{100*tot_wt_used/tot_wt_cap:6.1f}%')


if __name__ == '__main__':
    main()
