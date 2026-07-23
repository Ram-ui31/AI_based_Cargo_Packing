"""
test_api_rescue.py -- runs our best assignment + RLPackerAdapter (as always),
then rescues packer_unfit Economy packages via the 3dbinpacking.com API
(rescue_unfit_economy_via_api), and reports the before/after cost. Priority
placement is never touched by the rescue pass -- it stays exactly what
RLPackerAdapter already produced.

Usage:
    python scripts/test_api_rescue.py
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
from src.rl.adaptive_assign import rl_assign_argmax_adaptive
from src.rl.api_packer import ThreeDBinPackingAPIPacker, rescue_unfit_economy_via_api, _load_credentials

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt'


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


def report(label, placements, pkgs_df, k_value):
    cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = (
        compute_packing_cost(placements, pkgs_df, k_value)
    )
    by_reason = {}
    for p in placements:
        if p['ULD_ID'] == 'NONE':
            by_reason.setdefault(p['reason'], 0)
            by_reason[p['reason']] += 1
    print(f'{label}')
    print(f'  TOTAL COST   : {cost:,.0f}')
    print(f'  spread cost  : {spread_cost:,.0f}  (spread={n_prio})')
    print(f'  delay cost   : {delay_cost:,.0f}')
    print(f'  priority dropped: {len(unplaced_prio)}  (must be 0)')
    print(f'  economy dropped : {len(unplaced_eco)}  {dict(by_reason)}')
    print()
    return cost


def main():
    if not all(_load_credentials()):
        raise SystemExit(
            'Missing credentials. Set THREEDBINPACKING_USERNAME and '
            'THREEDBINPACKING_API_KEY in the environment, or populate '
            '~/Desktop/3dbinpacking_api/.credentials.json'
        )

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    print(f'K={k_value:.0f}  ULDs={len(ulds_df)}  Packages={len(pkgs_df)}\n')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    own_packer = RLPackerAdapter()
    assignment, own_placements, own_cost, _own_unfit, chosen = rl_assign_argmax_adaptive(
        model, pkgs_df, ulds_df, DEVICE, k_value, own_packer)
    print(f'Assignment stage chose: {chosen} (Priority strategy)\n')

    report('BEFORE rescue (RLPackerAdapter only)', own_placements, pkgs_df, k_value)

    api_packer = ThreeDBinPackingAPIPacker()
    rescued_placements = own_placements
    prev_cost = own_cost
    round_num = 0
    while True:
        round_num += 1
        print(f'\n--- Rescue round {round_num} ---')
        rescued_placements = rescue_unfit_economy_via_api(
            rescued_placements, assignment, pkgs_df, ulds_df, api_packer)
        print()
        rescued_cost = report(f'AFTER rescue round {round_num}', rescued_placements, pkgs_df, k_value)
        if rescued_cost >= prev_cost - 1:  # stop once a round buys us < 1 cost unit
            break
        prev_cost = rescued_cost

    print('=' * 60)
    print(f'Before: {own_cost:,.0f}   After: {rescued_cost:,.0f}   '
          f'Improvement: {(own_cost - rescued_cost) / own_cost * 100:.1f}%')
    print(f'Distance to 27,500 target: {rescued_cost - 27500:,.0f}')


if __name__ == '__main__':
    main()
