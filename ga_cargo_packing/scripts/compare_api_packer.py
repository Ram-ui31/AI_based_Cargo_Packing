"""
compare_api_packer.py -- runs our best assignment model (contrastive_v7,
adaptive Priority/Economy selection) on the real 400-package instance, then
packs the SAME assignment two ways:
    1. our own RLPackerAdapter (Big/256)
    2. the third-party 3dbinpacking.com API (ThreeDBinPackingAPIPacker)
and reports both costs side by side, plus what each packer left behind and
why -- see ~/Desktop/3dbinpacking_api/api_reference.md for the API research
and src/rl/api_packer.py for the implementation.

Requires THREEDBINPACKING_USERNAME and THREEDBINPACKING_API_KEY set in the
environment (see api_reference.md for how to get them) -- fails loudly at
startup if they're missing, before doing any assignment work.

Usage:
    export THREEDBINPACKING_USERNAME=...
    export THREEDBINPACKING_API_KEY=...
    python scripts/compare_api_packer.py
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
from src.rl.api_packer import ThreeDBinPackingAPIPacker, _load_credentials

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
    # Fail fast, before any GPU/model loading, if credentials are missing.
    if not all(_load_credentials()):
        raise SystemExit(
            'Missing credentials. Set THREEDBINPACKING_USERNAME and '
            'THREEDBINPACKING_API_KEY in the environment, or populate '
            '~/Desktop/3dbinpacking_api/.credentials.json -- see '
            '~/Desktop/3dbinpacking_api/api_reference.md for how to obtain them.'
        )

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    print(f'K={k_value:.0f}  ULDs={len(ulds_df)}  Packages={len(pkgs_df)}\n')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    own_packer = RLPackerAdapter()
    # Get our best assignment (heuristic-vs-model adaptive selection) using
    # our OWN packer first -- this is the assignment we'll re-pack with the
    # API too, so both packers work from the identical package->ULD mapping.
    assignment, own_placements, own_cost, _own_unfit, chosen = rl_assign_argmax_adaptive(
        model, pkgs_df, ulds_df, DEVICE, k_value, own_packer)
    print(f'Assignment stage chose: {chosen} (Priority strategy)\n')

    report('OWN PACKER (RLPackerAdapter, Big/256)', own_placements, pkgs_df, k_value)

    api_packer = ThreeDBinPackingAPIPacker()
    api_placements, _api_unfit = api_packer.pack(assignment, pkgs_df, ulds_df)
    api_cost = report('3dbinpacking.com API (same assignment)', api_placements, pkgs_df, k_value)

    print('=' * 60)
    if api_cost < own_cost:
        print(f'API packer WINS: {api_cost:,.0f} vs our {own_cost:,.0f} '
              f'({(own_cost - api_cost) / own_cost * 100:.1f}% better)')
    else:
        print(f'Our packer still wins: {own_cost:,.0f} vs API {api_cost:,.0f} '
              f'({(api_cost - own_cost) / own_cost * 100:.1f}% worse via API)')


if __name__ == '__main__':
    main()
