"""
test_min_envelope_packer.py -- A/B test adding a 4th placement strategy to
CombinedPacker: 'min_envelope' (greedily minimizes growth of the overall
bounding envelope of placed items -- a GLOBAL compactness criterion,
distinct from 'contact' (local touching-face area) and 'dblf' (corner
position)).

Why: selection-side experiments (knapsack ILP, ULD-ordering, weight-based
value density) have all failed to beat the current best of 30,475 -- this
tests the PACKING layer instead, now that the pow1.5 Economy selection
change has altered which specific packages get attempted (the earlier
"packing layer is at its ceiling" conclusion predates that change).

Usage:
    python scripts/test_min_envelope_packer.py
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

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1  # matches production default on this instance
    assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                           econ_sort_key='value_density_pow1.5')

    packers = {
        '3-way (rl+contact+dblf) baseline': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
        ]),
        '4-way (+min_envelope)': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
            ('min_envelope', HeuristicPacker(strategy='min_envelope')),
        ]),
        'min_envelope alone': CombinedPacker([
            ('min_envelope', HeuristicPacker(strategy='min_envelope')),
        ]),
    }

    for label, packer in packers.items():
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'{label:35s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')


if __name__ == '__main__':
    main()
