"""
test_econ_ascending_volume.py -- A/B test for the final Economy greedy
first-fit's sort order: current production default ('value_density' =
descending delay_cost/volume) vs an alternative ('ascending_volume' =
smallest packages first, ignoring delay_cost).

Why: a competing team's ULD-wise report showed ~78-80% volume/weight fill
(nearly identical to this pipeline's own 78.0%/83.9%) but placed 20 more
Economy packages (245 vs our 225 total). Since fill % is basically the
same, their extra packages must average smaller -- suggesting a selection
order that favors package COUNT over per-package value/volume ratio might
recover some of that gap. Cheap, isolated, additive test: only the
econ_sort_key argument differs; everything else (model, Priority
consolidation threshold, CombinedPacker) matches export_input_csv_placement.py exactly.

Usage:
    python scripts/test_econ_ascending_volume.py
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

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1  # matches production default on this instance

    # (econ_sort_key, uld_order_strategy) candidates. baseline = current
    # production best (30,475). Rest test: (a) ULD-ordering fix (best_fit /
    # worst_fit instead of fixed file order U1..U6, holding sort key fixed
    # at the known-best pow1.5), (b) whether weight-based value density
    # (some ULDs hit 93-98% weight-full vs <83% volume-full -- weight may be
    # the tighter binding constraint) beats volume-based.
    tests = [
        ('baseline: pow1.5',                       'value_density_pow1.5', 'file_order'),
        ('joint_pow1.0 (max(vol_frac,wt_frac))',    'value_density_joint_pow1.0', 'file_order'),
        ('joint_pow1.3',                            'value_density_joint_pow1.3', 'file_order'),
        ('joint_pow1.5',                            'value_density_joint_pow1.5', 'file_order'),
        ('joint_pow1.7',                            'value_density_joint_pow1.7', 'file_order'),
        ('joint_pow2.0',                            'value_density_joint_pow2.0', 'file_order'),
    ]
    for label, sort_key, uld_order in tests:
        assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                               econ_sort_key=sort_key, uld_order_strategy=uld_order)
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'{label:40s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')


if __name__ == '__main__':
    main()
