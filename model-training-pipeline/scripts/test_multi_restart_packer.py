"""
test_multi_restart_packer.py -- A/B test adding MultiRestartPacker
(epsilon-greedy multi-start local search over the 'contact' scoring key,
N restarts per ULD, keep the real cheapest) as a candidate to
CombinedPacker, on top of the current best (pow1.5 + rl/contact/dblf).

Why: every strategy tried so far (contact, dblf, min_envelope) is a single
deterministic greedy pass -- zero exploration. This tests whether
randomized multi-restart of the same validated 'contact' criterion can
find a genuinely better arrangement that the single deterministic path
missed.

Usage:
    python scripts/test_multi_restart_packer.py
"""
from __future__ import annotations
import os
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
from src.rl.multi_restart_packer import MultiRestartPacker
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

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                           econ_sort_key='value_density_pow1.5')

    packers = {
        '3-way baseline (rl+contact+dblf)': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
        ]),
        '4-way (+multi_restart contact, N=15)': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
            ('multi_restart', MultiRestartPacker(base_strategy='contact', n_restarts=15,
                                                  epsilon=0.15, max_pivots=200)),
        ]),
    }

    for label, packer in packers.items():
        t0 = time.time()
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        dt = time.time() - t0
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'{label:40s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}  '
              f'(pack time {dt:.1f}s)')


if __name__ == '__main__':
    main()
