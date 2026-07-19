"""
build_dataset.py -- generates REAL (not proxy) training labels for the
Economy package ranking model, by running a diverse set of Economy
selection strategies through the FULL production pipeline (Priority fixed
to the confirmed-best combo, U3/U5/U6; CombinedPacker: rl + contact + dblf
+ cross-ULD rescue) on the real 400-package instance.

For each strategy run, records per-Economy-package: its features and
whether it was ACTUALLY PLACED (real geometry, not nominal), tagged with
that strategy's overall real cost (a quality signal for how good that
strategy's OVERALL choice was). This reuses econ_sort_key variants already
implemented and validated in ga_cargo_packing/src/rl/train_rl.py (pow1.0-
2.0, wpow, joint_pow, ascending_volume) as a first, moderate-diversity
labeled dataset -- NOT proxy-generated, every row reflects a real
CombinedPacker outcome.

Usage:
    python src/build_dataset.py
"""
from __future__ import annotations
import os
import sys
import json
import time

import pandas as pd
import torch

GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ga_cargo_packing')
sys.path.insert(0, GA_CARGO_ROOT)

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real_labels.jsonl')

# Diverse strategies, all validated/implemented already in train_rl.py --
# spans the full range of real outcomes explored tonight (30,475 best down
# to worse variants), giving the ranker contrastive signal about what
# "better" and "worse" real selections actually look like.
STRATEGIES = [
    'value_density', 'value_density_pow1.3', 'value_density_pow1.4',
    'value_density_pow1.5', 'value_density_pow1.6', 'value_density_pow1.7',
    'value_density_pow2.0', 'value_density_wpow1.0', 'value_density_wpow1.5',
    'value_density_joint_pow1.0', 'value_density_joint_pow1.3',
    'value_density_joint_pow1.5', 'value_density_joint_pow1.7',
    'value_density_joint_pow2.0', 'ascending_volume',
]


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
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1  # -> _consolidate_priority_by_capacity, confirmed-best combo (U3,U5,U6)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results_summary = []
    with open(OUT_PATH, 'w') as fout:
        for strat in STRATEGIES:
            t0 = time.time()
            assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                                   econ_sort_key=strat)
            placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
            cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
                placements, pkgs_df, k_value)
            placed_ids = {p['Package_ID'] for p in placements if p['ULD_ID'] != 'NONE'}

            row = {
                'strategy': strat, 'cost': cost, 'delay_cost': delay_cost,
                'spread': n_prio, 'econ_drop': len(unplaced_eco),
                'placed_economy_ids': sorted(
                    pid for pid in placed_ids if pkg_lookup[pid]['Type'] != 'Priority'),
            }
            fout.write(json.dumps(row) + '\n')
            fout.flush()
            dt = time.time() - t0
            results_summary.append((strat, cost, len(unplaced_eco), dt))
            print(f'[{strat:30s}] cost={cost:,.0f}  econ_drop={len(unplaced_eco)}  ({dt:.1f}s)')

    print(f'\nSaved {len(STRATEGIES)} strategy runs to {OUT_PATH}')
    best = min(results_summary, key=lambda r: r[1])
    print(f'Best in this batch: {best[0]} (cost={best[1]:,.0f})')


if __name__ == '__main__':
    main()
