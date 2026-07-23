"""
build_dataset_random.py -- appends genuinely diverse (random-order)
training scenes to data/real_labels.jsonl, generated the same way as
build_dataset.py but via econ_sort_key='random_seedN' instead of the
value-density formula family, to give the ranker real contrastive signal
beyond correlated formula variants.

Usage:
    python src/build_dataset_random.py
"""
from __future__ import annotations
import os
import sys
import json
import time

import pandas as pd
import torch

GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model-training-pipeline')
sys.path.insert(0, GA_CARGO_ROOT)

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

from build_dataset import parse_input_csv, INPUT_PATH, CHECKPOINT, DENSITY_PACKER_CKPT, OUT_PATH

N_RANDOM_SEEDS = 10


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

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1

    with open(OUT_PATH, 'a') as fout:
        for seed in range(N_RANDOM_SEEDS):
            strat = f'random_seed{seed}'
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
            print(f'[{strat:30s}] cost={cost:,.0f}  econ_drop={len(unplaced_eco)}  ({dt:.1f}s)')

    print(f'\nAppended {N_RANDOM_SEEDS} random-order scenes to {OUT_PATH}')


if __name__ == '__main__':
    main()
