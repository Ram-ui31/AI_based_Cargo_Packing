"""
test_ems_packer.py -- A/B test: does EMS-based candidate-origin generation
beat pivot_points() on the real 400-package instance, holding the scoring
strategy (contact/dblf) fixed? Isolates the origin-generator as the only
variable, per the Stage 1 plan (see .claude/plans -- EMS candidate
generation).

Usage:
    python scripts/test_ems_packer.py
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


def log_ems_sizes(strategy, ulds_df, pkgs_df, assignment):
    """Diagnostic: confirm max_ems=300 isn't binding in practice on this
    instance -- if it regularly hits the cap, that's a confound to rule
    out before trusting a null A/B result."""
    packer = HeuristicPacker(strategy=strategy, origin_source='ems')
    uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    uld_pkg_ids = {uid: [] for uid in uld_lookup}
    for pid, uid in assignment.items():
        if uid != 'NONE' and uid in uld_pkg_ids:
            uld_pkg_ids[uid].append(pid)
    sizes = []
    for uid, pids in uld_pkg_ids.items():
        if not pids:
            continue
        hm, _ = packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
        sizes.append((uid, len(hm.ems_list)))
    print(f'  ems_list sizes per ULD ({strategy}): {sizes}  (cap=300)')


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
        'contact (pivot, baseline)': CombinedPacker([
            ('contact', HeuristicPacker(strategy='contact', origin_source='pivot')),
        ]),
        'contact (ems)': CombinedPacker([
            ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ]),
        'dblf (pivot, baseline)': CombinedPacker([
            ('dblf', HeuristicPacker(strategy='dblf', origin_source='pivot')),
        ]),
        'dblf (ems)': CombinedPacker([
            ('dblf_ems', HeuristicPacker(strategy='dblf', origin_source='ems')),
        ]),
        '3-way (rl+contact+dblf) baseline': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
        ]),
        '4-way (+ems_contact)': CombinedPacker([
            ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
            ('contact', HeuristicPacker(strategy='contact')),
            ('dblf', HeuristicPacker(strategy='dblf')),
            ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ]),
    }

    for label, packer in packers.items():
        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'{label:35s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')

    print()
    log_ems_sizes('contact', ulds_df, pkgs_df, assignment)
    log_ems_sizes('dblf', ulds_df, pkgs_df, assignment)


if __name__ == '__main__':
    main()
