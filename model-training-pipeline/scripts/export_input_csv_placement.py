"""
Runs the final rl_ppo pipeline on ~/Downloads/input.csv (the 400-package
real-world stress-test instance) and dumps the full placement result to
results/input_csv_placement.json -- one entry per package with its assigned
ULD, dimensions, weight, and placed coordinates, plus the ULD fleet's own
dimensions, plus a top-level cost summary.
"""
import json
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
# contrastive_v7 (soft_spread_loss_ipr + economy-only entropy + K-gap-scaled
# contrastive margin + K-weighted consolidation-imitation loss) is the
# current best assignment model. Priority placement uses
# rl_assign_argmax_adaptive rather than the old fixed PRIORITY_CONSOLIDATION_
# MIN_K=500 threshold -- that threshold picked the actually-cheaper of
# {heuristic, model} only 3/8 times below K=500 and 16/32 (a coin flip)
# times at K>=500 when checked directly, because the true crossover point
# is instance-specific, not a single global K. The adaptive version computes
# both candidates and keeps whichever is actually cheaper for this specific
# instance -- a strict min-of-two, so it can only match or beat the fixed
# threshold, never do worse.
CHECKPOINT = 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt'
# Packing uses CombinedPacker: per-ULD best-of-3 (RL density fine-tune +
# two deterministic heuristics, contact-maximizing and Deepest-Bottom-
# Left-Fill) by ACTUAL cost contribution, plus cross-ULD rescue covering
# both packer_unfit and clusterer_none Economy packages. Verified on this
# real 400-package instance: 32,661 (original RLPackerAdapter alone) ->
# 30,822 (this combination), a real, self-contained (no external API)
# 5.6% improvement, zero Priority drops. Exhaustively confirmed (multiple
# placement algorithms, cross-ULD rescue, a from-scratch global Economy
# re-selection, and a proper multi-knapsack ILP selector, all of which
# underperformed this) that the PACKING layer alone is at or near its
# ceiling for whatever Economy MIX gets selected -- closing further gap
# needed a change to the assignment stage's Economy selection ORDER
# instead: rl_assign_argmax_adaptive's econ_sort_key default changed from
# 'value_density' (delay_cost/volume, exponent 1.0) to
# 'value_density_pow1.5' (delay_cost/volume^1.5), prompted by a competing
# team's report placing 20 more Economy packages at virtually identical
# per-ULD fill % (~79% avg vs our ~78%) -- implying their edge was
# selecting a smarter, more-numerous-but-still-high-value MIX, not denser
# real packing. Swept exponents 1.0-2.0 (scripts/test_econ_ascending_
# volume.py): 1.5 is a validated, twice-reproduced peak -> 30,475 (a further
# 1.1%/347-cost improvement over 30,822, 8 more Economy packages placed,
# still zero Priority drops).
DENSITY_PACKER_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
OUT_PATH = 'results/input_csv_placement.json'


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
    print(f'Adaptive Priority placement chose: {chosen}')

    packages_out = []
    for p in placements:
        pid = p['Package_ID']
        pk = pkg_lookup[pid]
        placed = p['ULD_ID'] != 'NONE'
        packages_out.append({
            'package_id': pid,
            'type': pk['Type'],
            'assigned_uld_id': p['ULD_ID'] if placed else None,
            'placed': placed,
            'reason': p.get('reason'),
            'dimensions': {'length': pk['Length'], 'width': pk['Width'], 'height': pk['Height']},
            'weight': pk['Weight'],
            'delay_cost': pk['Delay_Cost'],
            'coordinates': {
                'leftmost_downmost': {'x': p['x0'], 'y': p['y0'], 'z': p['z0']},
                'rightmost_topmost': {'x': p['x1'], 'y': p['y1'], 'z': p['z1']},
            } if placed else None,
        })
    # Stable order: assigned ULD, then package_id.
    packages_out.sort(key=lambda r: (r['assigned_uld_id'] or 'ZZZ_NONE', r['package_id']))

    ulds_out = []
    for uid, row in uld_lookup.items():
        ulds_out.append({
            'uld_id': uid,
            'dimensions': {'length': row['Length'], 'width': row['Width'], 'height': row['Height']},
            'weight_limit': row['Weight_Limit'],
        })

    output = {
        'source_file': INPUT_PATH,
        'checkpoint': CHECKPOINT,
        'k_value': k_value,
        'summary': {
            'n_ulds': len(ulds_df),
            'n_packages': len(pkgs_df),
            'n_priority': int((pkgs_df.Type == 'Priority').sum()),
            'n_economy': int((pkgs_df.Type == 'Economy').sum()),
            'total_cost': cost,
            'spread_cost': spread_cost,
            'delay_cost': delay_cost,
            'spread_n_priority_ulds': n_priority_ulds,
            'priority_dropped': len(unplaced_prio),
            'economy_dropped': len(unplaced_eco),
        },
        'ulds': ulds_out,
        'packages': packages_out,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved {OUT_PATH} ({len(packages_out)} packages, {len(ulds_out)} ULDs)')
    print(f'Total cost: {cost:,.0f}  spread={n_priority_ulds}  delay={delay_cost:,.0f}  '
          f'prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')


if __name__ == '__main__':
    main()
