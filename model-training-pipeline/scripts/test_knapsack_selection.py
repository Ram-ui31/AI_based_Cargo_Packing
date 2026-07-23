"""
test_knapsack_selection.py -- tests replacing _assign_economy_by_value_density's
greedy Economy selection with an actual multi-knapsack ILP solve
(knapsack_economy_selector.py), on the real 400-package instance.

Pipeline: Priority placed first (best-of-3 strategies, same as
combined_packer.py) -> real remaining weight/volume capacity per ULD
computed from that -> knapsack ILP selects which Economy packages go where
-> full (Priority + knapsack-selected Economy) assignment fed through
CombinedPacker for real 3D packing (verifies/realizes the selection; some
knapsack-selected packages may still not physically fit even though they
fit nominally) -> final cost compared against the current best (30,822).

Usage:
    python scripts/test_knapsack_selection.py
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
from src.rl.knapsack_economy_selector import solve_economy_knapsack
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
    print(f'K={k_value:.0f}  ULDs={len(ulds_df)}  Packages={len(pkgs_df)}\n')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    rl_packer = RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)
    contact_packer = HeuristicPacker(strategy='contact')
    dblf_packer = HeuristicPacker(strategy='dblf')
    candidates = [('rl', rl_packer), ('contact', contact_packer), ('dblf', dblf_packer)]
    combined = CombinedPacker(candidates)

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1  # force heuristic Priority strategy (matches prod on this instance)
    assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value)

    # ── Priority phase only: best-of-3 per ULD, real remaining capacity ────
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
    uld_priority_ids = {uid: [] for uid in uld_lookup}
    for pid, uid in assignment.items():
        if uid != 'NONE' and pkg_lookup[pid]['Type'] == 'Priority':
            uld_priority_ids[uid].append(pid)

    hm_by_uld = {}
    for uid, pids in uld_priority_ids.items():
        if not pids:
            hm_by_uld[uid] = None
            continue
        best_hm, best_score = None, None
        for name, packer in candidates:
            hm, left = packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
            score = (len(left), -hm.utilization() if hm else 0.0)
            if best_score is None or score < best_score:
                best_hm, best_score = hm, score
        hm_by_uld[uid] = best_hm

    uld_capacities = {}
    for uid, row in uld_lookup.items():
        hm = hm_by_uld.get(uid)
        used_w = hm.weight_used if hm else 0.0
        used_v = hm.volume_used if hm else 0
        uld_capacities[uid] = (row['Weight_Limit'] - used_w, row['Length'] * row['Width'] * row['Height'] - used_v)
    for uid, (rw, rv) in uld_capacities.items():
        print(f'{uid}: remaining weight={rw:.0f}  remaining volume={rv:,.0f}')
    print()

    # ── Knapsack: select which Economy packages go where ───────────────────
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    for vuf in (0.72, 0.78, 0.82):
        t0 = time.time()
        knapsack_assignment = solve_economy_knapsack(economy_df, uld_capacities, time_limit=300,
                                                       volume_utilization_factor=vuf)
        print(f'[vuf={vuf}] knapsack solve time: {time.time()-t0:.1f}s, '
              f'selected {len(knapsack_assignment)}/{len(economy_df)} Economy packages')

        # ── Merge Priority (original) + knapsack-selected Economy, feed through real packer ──
        full_assignment = {}
        for uid, pids in uld_priority_ids.items():
            for pid in pids:
                full_assignment[pid] = uid
        for pid in economy_df['Package_ID']:
            full_assignment[pid] = knapsack_assignment.get(pid, 'NONE')

        placements, total_unfit = combined.pack(full_assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(placements, pkgs_df, k_value)
        n_placed_econ = len(economy_df) - len(unplaced_eco)
        print(f'[vuf={vuf}] KNAPSACK + real packing: cost={cost:,.0f}  spread={n_prio}  delay={delay_cost:,.0f}  '
              f'prio_drop={len(unplaced_prio)}  econ_placed={n_placed_econ}  econ_drop={len(unplaced_eco)}\n')

    print(f'Current best (value_density_pow1.5 greedy selection): 30,475')
    print(f'Competitor target to beat: 29,203')


if __name__ == '__main__':
    main()
