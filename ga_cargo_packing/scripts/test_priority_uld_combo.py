"""
test_priority_uld_combo.py -- tests whether WHICH 3 ULDs get chosen to hold
Priority affects final cost, holding Economy selection (value_density_
pow1.5) and packing (CombinedPacker) fixed.

Why: the current Priority consolidation (_consolidate_priority_by_capacity)
always picks the FEWEST, LARGEST-volume ULDs first (a fixed convention).
Weight is purely additive (unaffected by geometry/arrangement) -- so if
Priority's specific ULD choice differs from what the competitor does, the
PER-ULD remaining weight handed to Economy would differ across the fleet
even though total Priority weight is identical, which could fully explain
the weight-utilization gap found in report_uld_utilization.py (competitor
packs more weight into the same volume in 5/6 ULDs) WITHOUT needing any
change to Economy's selection formula at all -- a hypothesis not yet
tested, and orthogonal to the ~13 Economy-selection-side experiments
already tried and failed to beat 30,475.

Method: enumerate every 3-ULD combination, fast nominal (FFD trial-pack)
check for Priority feasibility, then for each FEASIBLE combo, force
Priority into exactly that combo (monkeypatching _consolidate_priority_by_
capacity, same technique already used in adaptive_assign.py) and run the
REAL pipeline (pow1.5 Economy selection + CombinedPacker) to get an actual
cost -- not a nominal proxy.

Usage:
    python scripts/test_priority_uld_combo.py
"""
from __future__ import annotations
import os
import sys
import itertools

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


def _ffd_trial_pack(candidate_ulds, prio_sorted, pkg_lookup, uld_lookup):
    """Same FFD nominal trial-pack logic as _consolidate_priority_by_capacity's
    internal _try_pack, exposed here for an EXPLICIT candidate ULD set."""
    weight_used = {u: 0.0 for u in candidate_ulds}
    volume_used = {u: 0.0 for u in candidate_ulds}
    assignment = {}
    for pid in prio_sorted:
        p = pkg_lookup[pid]
        pw = p['Weight']
        pv = p['Length'] * p['Width'] * p['Height']
        for u in candidate_ulds:
            cap_w = uld_lookup[u]['Weight_Limit']
            cap_v = uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height']
            if weight_used[u] + pw <= cap_w + 1e-6 and volume_used[u] + pv <= cap_v + 1e-6:
                weight_used[u] += pw
                volume_used[u] += pv
                assignment[pid] = u
                break
        else:
            return None
    return assignment, weight_used, volume_used


def main():
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    all_ulds = list(uld_lookup.keys())

    prio_ids = pkgs_df[pkgs_df['Type'].str.upper() == 'PRIORITY']['Package_ID'].tolist()
    prio_sorted = sorted(prio_ids, key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']))

    def _vol_desc(combo):
        # CRITICAL: FFD bin-packing is order-dependent -- must always try
        # the LARGEST ULD in a combo first, matching the exact convention
        # _consolidate_priority_by_capacity uses (ulds_by_vol_desc). An
        # earlier version of this script fed combos in itertools.combinations'
        # arbitrary (ID-ascending) order instead, which silently changed
        # which specific priority packages land in which ULD even for the
        # SAME 3-ULD set as production -- confirmed directly: order
        # [U3,U5,U6] left U3 at 2799/2800 (nearly maxed) and U6 at only
        # 1415/3500 (heavily underused), a much worse arrangement than
        # production's actual [U5,U6,U3] order (U5:3489, U6:3500, U3:725).
        return sorted(combo, key=lambda u: -(uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height']))

    # ── Fast nominal feasibility check across all 3-ULD combos ─────────────
    feasible_combos = []
    for combo in itertools.combinations(all_ulds, 3):
        result = _ffd_trial_pack(_vol_desc(combo), prio_sorted, pkg_lookup, uld_lookup)
        if result is not None:
            _, weight_used, volume_used = result
            feasible_combos.append(combo)
    print(f'{len(feasible_combos)}/20 combos nominally fit all {len(prio_ids)} Priority packages: {feasible_combos}\n')

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    orig_consolidate = tr._consolidate_priority_by_capacity

    def _make_forced_consolidate(combo):
        def _forced(packages_df, ulds_df_inner):
            result = _ffd_trial_pack(_vol_desc(combo), prio_sorted, pkg_lookup, uld_lookup)
            if result is None:
                return {}, {}, {}
            return result
        return _forced

    for combo in feasible_combos:
        tr.PRIORITY_CONSOLIDATION_MIN_K = -1
        tr._consolidate_priority_by_capacity = _make_forced_consolidate(combo)
        try:
            assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                                   econ_sort_key='value_density_pow1.5')
        finally:
            tr._consolidate_priority_by_capacity = orig_consolidate

        placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
        cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
            placements, pkgs_df, k_value)
        n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
        print(f'combo={combo!s:25s}: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
              f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')

    print(f'\nCurrent best (default largest-first combo): 30,475')
    print(f'Competitor target to beat: 29,203')


if __name__ == '__main__':
    main()
