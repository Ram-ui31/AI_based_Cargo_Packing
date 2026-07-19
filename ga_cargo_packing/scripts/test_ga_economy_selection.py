"""
test_ga_economy_selection.py -- tests replacing the greedy value_density_
pow1.5 Economy selection with a genetic algorithm (ga_economy_selector.py,
solve_economy_ga_real) that searches the full joint (package -> ULD | NONE)
assignment directly, seeded with the current best greedy solution so it
can only match or beat it under its own REAL (fast single-pass packer,
not nominal-proxy) fitness function.

Pipeline: Priority placed first (heuristic consolidation, same as
production) -> per-ULD Priority-only Heightmap kept as the GA's fitness
base (cloned per individual, never mutated) -> GA searches Economy
selection (seeded with the greedy pow1.5 solution), scoring each candidate
by really packing its assigned subset with a fast single-pass
HeuristicPacker (contact, max_pivots=20, no rescue) -> best individual's
Economy selection fed through the REAL, FULL CombinedPacker (3 strategies +
cross-ULD rescue) for final validation -- this is the only place the full
production packing quality appears; the GA's own fitness is real geometry
but a cheaper, single-pass approximation of it.

Usage:
    python scripts/test_ga_economy_selection.py
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
from src.rl.ga_economy_selector import solve_economy_ga_real
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
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    combined = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])
    candidates = combined.candidates

    tr.PRIORITY_CONSOLIDATION_MIN_K = -1  # matches production default on this instance

    # ── Priority phase only: best-of-3 per ULD, real remaining capacity ────
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}

    seed_full_assignment = tr.rl_assign_argmax_safe(model, pkgs_df, ulds_df, DEVICE, k_value,
                                                      econ_sort_key='value_density_pow1.5')
    uld_priority_ids = {uid: [] for uid in uld_lookup}
    for pid, uid in seed_full_assignment.items():
        if uid != 'NONE' and pkg_lookup[pid]['Type'] == 'Priority':
            uld_priority_ids[uid].append(pid)

    fast_packer = HeuristicPacker(strategy='contact', max_pivots=20)

    hm_by_uld = {}
    for uid, pids in uld_priority_ids.items():
        if not pids:
            # Empty (Priority-free) Heightmap -- still a real object the GA's
            # fitness function can clone, never None.
            row = uld_lookup[uid]
            hm_by_uld[uid] = fast_packer._geometry.Heightmap(
                length=int(row['Length']), width=int(row['Width']),
                height=int(row['Height']), weight_limit=float(row['Weight_Limit']),
            )
            continue
        best_hm, best_score = None, None
        for name, packer in candidates:
            hm, left = packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
            score = (len(left), -hm.utilization() if hm else 0.0)
            if best_score is None or score < best_score:
                best_hm, best_score = hm, score
        hm_by_uld[uid] = best_hm

    for uid, hm in hm_by_uld.items():
        row = uld_lookup[uid]
        cap_v = row['Length'] * row['Width'] * row['Height']
        print(f'{uid}: remaining weight={row["Weight_Limit"]-hm.weight_used:.0f}  '
              f'remaining volume={cap_v-hm.volume_used:,.0f}')
    print()

    # ── Seed assignment: the current best greedy (pow1.5) Economy selection ──
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    seed_assignment = {pid: uid for pid, uid in seed_full_assignment.items()
                        if uid != 'NONE' and pkg_lookup[pid]['Type'] != 'Priority'}

    # ── GA: search the full joint Economy selection, seeded with the above,
    #    fitness = REAL (not nominal) fast single-pass packing ─────────────
    uld_ids_list = list(uld_lookup.keys())
    t0 = time.time()
    ga_assignment = solve_economy_ga_real(economy_df, uld_ids_list, hm_by_uld, fast_packer,
                                           population=60, generations=200, patience=40,
                                           seed_assignment=seed_assignment, verbose=True)
    print(f'\n[ga-real] search time: {time.time()-t0:.1f}s, selected {len(ga_assignment)}/{len(economy_df)} '
          f'Economy packages (seed had {len(seed_assignment)})\n')

    # ── Merge Priority (original) + GA-selected Economy, feed through real packer ──
    full_assignment = {}
    for uid, pids in uld_priority_ids.items():
        for pid in pids:
            full_assignment[pid] = uid
    for pid in economy_df['Package_ID']:
        full_assignment[pid] = ga_assignment.get(pid, 'NONE')

    placements, total_unfit = combined.pack(full_assignment, pkgs_df, ulds_df)
    cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(placements, pkgs_df, k_value)
    n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
    print(f'GA + real packing: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  delay={delay_cost:,.0f}  '
          f'prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')
    print(f'Current best (greedy value_density_pow1.5): 30,475')
    print(f'Competitor target to beat: 29,203')


if __name__ == '__main__':
    main()
