"""
run_cherry.py -- self-contained entry point: run the Cherry pipeline on
any CSV instance file in the same format as ~/Downloads/input.csv, using
only Cherry's own bundled checkpoints (checkpoints/priority_clusterer.pt,
checkpoints/rl_placement_policy.pt) and its own bundled source (src/).

Cherry is Eclipse's pipeline (see the sibling `eclipse/` folder) plus a
final centrifuge-evict refinement pass: for every already-placed Economy
package, tentatively evict it, compact its container (slide remaining
boxes toward one corner to consolidate freed space), and try to refill
from the unplaced pool; keep the single best net-cost-improving move each
cycle, repeat for a few cycles. This script runs the EXHAUSTIVE version of
that check (every placed Economy package is tried, not just a trained
model's top-K shortlist) -- correct by construction, at the cost of being
slower on large instances than the model-guided shortlist described in
this project's README.

The only file outside this folder it touches is the sibling `rl_packer/`
directory two levels up (shared 3D placement-policy geometry/environment
code, not weights -- see the repository root README), which must be
present alongside this repo exactly as cloned from GitHub.

Input CSV format (matches ~/Downloads/input.csv from the project):
    line 1        : K value (delay-cost multiplier), alone on its own line
    blank line
    ULD rows      : ULD_ID,Length,Width,Height,Weight_Limit
    blank line
    package rows  : Package_ID,Length,Width,Height,Weight,Type,Delay_Cost
                    (Type is "Priority" or "Economy"; Delay_Cost is "-"
                    for Priority packages)

Usage:
    python run_cherry.py --input /path/to/your_instance.csv
    python run_cherry.py --input /path/to/your_instance.csv --output-dir results_judge
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.model_core.config import DEVICE
from src.model_core.priority_clusterer_model import TransformerClusterer
from src.packer.reward import compute_packing_cost
from src.packer.rl_packer_adapter import RLPackerAdapter
from src.packer.heuristic_packer import HeuristicPacker, _import_geometry
from src.packer.combined_packer import CombinedPacker
import src.model_core.train_rl as tr

PRIORITY_CHECKPOINT = os.path.join(HERE, 'checkpoints', 'priority_clusterer.pt')
PLACEMENT_CHECKPOINT = os.path.join(HERE, 'checkpoints', 'rl_placement_policy.pt')


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
            uld_rows.append({'ULD_ID': uid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight_Limit': float(weight_limit)})
        else:
            pid, length, width, height, weight, ptype, delay = parts
            delay_cost = 0.0 if delay.strip() == '-' else float(delay)
            pkg_rows.append({'Package_ID': pid, 'Length': float(length), 'Width': float(width),
                              'Height': float(height), 'Weight': float(weight), 'Type': ptype,
                              'Delay_Cost': delay_cost})
    if not uld_rows:
        raise ValueError(f'No ULD rows parsed from {path} -- check the file matches the documented format.')
    if not pkg_rows:
        raise ValueError(f'No package rows parsed from {path} -- check the file matches the documented format.')
    return k_value, pd.DataFrame(uld_rows), pd.DataFrame(pkg_rows)


def build_packer(device='cpu'):
    return CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=PLACEMENT_CHECKPOINT, device=device)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
        ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ('dblf_ems', HeuristicPacker(strategy='dblf', origin_source='ems')),
    ])


def greedy_first_fit(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment):
    """Economy-only greedy first-fit by value-density order, into whatever
    capacity Priority consolidation left behind. Matches the exact
    methodology validated throughout this project (see e.g.
    model-training-pipeline/scripts and the online-3d-bpp-benchmark comparison
    scripts) -- deliberately NOT the same as rl_assign_argmax_safe's own
    internal Economy ordering, which sorts by ascending volume instead."""
    weight_used = {u: 0.0 for u in uld_lookup}
    volume_used = {u: 0.0 for u in uld_lookup}
    for pid, uid in prio_assignment.items():
        weight_used[uid] += pkg_lookup_all[pid]['Weight']
        volume_used[uid] += pkg_lookup_all[pid]['Length'] * pkg_lookup_all[pid]['Width'] * pkg_lookup_all[pid]['Height']
    assignment = dict(prio_assignment)
    for pid in ranked_pids:
        p = pkg_lookup_all[pid]
        pw, pv = p['Weight'], p['Length'] * p['Width'] * p['Height']
        placed = False
        for uid in uld_lookup:
            cap_w = uld_lookup[uid]['Weight_Limit']
            cap_v = uld_lookup[uid]['Length'] * uld_lookup[uid]['Width'] * uld_lookup[uid]['Height']
            if weight_used[uid] + pw <= cap_w + 1e-6 and volume_used[uid] + pv <= cap_v + 1e-6:
                weight_used[uid] += pw
                volume_used[uid] += pv
                assignment[pid] = uid
                placed = True
                break
        if not placed:
            assignment[pid] = 'NONE'
    return assignment


def local_search(assignment, pkgs_df, ulds_df, packer, k_value, rounds=15, seed=0, progress_cb=None):
    """Simple hill-climbing local search over the Economy assignment: each
    round tries a handful of real-evaluated swap/relocate moves and keeps
    the assignment if cost improves. A simplified, self-contained version
    of this project's research-grade local search -- sufficient to close
    most of the gap between a one-shot greedy assignment and the fully
    converged result in a few minutes, not a bit-exact reproduction of the
    original multi-day search."""
    import random
    rng = random.Random(seed)
    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    econ_pids = [pid for pid, p in pkg_lookup.items() if p['Type'] != 'Priority']
    uld_ids = list(ulds_df['ULD_ID'])

    def evaluate(assign):
        placements, _ = packer.pack(assign, pkgs_df, ulds_df)
        cost, *_ = compute_packing_cost(placements, pkgs_df, k_value)
        return cost, placements

    best_assignment = dict(assignment)
    best_cost, best_placements = evaluate(best_assignment)
    print(f'  round 0 (initial): cost={best_cost:,.0f}')

    for r in range(1, rounds + 1):
        candidate = dict(best_assignment)
        move = rng.choice(['swap', 'relocate'])
        a, b = rng.sample(econ_pids, 2)
        if move == 'swap':
            candidate[a], candidate[b] = candidate.get(b, 'NONE'), candidate.get(a, 'NONE')
        else:
            candidate[a] = rng.choice(uld_ids)
        cost, placements = evaluate(candidate)
        if cost < best_cost:
            best_cost, best_assignment, best_placements = cost, candidate, placements
            print(f'  round {r}: cost={cost:,.0f}  (improved)')
        else:
            print(f'  round {r}: cost={cost:,.0f}  (no improvement, kept previous best {best_cost:,.0f})')
        if progress_cb:
            progress_cb(r / rounds, f'Local search round {r}/{rounds} -- best cost so far: {best_cost:,.0f}')

    return best_assignment, best_placements, best_cost


def centrifuge_evict_refine(placements, pkgs_df, ulds_df, k_value, n_cycles=3, strategy='dblf', rl_packer_src=None, progress_cb=None):
    """Exhaustive evict-compact-refill refinement: for every placed Economy
    package, tentatively evict it, compact its container, and try to refill
    from the unplaced pool (including the evicted package itself); keep the
    single best net-cost-improving move found each cycle, repeat until no
    improving move is found or n_cycles is reached."""
    rl_packer_src = rl_packer_src or os.path.join(HERE, '..', 'rl_packer', 'src')
    geometry = _import_geometry(rl_packer_src)
    Heightmap = geometry.Heightmap
    filler = HeuristicPacker(rl_packer_src=rl_packer_src, strategy=strategy, origin_source='ems')

    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup.items():
        row['Package_ID'] = pid
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    by_uld, none_entries = {}, []
    for p in placements:
        (none_entries if p['ULD_ID'] == 'NONE' else by_uld.setdefault(p['ULD_ID'], [])).append(p)
    unplaced_pids = {p['Package_ID'] for p in none_entries if pkg_lookup[p['Package_ID']]['Type'] != 'Priority'}

    def refill_and_score(hm, candidate_pids):
        if not candidate_pids:
            return hm, set()
        cand_df = pd.DataFrame([pkg_lookup[pid] for pid in candidate_pids]).assign(
            _vol=lambda d: d['Length'] * d['Width'] * d['Height']).assign(
            _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5)).sort_values('_vdp', ascending=False)
        left_behind = set(filler._greedy_pack_into(hm, cand_df))
        return hm, candidate_pids - left_behind

    for cycle in range(n_cycles):
        best_move = None  # (uid, evict_pid, new_hm, newly_placed_pids)
        best_gain = 0.0

        for uid, plist in by_uld.items():
            econ_placed = [p for p in plist if pkg_lookup[p['Package_ID']]['Type'] != 'Priority']
            for evict_p in econ_placed:
                evict_pid = evict_p['Package_ID']
                evict_delay = pkg_lookup[evict_pid]['Delay_Cost']
                remaining = [p for p in plist if p['Package_ID'] != evict_pid]

                uld_row = uld_lookup[uid]
                hm = Heightmap(length=int(uld_row['Length']), width=int(uld_row['Width']),
                                height=int(uld_row['Height']), weight_limit=float(uld_row['Weight_Limit']))
                for p in remaining:
                    dx, dy, dz = p['x1'] - p['x0'], p['y1'] - p['y0'], p['z1'] - p['z0']
                    hm.place(p['Package_ID'], dx, dy, dz, p['x0'], p['y0'], p['z0'],
                              pkg_lookup[p['Package_ID']]['Weight'])
                weight_by_pid = {p['Package_ID']: pkg_lookup[p['Package_ID']]['Weight'] for p in remaining}
                compacted = hm.compact(weight_by_pid)

                candidates = unplaced_pids | {evict_pid}
                compacted, newly_placed = refill_and_score(compacted, candidates)
                gained_delay = sum(pkg_lookup[pid]['Delay_Cost'] for pid in newly_placed)
                evicted_stays_out = evict_pid not in newly_placed
                net_gain = gained_delay - (evict_delay if evicted_stays_out else 0.0)

                if net_gain > best_gain + 1e-6:
                    best_gain = net_gain
                    best_move = (uid, evict_pid, compacted, newly_placed)

        if best_move is None:
            print(f'  cycle {cycle + 1}: no improving evict-compact-refill move found, stopping')
            break

        uid, evict_pid, compacted, newly_placed = best_move
        print(f'  cycle {cycle + 1}: evict {evict_pid} from {uid}, compact, refill -- net gain {best_gain:,.0f}')
        if progress_cb:
            progress_cb((cycle + 1) / n_cycles, f'Centrifuge cycle {cycle + 1}/{n_cycles} -- evicted {evict_pid} from {uid}, net gain {best_gain:,.0f}')
        by_uld[uid] = [
            {'Package_ID': p.package_id, 'ULD_ID': uid, 'x0': p.x0, 'y0': p.y0, 'z0': p.z0,
             'x1': p.x1, 'y1': p.y1, 'z1': p.z1, 'reason': 'placed'}
            for p in compacted.placements
        ]
        unplaced_pids = (unplaced_pids | {evict_pid}) - newly_placed

    new_placements = [entry for plist in by_uld.values() for entry in plist]
    placed_pids = {e['Package_ID'] for e in new_placements}
    for pid in unplaced_pids:
        new_placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                'x0': -1, 'y0': -1, 'z0': -1, 'x1': -1, 'y1': -1, 'z1': -1,
                                'reason': 'centrifuge_left_behind'})
    for p in none_entries:
        if p['Package_ID'] not in placed_pids and p['Package_ID'] not in unplaced_pids:
            new_placements.append(p)
    return new_placements


def run_cherry(input_path, device='cpu', search_rounds=15, progress_cb=None):
    k_value, ulds_df, pkgs_df = parse_input_csv(input_path)

    n_prio = int((pkgs_df['Type'].str.strip().str.lower() == 'priority').sum())
    n_econ = int((pkgs_df['Type'].str.strip().str.lower() == 'economy').sum())
    print(f'Parsed {len(ulds_df)} ULDs, {len(pkgs_df)} packages '
          f'({n_prio} Priority, {n_econ} Economy), K={k_value:,.0f}')

    if progress_cb:
        progress_cb(0.0, 'Loading Priority Clusterer checkpoint...')
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    clusterer = TransformerClusterer().to(device)
    ckpt = torch.load(PRIORITY_CHECKPOINT, map_location=device, weights_only=False)
    clusterer.load_state_dict(ckpt['model_state_dict'], strict=True)
    clusterer.eval()

    t0 = time.time()
    if progress_cb:
        progress_cb(0.03, 'Assigning Priority packages to ULDs...')
    full0 = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, device, k_value,
                                      econ_sort_key='value_density_pow1.5')
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup_all.items():
        row['Package_ID'] = pid
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    # Only the Priority portion of the clusterer's assignment is used --
    # Economy gets its own explicit value-density order + greedy first-fit
    # below, matching this project's validated methodology exactly (see
    # model-training-pipeline's assignment stage).
    prio_assignment = {pid: uid for pid, uid in full0.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    econ_sorted = economy_df.assign(_vol=lambda d: d['Length'] * d['Width'] * d['Height']).assign(
        _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5)).sort_values('_vdp', ascending=False)
    base_order = econ_sorted['Package_ID'].tolist()
    assignment = greedy_first_fit(base_order, pkg_lookup_all, uld_lookup, prio_assignment)

    if progress_cb:
        progress_cb(0.06, 'Packing with the best-of-5 ensemble (RL policy + heuristics)...')
    packer = build_packer(device=device)
    if search_rounds > 0:
        print(f'Running local search ({search_rounds} rounds, real-evaluated)...')
        assignment, placements, _ = local_search(assignment, pkgs_df, ulds_df, packer, k_value, rounds=search_rounds,
                                                  progress_cb=(lambda frac, msg: progress_cb(0.06 + frac * 0.54, msg)) if progress_cb else None)
    else:
        placements, _ = packer.pack(assignment, pkgs_df, ulds_df)

    print('Running centrifuge-evict refinement (exhaustive)...')
    if progress_cb:
        progress_cb(0.6, 'Running centrifuge-evict refinement...')
    placements = centrifuge_evict_refine(placements, pkgs_df, ulds_df, k_value,
                                          progress_cb=(lambda frac, msg: progress_cb(0.6 + frac * 0.4, msg)) if progress_cb else None)
    elapsed = time.time() - t0

    cost, delay_cost, spread_cost, n_prio_ulds, unplaced_prio, unplaced_econ = \
        compute_packing_cost(placements, pkgs_df, k_value)

    print(f'\n=== Cherry result ===')
    print(f'Total cost: {cost:,.0f}  (delay={delay_cost:,.0f}, spread={spread_cost:,.0f})')
    print(f'Priority ULDs used: {n_prio_ulds}')
    print(f'Priority placed: {n_prio - len(unplaced_prio)}/{n_prio}'
          + (f'  *** WARNING: {len(unplaced_prio)} Priority packages unplaced -- '
             f'this should not happen; please report this instance. ***' if unplaced_prio else ''))
    print(f'Economy placed: {n_econ - len(unplaced_econ)}/{n_econ}')
    print(f'Wall time: {elapsed:.1f}s')

    return {
        'model': 'Cherry', 'k_value': k_value, 'total_cost': cost,
        'delay_cost': delay_cost, 'spread_cost': spread_cost,
        'n_priority_ulds': n_prio_ulds,
        'n_priority_total': n_prio, 'n_priority_unplaced': len(unplaced_prio),
        'n_economy_total': n_econ, 'n_economy_unplaced': len(unplaced_econ),
        'unplaced_priority_ids': unplaced_prio,
        'elapsed_seconds': elapsed,
    }, placements


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run the Cherry pipeline on a new CSV instance.')
    p.add_argument('--input', type=str, required=True, help='path to your instance CSV file')
    p.add_argument('--output-dir', type=str, default=os.path.join(HERE, 'results_judge'),
                    help='where to save final_metrics.json / final_placements.json (default: results_judge/)')
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    p.add_argument('--search-rounds', type=int, default=15,
                    help='local search rounds after the initial assignment (0 to skip; each round is a real, '
                         'full re-pack, so more rounds = better result but longer runtime)')
    args = p.parse_args()

    metrics, placements = run_cherry(args.input, device=args.device, search_rounds=args.search_rounds)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, 'final_placements.json'), 'w') as f:
        json.dump(placements, f, indent=2, default=str)
    print(f'\nSaved {args.output_dir}/final_metrics.json and final_placements.json')
