"""
generate_centrifuge_data.py -- builds labeled training data for the
CentrifugeEvictProposer: for a sample of synthetic instances (from
good_data/synthetic_train), pack each with the model-training-pipeline heuristic
ensemble, then exhaustively test every valid "evict one placed Economy
package -> compact its container -> refill from the unplaced pool" move,
recording the full context (container contents, evict candidate, unplaced
pool, ULD/global stats) plus the REAL net delay-cost gain as the label.

This mirrors exactly the move validated by hand on the real 400-package
benchmark (model-training-pipeline/scripts -- see aeropack README's Open Work
section): a real, generalizable, consistently-positive-but-small move
family (~12% of exhaustively-tested evictions are profitable, confirmed
across 10 held-out synthetic instances). The exhaustive geometric check
(build heightmap, compact, greedy refill) is the expensive step this
model is meant to replace at inference time -- so training labels come
from running that exact expensive check now, offline, across many
instances, to get enough (candidate, net_gain) pairs to learn from.

Output: one JSONL file, one line per (instance, uld, evict_candidate)
example:
    {
      "instance": "instance_042",
      "uld": {"length":.., "width":.., "height":.., "weight_limit":..},
      "container_pkgs": [{"length":.., "width":.., "height":.., "weight":.., "delay_cost":.., "is_priority":bool}, ...],
      "evict_pkg": {...},
      "unplaced_pool": [{...}, ...],   # capped, sorted by value density desc
      "k_value": ..,
      "net_gain": ..   # REAL label: gained_delay - evict_delay after compact+refill
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

GA_ROOT = os.path.expanduser('~/Desktop/cargoism/git/model-training-pipeline')
sys.path.insert(0, GA_ROOT)

from src.rl.heuristic_packer import HeuristicPacker, _import_geometry
from src.rl.combined_packer import CombinedPacker
from src.rl.reward import compute_packing_cost

GOOD_DATA_ROOT = os.path.expanduser('~/Desktop/good_data')
UNPLACED_POOL_CAP = 40  # cap tokens fed to the model's pool encoder


def _geometry():
    return _import_geometry(os.path.join(GA_ROOT, '..', 'rl_packer', 'src'))


def greedy_first_fit_all(pkgs_df, uld_lookup, pkg_lookup_all):
    prio_pids = list(pkgs_df[pkgs_df['Type'] == 'Priority']['Package_ID'])
    econ_df = pkgs_df[pkgs_df['Type'] != 'Priority'].copy()
    econ_df['_vol'] = econ_df['Length'] * econ_df['Width'] * econ_df['Height']
    econ_df['_vdp'] = econ_df['Delay_Cost'] / econ_df['_vol'].clip(lower=1) ** 1.5
    econ_pids = list(econ_df.sort_values('_vdp', ascending=False)['Package_ID'])

    weight_used = {u: 0.0 for u in uld_lookup}
    volume_used = {u: 0.0 for u in uld_lookup}
    assignment = {}
    for pid in prio_pids + econ_pids:
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


def pkg_record(pkg_row):
    return {
        'length': float(pkg_row['Length']), 'width': float(pkg_row['Width']), 'height': float(pkg_row['Height']),
        'weight': float(pkg_row['Weight']), 'delay_cost': float(pkg_row['Delay_Cost']),
        'is_priority': pkg_row['Type'] == 'Priority',
    }


def process_instance(inst, meta_row, Heightmap, filler, packer, out_f):
    pkgs_df = pd.read_csv(os.path.join(GOOD_DATA_ROOT, 'synthetic_train', f'{inst}_packages.csv'))
    ulds_df = pd.read_csv(os.path.join(GOOD_DATA_ROOT, 'synthetic_train', f'{inst}_ulds.csv'))
    k_value = float(meta_row['K'])
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, row in pkg_lookup_all.items():
        row['Package_ID'] = pid
    uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}

    assignment = greedy_first_fit_all(pkgs_df, uld_lookup, pkg_lookup_all)
    placements, _ = packer.pack(assignment, pkgs_df, ulds_df)

    by_uld = {}
    none_entries = []
    for p in placements:
        if p['ULD_ID'] == 'NONE':
            none_entries.append(p)
        else:
            by_uld.setdefault(p['ULD_ID'], []).append(p)
    unplaced_econ_pids = [p['Package_ID'] for p in none_entries if pkg_lookup_all[p['Package_ID']]['Type'] != 'Priority']
    if not unplaced_econ_pids:
        return 0

    cand_df_base = pd.DataFrame([pkg_lookup_all[pid] for pid in unplaced_econ_pids])
    cand_df_base = cand_df_base.assign(_vol=lambda d: d['Length'] * d['Width'] * d['Height']).assign(
        _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5)).sort_values('_vdp', ascending=False)
    unplaced_pool_records = [pkg_record(pkg_lookup_all[pid]) for pid in cand_df_base['Package_ID'].tolist()[:UNPLACED_POOL_CAP]]

    n_written = 0
    for uid, plist in by_uld.items():
        uld_row = uld_lookup[uid]
        econ_placed = [p for p in plist if pkg_lookup_all[p['Package_ID']]['Type'] != 'Priority']
        if not econ_placed:
            continue
        for evict_p in econ_placed:
            evict_pid = evict_p['Package_ID']
            evict_delay = pkg_lookup_all[evict_pid]['Delay_Cost']
            remaining = [p for p in plist if p['Package_ID'] != evict_pid]
            container_pkgs = [pkg_record(pkg_lookup_all[p['Package_ID']]) for p in remaining]
            hm = Heightmap(length=int(uld_row['Length']), width=int(uld_row['Width']), height=int(uld_row['Height']),
                           weight_limit=float(uld_row['Weight_Limit']))
            for p in remaining:
                pid = p['Package_ID']
                dx, dy, dz = p['x1'] - p['x0'], p['y1'] - p['y0'], p['z1'] - p['z0']
                hm.place(pid, dx, dy, dz, p['x0'], p['y0'], p['z0'], pkg_lookup_all[pid]['Weight'])
            weight_by_pid = {p['Package_ID']: pkg_lookup_all[p['Package_ID']]['Weight'] for p in remaining}
            compacted = hm.compact(weight_by_pid)
            before_ids = {p.package_id for p in compacted.placements}
            filler._greedy_pack_into(compacted, cand_df_base)
            after_ids = {p.package_id for p in compacted.placements}
            newly_placed = after_ids - before_ids
            gained_delay = sum(pkg_lookup_all[pid]['Delay_Cost'] for pid in newly_placed)
            net_gain = gained_delay - evict_delay

            record = {
                'instance': inst,
                'uld': {'length': float(uld_row['Length']), 'width': float(uld_row['Width']),
                        'height': float(uld_row['Height']), 'weight_limit': float(uld_row['Weight_Limit'])},
                'container_pkgs': container_pkgs,
                'evict_pkg': pkg_record(pkg_lookup_all[evict_pid]),
                'unplaced_pool': unplaced_pool_records,
                'k_value': k_value,
                'net_gain': net_gain,
            }
            out_f.write(json.dumps(record) + '\n')
            n_written += 1
    return n_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-instances', type=int, default=150)
    p.add_argument('--out', type=str, default='data/centrifuge_train.jsonl')
    args = p.parse_args()

    meta = pd.read_csv(os.path.join(GOOD_DATA_ROOT, 'synthetic_train', 'metadata_with_K.csv')).sort_values('n_packages')
    step = max(1, len(meta) // args.n_instances)
    sample = meta.iloc[::step].head(args.n_instances)
    print(f'Sampling {len(sample)} instances (n_packages range {sample["n_packages"].min()}-{sample["n_packages"].max()})')

    geometry = _geometry()
    Heightmap = geometry.Heightmap
    filler = HeuristicPacker(strategy='dblf', origin_source='ems')
    packer = CombinedPacker([
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
        ('contact_ems', HeuristicPacker(strategy='contact', origin_source='ems')),
        ('dblf_ems', HeuristicPacker(strategy='dblf', origin_source='ems')),
    ])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    total_examples, total_wins = 0, 0
    with open(args.out, 'w') as out_f:
        for i, (_, meta_row) in enumerate(sample.iterrows()):
            inst = meta_row['instance']
            n = process_instance(inst, meta_row, Heightmap, filler, packer, out_f)
            total_examples += n
            elapsed = time.time() - t0
            print(f'[{i+1}/{len(sample)}] {inst}: {n} examples written (total={total_examples}) [{elapsed:.0f}s elapsed]')
            out_f.flush()

    print(f'\nDone. {total_examples} total examples written to {args.out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
