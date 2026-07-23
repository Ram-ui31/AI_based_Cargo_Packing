"""
save_h1h2_packing_results.py — runs the h1_h2 heuristic's own packer
(GreedyPipeline, extreme-point placement) on every synthetic_test instance
and saves the full result: real x,y,z coordinates for every placed package,
plus the cost breakdown, per instance.

K is assigned the same way as the rest of the project (seed=42, 200 instances
per K value out of synthetic_train's 1000, synthetic_test's tags inherit
their train counterpart's K since the tag strings overlap).

Usage:
    python results/save_h1h2_packing_results.py
"""
import json
import os
import random
import sys

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
sys.path.insert(0, _SRC_DIR)

from geometry import Package, ULD
from greedy_pipeline import GreedyPipeline

DATA_ROOT = os.path.expanduser('~/Desktop/good_data')
K_VALUES = [100, 500, 1000, 3000, 5000]


def build_k_map(data_root, seed=42):
    rng = random.Random(seed)
    meta = pd.read_csv(os.path.join(data_root, 'synthetic_train', 'metadata.csv'))
    instances = meta['instance'].tolist()
    rng.shuffle(instances)
    per_k = len(instances) // len(K_VALUES)
    k_map = {}
    for i, k_val in enumerate(K_VALUES):
        for tag in instances[i * per_k:(i + 1) * per_k]:
            k_map[tag] = k_val
    return k_map


def pack_instance(pkgs_df, ulds_df, k_value):
    packages = [
        Package(id=r.Package_ID, length=r.Length, width=r.Width, height=r.Height,
                weight=r.Weight, is_priority=(r.Type == 'Priority'), delay_cost=r.Delay_Cost)
        for r in pkgs_df.itertuples()
    ]
    ulds = [
        ULD(id=r.ULD_ID, length=r.Length, width=r.Width, height=r.Height, weight_limit=r.Weight_Limit)
        for r in ulds_df.itertuples()
    ]
    result = GreedyPipeline(ulds=ulds, packages=packages, k_penalty=k_value).solve()

    delay_costs = {p.id: p.delay_cost for p in packages}
    cost = result.total_cost(delay_costs, k_value)
    n_priority_ulds = result.uld_priority_count()
    delay_cost = sum(delay_costs.get(pid, 0.0) for pid in result.left_behind)
    spread_cost = k_value * n_priority_ulds

    placements = [
        {'Package_ID': b.package_id, 'ULD_ID': b.uld_id,
         'x0': b.x0, 'y0': b.y0, 'z0': b.z0, 'x1': b.x1, 'y1': b.y1, 'z1': b.z1}
        for b in result.placed_boxes
    ]
    unplaced = [{'Package_ID': pid, 'ULD_ID': 'NONE'} for pid in result.left_behind + result.unplaceable]

    n_priority = int((pkgs_df['Type'] == 'Priority').sum())
    n_economy = len(pkgs_df) - n_priority

    return {
        'K': k_value,
        'cost': cost,
        'delay_cost': delay_cost,
        'spread_cost': spread_cost,
        'n_priority_ulds': n_priority_ulds,
        'n_priority': n_priority,
        'n_economy': n_economy,
        'n_priority_unplaced': len(result.unplaceable),
        'n_economy_unplaced': len(result.left_behind),
        'placements': placements + unplaced,
    }


def main():
    k_map = build_k_map(DATA_ROOT)
    test_dir = os.path.join(DATA_ROOT, 'synthetic_test')
    test_meta = pd.read_csv(os.path.join(test_dir, 'metadata.csv'))

    all_results = {}
    for _, row in test_meta.iterrows():
        tag = row['instance']
        k_value = k_map.get(tag, K_VALUES[0])
        pkgs_df = pd.read_csv(os.path.join(test_dir, f'{tag}_packages.csv'))
        ulds_df = pd.read_csv(os.path.join(test_dir, f'{tag}_ulds.csv'))
        all_results[tag] = pack_instance(pkgs_df, ulds_df, k_value)
        print(f'{tag} (K={k_value}): cost={all_results[tag]["cost"]:.0f}  '
              f'placed={len(all_results[tag]["placements"]) - all_results[tag]["n_priority_unplaced"] - all_results[tag]["n_economy_unplaced"]}')

    out_path = os.path.join(_THIS_DIR, 'h1h2_packing_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved {len(all_results)} instances -> {out_path}')

    summary_rows = [
        {'instance': tag, 'K': r['K'], 'cost': r['cost'], 'delay_cost': r['delay_cost'],
         'spread_cost': r['spread_cost'], 'n_priority_ulds': r['n_priority_ulds'],
         'n_priority': r['n_priority'], 'n_economy': r['n_economy'],
         'n_priority_unplaced': r['n_priority_unplaced'], 'n_economy_unplaced': r['n_economy_unplaced']}
        for tag, r in all_results.items()
    ]
    summary_csv = os.path.join(_THIS_DIR, 'h1h2_packing_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f'Saved per-instance summary -> {summary_csv}')


if __name__ == '__main__':
    main()
