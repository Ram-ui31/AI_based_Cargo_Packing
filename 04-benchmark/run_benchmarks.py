"""
Runs all three independent classical heuristics (FFD, LAFF, BFD) on the
real 400-package instance and the same 20 held-out synthetic instances used
throughout this project, saving results per heuristic under results/.
"""
import os
import json
import time
from collections import defaultdict

import pandas as pd

from heuristics import PACKERS, compute_cost, parse_input_csv, INSTANCES, GOOD_DATA

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')


def run_real_instance(name, packer):
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    t0 = time.time()
    placement = packer(pkgs_df, ulds_df)
    elapsed = time.time() - t0
    result = compute_cost(placement, pkgs_df, k_value)
    result['placement'] = placement
    result['elapsed_seconds'] = elapsed

    n_prio = int((pkgs_df['Type'] == 'Priority').sum())
    n_econ = int((pkgs_df['Type'] == 'Economy').sum())
    print(f"  [{name}] Total cost: {result['total_cost']:,.0f}  (delay={result['delay_cost']:,.0f}, "
          f"spread={result['spread_cost']:,.0f})  |  Priority {n_prio - len(result['unplaced_priority'])}/{n_prio}"
          f"  Economy {n_econ - result['unplaced_economy_count']}/{n_econ}  "
          f"|  ULDs used: {result['n_priority_ulds']}  |  {elapsed:.1f}s")

    with open(os.path.join(HERE, 'results', f'{name}_real_instance.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def run_20_instance_sweep(name, packer):
    meta = pd.read_csv(os.path.join(GOOD_DATA, 'metadata_with_K.csv')).set_index('instance')
    results = []
    t0 = time.time()
    for inst in INSTANCES:
        k_value = float(meta.loc[inst, 'K'])
        pkgs_df = pd.read_csv(os.path.join(GOOD_DATA, f'{inst}_packages.csv'))
        ulds_df = pd.read_csv(os.path.join(GOOD_DATA, f'{inst}_ulds.csv'))
        placement = packer(pkgs_df, ulds_df)
        r = compute_cost(placement, pkgs_df, k_value)
        results.append({'instance': inst, 'K': k_value, 'cost': r['total_cost'],
                         'delay_cost': r['delay_cost'], 'spread_cost': r['spread_cost'],
                         'unplaced_priority_count': len(r['unplaced_priority']),
                         'unplaced_economy_count': r['unplaced_economy_count']})
    elapsed = time.time() - t0

    with open(os.path.join(HERE, 'results', f'{name}_20instance.json'), 'w') as f:
        json.dump(results, f, indent=2)

    by_k = defaultdict(list)
    for r in results:
        by_k[r['K']].append(r['cost'])
    per_k_avg = {k: sum(v) / len(v) for k, v in by_k.items()}
    grand_avg = sum(per_k_avg.values()) / len(per_k_avg)
    total_unplaced_prio = sum(r['unplaced_priority_count'] for r in results)
    n_with_unplaced = sum(1 for r in results if r['unplaced_priority_count'] > 0)

    print(f"  [{name}] Grand average: {grand_avg:,.1f}  |  per-K: " +
          ', '.join(f"K={k:.0f}: {v:,.0f}" for k, v in sorted(per_k_avg.items())) +
          f"  |  priority-drops: {n_with_unplaced}/20 instances ({total_unplaced_prio} pkgs)  |  {elapsed:.0f}s")

    return {'per_k_avg': per_k_avg, 'grand_avg': grand_avg,
            'instances_with_unplaced_priority': n_with_unplaced,
            'total_unplaced_priority': total_unplaced_prio}


if __name__ == '__main__':
    os.chdir(HERE)
    summary = {}
    for name, packer in PACKERS.items():
        print(f"\n=== {name.upper()} ===")
        real_result = run_real_instance(name, packer)
        sweep_summary = run_20_instance_sweep(name, packer)
        summary[name] = {
            'real_instance': {
                'total_cost': real_result['total_cost'],
                'delay_cost': real_result['delay_cost'],
                'spread_cost': real_result['spread_cost'],
                'n_priority_ulds': real_result['n_priority_ulds'],
                'unplaced_priority_count': len(real_result['unplaced_priority']),
                'unplaced_economy_count': real_result['unplaced_economy_count'],
            },
            'sweep_20_instance': sweep_summary,
        }

    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Grand summary ===")
    for name in PACKERS:
        s = summary[name]
        print(f"{name.upper():5s}  real={s['real_instance']['total_cost']:>8,.0f}   "
              f"grand_avg={s['sweep_20_instance']['grand_avg']:>8,.1f}")
    print("\nSaved results/{ffd,laff,bfd}_{real_instance,20instance}.json and results/summary.json")
