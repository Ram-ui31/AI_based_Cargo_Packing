"""
verify_pipeline.py — runs the GA/IL/RL pipeline (assignment via the RL-
fine-tuned TransformerClusterer, packing via rl_packer/RLPackerAdapter) over
a sample of synthetic_test instances and asserts the four stated conditions:

  1. Every Priority package is packed (0 unplaced Priority).
  2. No overlapping placements (redundant re-check on top of what
     Heightmap.fits() already enforces at placement time).
  3. Per-ULD weight and volume never exceed limits (also redundant on top of
     Heightmap.fits(), re-verified here from the raw placement output).
  4. Reports K*spread + sum(delay_cost of unplaced economy) per instance,
     and compares against common/results/eval_results*.csv baselines
     (H1H2+RL, IL-only) when present.

Usage:
    python eval/verify_pipeline.py \
        --data-root ~/Desktop/good_data \
        --checkpoint ../checkpoints/rl/transformer_rl_ga.pt \
        --n-instances 20
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

from src.rl.config import DEVICE                                   # noqa: E402
from src.rl.model import TransformerClusterer                       # noqa: E402
from src.rl.reward import compute_packing_cost                      # noqa: E402
from src.rl.rl_packer_adapter import RLPackerAdapter                # noqa: E402
from src.rl.adaptive_assign import rl_assign_argmax_adaptive        # noqa: E402


def _overlap(a, b) -> bool:
    return not (a['x1'] <= b['x0'] or b['x1'] <= a['x0'] or
                a['y1'] <= b['y0'] or b['y1'] <= a['y0'] or
                a['z1'] <= b['z0'] or b['z1'] <= a['z0'])


def verify_instance(model, packer, pkgs_df, ulds_df, k_value, device):
    assignment, placements, cost, total_unfit, _chosen = rl_assign_argmax_adaptive(
        model, pkgs_df, ulds_df, device, k_value, packer)
    _, delay_cost, spread_cost, n_priority_ulds, unplaced_prio, unplaced_econ = (
        compute_packing_cost(placements, pkgs_df, k_value)
    )

    violations = []
    if unplaced_prio:
        violations.append(f'{len(unplaced_prio)} Priority package(s) unplaced: {unplaced_prio[:5]}')

    placed = [p for p in placements if p['ULD_ID'] != 'NONE']
    by_uld = {}
    for p in placed:
        by_uld.setdefault(p['ULD_ID'], []).append(p)

    pkg_lookup = pkgs_df.set_index('Package_ID').to_dict('index')
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

    for uid, boxes in by_uld.items():
        uld = uld_lookup[uid]
        total_w = sum(pkg_lookup[b['Package_ID']]['Weight'] for b in boxes)
        total_v = sum(
            (b['x1'] - b['x0']) * (b['y1'] - b['y0']) * (b['z1'] - b['z0']) for b in boxes
        )
        if total_w > uld['Weight_Limit'] + 1e-6:
            violations.append(f'ULD {uid} weight overflow: {total_w:.1f} > {uld["Weight_Limit"]}')
        uld_vol = uld['Length'] * uld['Width'] * uld['Height']
        if total_v > uld_vol + 1e-6:
            violations.append(f'ULD {uid} volume overflow: {total_v:.0f} > {uld_vol}')
        for i in range(len(boxes)):
            b = boxes[i]
            if (b['x1'] > uld['Length'] + 1e-6 or b['y1'] > uld['Width'] + 1e-6
                    or b['z1'] > uld['Height'] + 1e-6 or b['x0'] < -1e-6
                    or b['y0'] < -1e-6 or b['z0'] < -1e-6):
                violations.append(f'ULD {uid} out-of-bounds placement: {b}')
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]):
                    violations.append(f'ULD {uid} overlap: {boxes[i]["Package_ID"]} vs {boxes[j]["Package_ID"]}')

    return {
        'cost': cost, 'delay_cost': delay_cost, 'spread_cost': spread_cost,
        'n_priority_ulds': n_priority_ulds, 'n_unplaced_priority': len(unplaced_prio),
        'n_unplaced_economy': len(unplaced_econ), 'total_unfit': total_unfit,
        'violations': violations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--checkpoint', required=True, help='transformer_rl_ga.pt or transformer_imitation_ga.pt')
    ap.add_argument('--n-instances', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--baselines-dir', default=None,
                     help='common/results/ dir with eval_results*.csv to compare against')
    args = ap.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    ckpt_path = os.path.abspath(os.path.expanduser(args.checkpoint))

    test_dir = os.path.join(data_root, 'synthetic_test')
    meta_k_path = os.path.join(test_dir, 'metadata_with_K.csv')
    meta = pd.read_csv(meta_k_path if os.path.exists(meta_k_path) else os.path.join(test_dir, 'metadata.csv'))
    if 'K' not in meta.columns:
        meta['K'] = 1000  # fallback; real runs always have metadata_with_K.csv

    rng = np.random.default_rng(args.seed)
    sample = meta.sample(n=min(args.n_instances, len(meta)), random_state=args.seed).reset_index(drop=True)

    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    packer = RLPackerAdapter()
    print(f'Checkpoint : {ckpt_path}')
    print(f'Packer     : rl_packer ({packer.weights_path})')
    print(f'Instances  : {len(sample)}\n')

    rows = []
    n_violations = 0
    for _, row in sample.iterrows():
        tag = row['instance']
        k_value = int(row['K'])
        pkgs_df = pd.read_csv(os.path.join(test_dir, f'{tag}_packages.csv'))
        ulds_df = pd.read_csv(os.path.join(test_dir, f'{tag}_ulds.csv'))

        result = verify_instance(model, packer, pkgs_df, ulds_df, k_value, DEVICE)
        status = 'OK' if not result['violations'] else f"{len(result['violations'])} VIOLATION(S)"
        print(f"{tag:>16}  K={k_value:>5}  cost={result['cost']:>10.1f}  "
              f"spread={result['n_priority_ulds']}  unplaced_econ={result['n_unplaced_economy']:>3}  {status}")
        for v in result['violations']:
            print(f'    ! {v}')
        n_violations += len(result['violations'])

        rows.append({
            'instance': tag, 'K': k_value, 'cost': result['cost'],
            'spread': result['n_priority_ulds'], 'delay_cost': result['delay_cost'],
            'n_unplaced_economy': result['n_unplaced_economy'],
            'priority_dropped': result['n_unplaced_priority'],
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(_THIS_DIR, 'verify_results.csv')
    df.to_csv(out_csv, index=False)

    print(f'\n{"="*72}')
    print(f'Mean cost           : {df["cost"].mean():.1f}')
    print(f'Mean spread         : {df["spread"].mean():.2f}')
    print(f'Total priority drops: {df["priority_dropped"].sum()}  (must be 0)')
    print(f'Total violations    : {n_violations}  (must be 0)')
    print(f'Results saved -> {out_csv}')

    if args.baselines_dir and os.path.isdir(args.baselines_dir):
        print(f'\n{"="*72}\nBaseline comparison ({args.baselines_dir}):')
        for fname in ['eval_results.csv', 'eval_results_hybrid.csv']:
            fpath = os.path.join(args.baselines_dir, fname)
            if os.path.exists(fpath):
                bdf = pd.read_csv(fpath)
                common = bdf[bdf['instance'].isin(df['instance'])]
                if len(common):
                    print(f'  {fname:<28} mean cost = {common["cost"].mean():.1f}  (n={len(common)} overlapping instances)')

    assert df['priority_dropped'].sum() == 0, 'FAIL: some Priority packages were dropped'
    assert n_violations == 0, 'FAIL: overlap/weight/volume/bounds violations found'
    print('\nAll 4 conditions verified with zero violations.')


if __name__ == '__main__':
    main()
