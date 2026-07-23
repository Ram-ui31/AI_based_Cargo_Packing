"""
generate_comparison_data_ppo.py — GA vs PPO (checkpoints/rl_ppo/) cost/spread
per test instance, for the updated cost_vs_k.png / spread_vs_k.png plots.

Mirrors generate_comparison_data.py's GA-lookup + single-chunk-instance
filtering, but only loads the RL(PPO) checkpoint -- generate_comparison_data.py
also loads an IL checkpoint into src.rl.model.TransformerClusterer, which
now has the dual K-injection architecture added during the K-conditioning
work; the old (K-blind) IL checkpoint no longer strict-loads into that
class. Since this comparison only needs GA vs PPO, the IL step is skipped
entirely rather than worked around.

Usage:
    python eval/generate_comparison_data_ppo.py \
        --data-root ~/Desktop/good_data \
        --ga-cache cache/ga_cache.pkl \
        --rl-checkpoint checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt \
        --out results/comparison_ga_ppo.csv
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import pandas as pd
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

from src.rl.config import DEVICE                     # noqa: E402
from src.rl.model import TransformerClusterer         # noqa: E402
from src.rl.reward import compute_packing_cost         # noqa: E402
from src.rl.rl_packer_adapter import RLPackerAdapter    # noqa: E402
from src.rl.adaptive_assign import rl_assign_argmax_adaptive  # noqa: E402


def _load_model(checkpoint_path):
    model = TransformerClusterer().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--ga-cache', required=True)
    ap.add_argument('--rl-checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    test_dir = os.path.join(data_root, 'synthetic_test')
    meta = pd.read_csv(os.path.join(test_dir, 'metadata_with_K.csv'))

    with open(os.path.abspath(os.path.expanduser(args.ga_cache)), 'rb') as f:
        ga_cache = pickle.load(f)

    # Same "only genuinely single-chunk instances" filter as
    # generate_comparison_data.py -- a (tag, 0, 0) cache entry exists even
    # for multi-chunk instances (just that instance's first package chunk's
    # partial assignment), so only instances whose ONLY cache entry is
    # (0, 0) are complete, comparable GA solves.
    chunk_counts = {}
    for key_tag, ui, pi in ga_cache:
        chunk_counts.setdefault(key_tag, set()).add((ui, pi))
    single_chunk_tags = {t for t, chunks in chunk_counts.items() if chunks == {(0, 0)}}

    rl_model = _load_model(os.path.abspath(os.path.expanduser(args.rl_checkpoint)))
    packer = RLPackerAdapter()

    rows = []
    for _, row in meta.iterrows():
        tag = row['instance']
        k_value = int(row['K'])
        full_tag = f'synthetic_test/{tag}'
        if full_tag not in single_chunk_tags:
            continue
        ga_key = (full_tag, 0, 0)

        pkgs_df = pd.read_csv(os.path.join(test_dir, f'{tag}_packages.csv'))
        ulds_df = pd.read_csv(os.path.join(test_dir, f'{tag}_ulds.csv'))

        ga_assignment = ga_cache[ga_key]
        # RL uses rl_assign_argmax_adaptive rather than the fixed
        # PRIORITY_CONSOLIDATION_MIN_K=500 threshold -- verified the fixed
        # threshold picks the actually-cheaper option only ~50% of the time
        # since the true crossover point is instance-specific; adaptive
        # computes both {heuristic, model} candidates and keeps whichever is
        # cheaper for THIS instance, a strict min-of-two that can only match
        # or beat the fixed threshold.
        _, rl_placements, rl_cost, _, _ = rl_assign_argmax_adaptive(
            rl_model, pkgs_df, ulds_df, DEVICE, k_value, packer)

        for method in ('GA', 'RL'):
            if method == 'GA':
                placements, _ = packer.pack(ga_assignment, pkgs_df, ulds_df)
            else:
                placements = rl_placements
            cost, delay_cost, spread_cost, n_priority_ulds, unplaced_prio, unplaced_eco = (
                compute_packing_cost(placements, pkgs_df, k_value)
            )
            rows.append({
                'instance': tag, 'K': k_value, 'method': method,
                'cost': cost, 'delay_cost': delay_cost, 'spread': n_priority_ulds,
                'priority_dropped': len(unplaced_prio), 'economy_dropped': len(unplaced_eco),
            })

        print(f'{tag} (K={k_value}) done', flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows ({df["instance"].nunique()} instances x 2 methods) -> {out_path}')


if __name__ == '__main__':
    main()
