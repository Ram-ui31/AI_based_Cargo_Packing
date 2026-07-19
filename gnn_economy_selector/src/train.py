"""
train.py -- trains PackageSetRanker on real (from build_dataset.py)
per-package placed/not-placed labels across a diverse set of Economy
selection strategies, all run through the FULL production CombinedPacker
on the SAME real 400-package instance.

Framing: all training scenes share the exact same 297-package candidate
set (only the SELECTION differs per strategy) -- this is intentional,
since the goal is a single specific instance's cost, not generalization to
new instances. Loss: per-scene weighted binary cross-entropy -- each
scene's contribution to the loss is weighted by how good that strategy's
REAL cost was (better cost = more trusted signal about which packages are
worth including), so the model learns "include packages resembling those
that appeared in the BEST real outcomes," not an unweighted average across
scenes of wildly differing quality.

Usage:
    python src/train.py
"""
from __future__ import annotations
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ga_cargo_packing')
sys.path.insert(0, GA_CARGO_ROOT)

from model import PackageSetRanker
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real_labels.jsonl')
CKPT_OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'ranker.pt')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--d-model', type=int, default=24)
    p.add_argument('--n-heads', type=int, default=2)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--quality-power', type=float, default=2.0)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=500)
    p.add_argument('--exclude-random', action='store_true',
                    help='drop random_seed* scenes (confirmed to hurt LOSO-AUC in sweep.py)')
    p.add_argument('--only-strategy', type=str, default=None,
                    help='restrict training to a single named strategy scene (e.g. value_density_pow1.5) -- '
                         'sanity check: pure single-best-demonstration imitation, should reproduce ~its own cost')
    p.add_argument('--ckpt-out', type=str, default=CKPT_OUT_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


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
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pid_to_i = {pid: i for i, pid in enumerate(economy_df['Package_ID'])}
    n_items = len(economy_df)

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
    feats, feat_mean, feat_std = normalize_features(feats)

    global_feats = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats, gmean, gstd = normalize_features(global_feats.reshape(1, -1))

    scenes = []
    with open(LABELS_PATH) as f:
        for line in f:
            scenes.append(json.loads(line))
    if args.exclude_random:
        scenes = [s for s in scenes if not s['strategy'].startswith('random_seed')]
    if args.only_strategy:
        scenes = [s for s in scenes if s['strategy'] == args.only_strategy]
    if not scenes:
        raise RuntimeError(f'No scenes found in {LABELS_PATH} -- run build_dataset.py first.')

    costs = np.array([s['cost'] for s in scenes], dtype=np.float32)
    min_c, max_c = costs.min(), costs.max()
    if max_c - min_c < 1e-6:
        # Degenerate: a single scene (--only-strategy) or all scenes tied on
        # cost -- the range-based formula below would divide by ~0 and give
        # every scene weight 0, silently killing the gradient. Uniform
        # weight instead: with nothing to contrast against, every scene
        # here is equally the only evidence available.
        quality = np.ones_like(costs)
    else:
        # Quality weight: 1.0 for the best scene, ->0 for the worst.
        quality = ((max_c - costs) / (max_c - min_c)) ** args.quality_power

    labels = np.zeros((len(scenes), n_items), dtype=np.float32)
    for s_idx, s in enumerate(scenes):
        for pid in s['placed_economy_ids']:
            if pid in pid_to_i:
                labels[s_idx, pid_to_i[pid]] = 1.0

    print(f'Loaded {len(scenes)} scenes, {n_items} economy packages each.')
    print(f'Cost range: {min_c:,.0f} (best: {scenes[int(np.argmin(costs))]["strategy"]}) '
          f'to {max_c:,.0f} (worst: {scenes[int(np.argmax(costs))]["strategy"]})')

    device = 'cpu'
    arch = dict(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
    model = PackageSetRanker(**arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pkg_feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(scenes), 1, 1)
    global_feats_t = torch.tensor(global_feats, dtype=torch.float32, device=device).repeat(len(scenes), 1)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    quality_t = torch.tensor(quality, dtype=torch.float32, device=device)

    bce = nn.BCEWithLogitsLoss(reduction='none')
    n_epochs = args.n_epochs
    best_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(pkg_feats_t, global_feats_t)  # (n_scenes, n_items)
        per_elem_loss = bce(scores, labels_t)         # (n_scenes, n_items)
        per_scene_loss = per_elem_loss.mean(dim=1)    # (n_scenes,)
        loss = (per_scene_loss * quality_t).sum() / quality_t.sum().clamp(min=1e-6)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(), 'arch': arch,
                'feat_mean': feat_mean, 'feat_std': feat_std,
                'gmean': gmean, 'gstd': gstd,
                'avg_uld_volume': avg_uld_volume, 'avg_uld_weight': avg_uld_weight,
            }, args.ckpt_out)

        if epoch % 25 == 0 or epoch == n_epochs - 1:
            print(f'epoch {epoch:4d}  quality-weighted BCE={loss.item():.4f}  best={best_loss:.4f}')

    print(f'\nSaved best checkpoint to {args.ckpt_out}')


if __name__ == '__main__':
    main()
