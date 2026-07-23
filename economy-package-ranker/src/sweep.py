"""
sweep.py -- tries many PackageSetRanker configurations cheaply, using
leave-one-scene-out cross-validation (LOSO) as a proxy metric instead of
the expensive real CombinedPacker (each real validation costs ~3-10 min;
LOSO CV costs seconds since it's pure PyTorch on a tiny dataset).

For each config, for each of the 25 scenes: train on the OTHER 24, predict
scores for the held-out scene's 297 packages, compute AUC-ROC between
predicted score and that scene's real placed/not-placed label. Average AUC
across all 25 folds = the config's held-out ranking quality -- how well it
predicts real placement on data it didn't see, the standard proxy for "will
this generalize to a genuinely new sort order" (as good a signal as we can
get without paying for full real-packer validation each time).

Saves results to data/sweep_results.json and plots
data/sweep_results.png (matplotlib, no seaborn dependency).

Usage:
    python src/sweep.py
"""
from __future__ import annotations
import os
import sys
import json
import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PackageSetRanker
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real_labels.jsonl')
SWEEP_OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep_results.json')


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


def load_scenes():
    scenes = []
    with open(LABELS_PATH) as f:
        for line in f:
            scenes.append(json.loads(line))
    return scenes


def train_one_fold(pkg_feats, global_feats, labels, quality, arch, n_epochs, lr, weight_decay, seed=0):
    torch.manual_seed(seed)
    model = PackageSetRanker(**arch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    pkg_feats_t = torch.tensor(pkg_feats, dtype=torch.float32)
    global_feats_t = torch.tensor(global_feats, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    quality_t = torch.tensor(quality, dtype=torch.float32)
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(pkg_feats_t, global_feats_t)
        per_elem = bce(scores, labels_t)
        per_scene = per_elem.mean(dim=1)
        loss = (per_scene * quality_t).sum() / quality_t.sum().clamp(min=1e-6)
        loss.backward()
        optimizer.step()
    return model


def run_loso_cv(feats, global_feats_single, labels, quality, arch, n_epochs=200, lr=1e-3,
                weight_decay=1e-3, scene_subset=None):
    """Leave-one-scene-out CV. scene_subset: optional list of scene indices
    to restrict to (e.g., formula-only vs formula+random)."""
    n_scenes = labels.shape[0]
    indices = scene_subset if scene_subset is not None else list(range(n_scenes))
    aucs = []
    for held_out in indices:
        train_idx = [i for i in indices if i != held_out]
        pkg_feats_train = np.repeat(feats[None, :, :], len(train_idx), axis=0)
        global_feats_train = np.repeat(global_feats_single[None, :], len(train_idx), axis=0)
        labels_train = labels[train_idx]
        quality_train = quality[train_idx]

        model = train_one_fold(pkg_feats_train, global_feats_train, labels_train, quality_train,
                                arch, n_epochs, lr, weight_decay)
        model.eval()
        with torch.no_grad():
            held_scores = model(
                torch.tensor(feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(global_feats_single, dtype=torch.float32).unsqueeze(0),
            ).squeeze(0).numpy()
        held_labels = labels[held_out]
        if held_labels.min() == held_labels.max():
            continue  # degenerate (all placed or all dropped) -- AUC undefined
        aucs.append(roc_auc_score(held_labels, held_scores))
    return float(np.mean(aucs)) if aucs else float('nan')


def main():
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pid_to_i = {pid: i for i, pid in enumerate(economy_df['Package_ID'])}
    n_items = len(economy_df)

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
    feats, _, _ = normalize_features(feats)

    global_feats = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats, _, _ = normalize_features(global_feats.reshape(1, -1))
    global_feats = global_feats[0]

    scenes = load_scenes()
    costs = np.array([s['cost'] for s in scenes], dtype=np.float32)
    is_random = np.array([s['strategy'].startswith('random_seed') for s in scenes])
    formula_idx = list(np.where(~is_random)[0])
    all_idx = list(range(len(scenes)))

    labels = np.zeros((len(scenes), n_items), dtype=np.float32)
    for s_idx, s in enumerate(scenes):
        for pid in s['placed_economy_ids']:
            if pid in pid_to_i:
                labels[s_idx, pid_to_i[pid]] = 1.0

    def quality_for(idx_subset, power):
        c = costs[idx_subset]
        q_full = np.zeros(len(scenes), dtype=np.float32)
        qn = ((c.max() - c) / max(c.max() - c.min(), 1e-6)) ** power
        for j, i in enumerate(idx_subset):
            q_full[i] = qn[j]
        return q_full

    configs = [
        dict(name='A: d16_l1_qp1',  arch=dict(d_model=16, n_heads=2, n_layers=1, dropout=0.2), qp=1, data='formula'),
        dict(name='B: d24_l2_qp2',  arch=dict(d_model=24, n_heads=2, n_layers=2, dropout=0.3), qp=2, data='formula'),
        dict(name='C: d32_l2_qp1',  arch=dict(d_model=32, n_heads=4, n_layers=2, dropout=0.2), qp=1, data='formula'),
        dict(name='D: d64_l3_qp1',  arch=dict(d_model=64, n_heads=4, n_layers=3, dropout=0.15), qp=1, data='formula'),
        dict(name='E: d24_l2_qp3',  arch=dict(d_model=24, n_heads=2, n_layers=2, dropout=0.5), qp=3, data='formula'),
        dict(name='F: d16_l1_qp2',  arch=dict(d_model=16, n_heads=2, n_layers=1, dropout=0.2), qp=2, data='formula'),
        dict(name='G: d24_l2_qp2_allscenes', arch=dict(d_model=24, n_heads=2, n_layers=2, dropout=0.3), qp=2, data='all'),
        dict(name='H: d16_l1_qp1_allscenes', arch=dict(d_model=16, n_heads=2, n_layers=1, dropout=0.2), qp=1, data='all'),
    ]

    results = []
    for cfg in configs:
        t0 = time.time()
        idx_subset = formula_idx if cfg['data'] == 'formula' else all_idx
        quality = quality_for(idx_subset, cfg['qp'])
        auc = run_loso_cv(feats, global_feats, labels, quality, cfg['arch'], scene_subset=idx_subset)
        dt = time.time() - t0
        print(f"{cfg['name']:30s}  LOSO-AUC={auc:.4f}  ({dt:.1f}s)")
        results.append({'name': cfg['name'], 'auc': auc, 'arch': cfg['arch'], 'qp': cfg['qp'],
                         'data': cfg['data'], 'time_s': dt})

    os.makedirs(os.path.dirname(SWEEP_OUT_JSON), exist_ok=True)
    with open(SWEEP_OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved sweep results to {SWEEP_OUT_JSON}')
    best = max(results, key=lambda r: r['auc'])
    print(f"Best config by LOSO-AUC: {best['name']} (AUC={best['auc']:.4f})")


if __name__ == '__main__':
    main()
