"""
infer_and_validate.py -- loads the trained PackageSetRanker, uses its
scores as a greedy-first-fit ORDER (exactly the same mechanism as the
hand-tuned econ_sort_key formulas -- this model is a drop-in replacement
for the sort key, not a new packing algorithm), and validates the result
through the REAL, FULL production CombinedPacker on the real 400-package
instance.

Usage:
    python src/infer_and_validate.py
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model-training-pipeline')
sys.path.insert(0, GA_CARGO_ROOT)

# NOTE: both model-training-pipeline and this project use "src" as their package
# name -- importing model-training-pipeline's src.rl.* first (with GA_CARGO_ROOT on
# sys.path) before adding THIS project's own src/ directory (unprefixed
# imports below) avoids the naming collision between the two "src" packages.
from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PackageSetRanker
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
RANKER_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'ranker.pt')


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=RANKER_CKPT)
    p.add_argument('--label', type=str, default=None, help='printed name for this run')
    return p.parse_args()


def main():
    args = parse_args()
    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    arch = dict(ckpt['arch'])
    arch['dropout'] = 0.0
    ranker = PackageSetRanker(**arch)
    ranker.load_state_dict(ckpt['model_state_dict'])
    ranker.eval()

    feats = build_package_features(economy_df, ckpt['avg_uld_volume'], ckpt['avg_uld_weight'])
    feats, _, _ = normalize_features(feats, ckpt['feat_mean'], ckpt['feat_std'])
    global_feats = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats, _, _ = normalize_features(global_feats.reshape(1, -1), ckpt['gmean'], ckpt['gstd'])

    with torch.no_grad():
        scores = ranker(
            torch.tensor(feats, dtype=torch.float32).unsqueeze(0),
            torch.tensor(global_feats, dtype=torch.float32),
        ).squeeze(0).numpy()

    order = np.argsort(-scores)
    ranked_pids = economy_df.loc[order, 'Package_ID'].tolist()
    rank_of_pid = {pid: i for i, pid in enumerate(ranked_pids)}

    # Monkeypatch a new econ_sort_key handled entirely here: temporarily
    # patch rl_assign_argmax_safe's econ ordering by injecting a
    # precomputed rank column the same way train_rl.py's existing
    # econ_sort_key branches do internally -- simplest correct approach:
    # replicate the final Economy greedy first-fit loop directly here using
    # the model's ranking, reusing tr's helper machinery for Priority.
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    full_assignment = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                econ_sort_key='value_density_pow1.5')
    # Only the ECONOMY portion needs replacing with the model's own order;
    # Priority placement is untouched (already confirmed optimal).
    prio_assignment = {pid: uid for pid, uid in full_assignment.items()
                        if uid != 'NONE' and pkgs_df.set_index('Package_ID').loc[pid, 'Type'] == 'Priority'}
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    weight_used = {u: 0.0 for u in uld_lookup}
    volume_used = {u: 0.0 for u in uld_lookup}
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    for pid, uid in prio_assignment.items():
        weight_used[uid] += pkg_lookup_all[pid]['Weight']
        volume_used[uid] += pkg_lookup_all[pid]['Length'] * pkg_lookup_all[pid]['Width'] * pkg_lookup_all[pid]['Height']

    model_assignment = dict(prio_assignment)
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
                model_assignment[pid] = uid
                placed = True
                break
        if not placed:
            model_assignment[pid] = 'NONE'

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])
    placements, total_unfit = packer.pack(model_assignment, pkgs_df, ulds_df)
    cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
        placements, pkgs_df, k_value)
    n_placed = sum(1 for p in placements if p['ULD_ID'] != 'NONE')
    label = args.label or args.ckpt
    print(f'[{label}] GNN ranker + real packing: cost={cost:,.0f}  placed={n_placed}  spread={n_prio}  '
          f'delay={delay_cost:,.0f}  prio_drop={len(unplaced_prio)}  econ_drop={len(unplaced_eco)}')
    print(f'Current best (greedy value_density_pow1.5): 30,475')
    print(f'Competitor target to beat: 29,203')


if __name__ == '__main__':
    main()
