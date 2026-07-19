"""
bootstrap_loop.py -- iterative self-improvement loop for PackageSetRanker,
one real evaluation per round (not full RL's thousands of steps).

Each round:
  1. Train the ranker on all real-labeled scenes collected so far (starts
     with the 15 formula-only demonstrations from build_dataset.py).
  2. Take the model's OWN top-ranked ordering (its current best hypothesis
     about which packages matter).
  3. Real-evaluate that ordering ONCE through the full production
     CombinedPacker (~3-10 min) -- the only place real geometry appears.
  4. Add that (selection, real outcome) as a NEW training scene, whether it
     turned out better or worse than anything seen before.
  5. Repeat.

Why this escapes the ceiling found in sweep.py: every prior GNN attempt
only imitated 15 fixed, hand-picked formula demonstrations, so its ceiling
was bounded by what those 15 already contained (measured hard ceiling:
~31,000, see full_comparison.png). Here, every round injects a genuinely
NEW data point reflecting the model's OWN evolving hypothesis, verified
for real -- not a re-mix of the same fixed formulas. Over many rounds this
can discover combinations no single formula or supervised-only fit could,
without paying for full RL's per-step cost.

Keeps the single best real-cost checkpoint ever seen across all rounds
(best_ever.pt), separate from the latest round's checkpoint, since later
rounds are not guaranteed to improve monotonically.

Usage:
    python src/bootstrap_loop.py --rounds 15
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GA_CARGO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ga_cargo_packing')
sys.path.insert(0, GA_CARGO_ROOT)

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.heuristic_packer import HeuristicPacker
from src.rl.combined_packer import CombinedPacker
import src.rl.train_rl as tr

from model import PackageSetRanker
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CHECKPOINT = os.path.join(GA_CARGO_ROOT, 'checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
DENSITY_PACKER_CKPT = os.path.join(
    GA_CARGO_ROOT, '..', '..', 'uld_heightmap_rl', 'checkpoints', 'rl_packer', 'placement_policy_density.pt',
)
SEED_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real_labels.jsonl')
BOOTSTRAP_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'bootstrap_labels.jsonl')
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
BEST_EVER_CKPT = os.path.join(CKPT_DIR, 'best_ever.pt')
BEST_EVER_META = os.path.join(CKPT_DIR, 'best_ever_meta.json')

ARCH = dict(d_model=24, n_heads=2, n_layers=2, dropout=0.3)
QUALITY_POWER = 2.0


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
    with open(SEED_LABELS_PATH) as f:
        for line in f:
            s = json.loads(line)
            if not s['strategy'].startswith('random_seed'):  # confirmed harmful in sweep.py
                scenes.append(s)
    if os.path.exists(BOOTSTRAP_LABELS_PATH):
        with open(BOOTSTRAP_LABELS_PATH) as f:
            for line in f:
                scenes.append(json.loads(line))
    return scenes


def train_ranker(feats, global_feats, scenes, pid_to_i, n_items, seed, n_epochs=500,
                  init_state_dict=None, lr=1e-3):
    """
    init_state_dict : if given, the model WARM-STARTS from these weights
        (typically the best-ever checkpoint) instead of a fresh random
        init -- turns the bootstrap loop into genuine hill-climbing
        (incremental fine-tuning on the growing dataset) instead of an
        independent random re-roll every round, which is what caused the
        round-to-round variance to look more like noise than a trend
        (confirmed: seed-to-seed spread ~750-860 points, comparable to the
        differences we were attributing to architecture choice).
    """
    torch.manual_seed(seed)
    costs = np.array([s['cost'] for s in scenes], dtype=np.float32)
    min_c, max_c = costs.min(), costs.max()
    quality = ((max_c - costs) / max(max_c - min_c, 1e-6)) ** QUALITY_POWER

    labels = np.zeros((len(scenes), n_items), dtype=np.float32)
    for s_idx, s in enumerate(scenes):
        for pid in s['placed_economy_ids']:
            if pid in pid_to_i:
                labels[s_idx, pid_to_i[pid]] = 1.0

    model = PackageSetRanker(**ARCH)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    pkg_feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).repeat(len(scenes), 1, 1)
    global_feats_t = torch.tensor(global_feats, dtype=torch.float32).repeat(len(scenes), 1)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    quality_t = torch.tensor(quality, dtype=torch.float32)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(pkg_feats_t, global_feats_t)
        per_elem = bce(scores, labels_t)
        per_scene = per_elem.mean(dim=1)
        loss = (per_scene * quality_t).sum() / quality_t.sum().clamp(min=1e-6)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, loss.item()


def real_evaluate_ranking(ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment, pkgs_df, ulds_df, k_value, packer):
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

    placements, total_unfit = packer.pack(assignment, pkgs_df, ulds_df)
    cost, delay_cost, spread_cost, n_prio, unplaced_prio, unplaced_eco = compute_packing_cost(
        placements, pkgs_df, k_value)
    placed_ids = {p['Package_ID'] for p in placements if p['ULD_ID'] != 'NONE'}
    return cost, delay_cost, len(unplaced_eco), placed_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--seed-start', type=int, default=100)  # avoid colliding with sweep.py's seeds 0-2
    args = p.parse_args()

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pkg_lookup_all = pkgs_df.set_index('Package_ID').to_dict('index')
    pid_to_i = {pid: i for i, pid in enumerate(economy_df['Package_ID'])}
    n_items = len(economy_df)
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')

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

    clusterer = TransformerClusterer().to(DEVICE)
    clusterer_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    clusterer.load_state_dict(clusterer_ckpt['model_state_dict'], strict=True)
    clusterer.eval()
    tr.PRIORITY_CONSOLIDATION_MIN_K = -1
    full_assignment = tr.rl_assign_argmax_safe(clusterer, pkgs_df, ulds_df, DEVICE, k_value,
                                                econ_sort_key='value_density_pow1.5')
    prio_assignment = {pid: uid for pid, uid in full_assignment.items()
                        if uid != 'NONE' and pkg_lookup_all[pid]['Type'] == 'Priority'}

    packer = CombinedPacker([
        ('rl', RLPackerAdapter(weights_path=DENSITY_PACKER_CKPT)),
        ('contact', HeuristicPacker(strategy='contact')),
        ('dblf', HeuristicPacker(strategy='dblf')),
    ])

    best_ever_cost = float('inf')
    warm_start_state = None
    if os.path.exists(BEST_EVER_META):
        with open(BEST_EVER_META) as f:
            best_ever_cost = json.load(f)['cost']
        print(f'Resuming: best-ever real cost so far = {best_ever_cost:,.0f}')
        if os.path.exists(BEST_EVER_CKPT):
            warm_start_state = torch.load(BEST_EVER_CKPT, map_location='cpu', weights_only=False)['model_state_dict']
            print('Warm-starting from best_ever.pt (hill-climbing, not fresh random init each round)')

    os.makedirs(CKPT_DIR, exist_ok=True)
    for round_idx in range(args.rounds):
        scenes = load_scenes()
        t0 = time.time()
        seed = args.seed_start + round_idx
        # Warm-start from whatever the best-ever checkpoint is AT THE START of
        # this round (elitist hill-climbing: build forward only from the best
        # known point, never drift from a bad round) -- fresh random init only
        # for the very first round ever run (no best-ever checkpoint exists yet).
        lr = 1e-3 if warm_start_state is None else 3e-4
        model, train_loss = train_ranker(feats, global_feats, scenes, pid_to_i, n_items, seed,
                                          init_state_dict=warm_start_state, lr=lr)
        train_dt = time.time() - t0

        with torch.no_grad():
            scores = model(
                torch.tensor(feats, dtype=torch.float32).unsqueeze(0),
                torch.tensor(global_feats, dtype=torch.float32),
            ).squeeze(0).numpy()
        order = np.argsort(-scores)
        ranked_pids = economy_df.loc[order, 'Package_ID'].tolist()

        t1 = time.time()
        cost, delay_cost, econ_drop, placed_ids = real_evaluate_ranking(
            ranked_pids, pkg_lookup_all, uld_lookup, prio_assignment, pkgs_df, ulds_df, k_value, packer)
        eval_dt = time.time() - t1

        strategy_name = f'bootstrap_round{round_idx}_seed{seed}'
        new_scene = {
            'strategy': strategy_name, 'cost': cost, 'delay_cost': delay_cost,
            'econ_drop': econ_drop,
            'placed_economy_ids': sorted(pid for pid in placed_ids if pkg_lookup_all[pid]['Type'] != 'Priority'),
        }
        with open(BOOTSTRAP_LABELS_PATH, 'a') as f:
            f.write(json.dumps(new_scene) + '\n')

        is_best = cost < best_ever_cost
        if is_best:
            best_ever_cost = cost
            warm_start_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': model.state_dict(), 'arch': ARCH,
                'feat_mean': feat_mean, 'feat_std': feat_std, 'gmean': gmean, 'gstd': gstd,
                'avg_uld_volume': avg_uld_volume, 'avg_uld_weight': avg_uld_weight,
            }, BEST_EVER_CKPT)
            with open(BEST_EVER_META, 'w') as f:
                json.dump({'cost': cost, 'round': round_idx, 'strategy': strategy_name}, f)

        print(f'[round {round_idx:3d}] n_scenes={len(scenes):3d}  train_loss={train_loss:.4f} ({train_dt:.1f}s)  '
              f'real_cost={cost:,.0f} ({eval_dt:.1f}s)  econ_drop={econ_drop}  '
              f'{"** NEW BEST **" if is_best else ""}')

    print(f'\nDone. Best-ever real cost: {best_ever_cost:,.0f}  '
          f'(current external best: 30,475, competitor target: 29,203)')


if __name__ == '__main__':
    main()
