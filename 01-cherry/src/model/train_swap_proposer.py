"""
train_swap_proposer.py -- trains SwapProposer on real (move, delta) data
generated as a byproduct of ga_cargo_packing/scripts/beam_search_economy.py
(beam_moves_*.jsonl -- every TRIED swap, not just survivors, with its real
cost delta).

Uses a PAIRWISE RANKING loss, not plain delta regression. First version
regressed on raw delta with a Huber loss and got a misleadingly high
val_dir_acc (0.984) -- almost all logged swaps are cost-worsening (~82%
positive delta, ~13% zero, ~5% negative), so a model that just learns
"predict positive" scores well on that metric without learning what
actually makes one swap better than another. That's also not what the
guided beam search needs: it never asks "is this swap's delta negative in
absolute terms", it asks "which of these ~48 candidate swaps should I
spend a real evaluation on" -- a RANKING question. A pairwise loss over
moves that share the same parent (same beam member, same round -- so
their deltas are directly comparable) targets that question directly, and
sidesteps the class-imbalance problem: even "least bad of the bad" is a
useful, informative comparison for ranking, whereas it was a wasted
example for regression.

Only 'boundary_swap' and 'random_swap' moves are used (both are pairwise
swaps with pid_a/pid_b) -- 'relocate' moves have a different structure
(single package + insertion point, doesn't fit this pairwise architecture)
and are excluded. Moves also need a 'group_id' field (added after the
original logging code -- see beam_search_economy.py's move_log_entries)
to know which moves share a parent; older entries logged before that
field existed are silently skipped.

Usage:
    python src/train_swap_proposer.py
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swap_proposer import SwapProposer
from features import build_package_features, build_global_features, normalize_features

INPUT_PATH = os.path.expanduser('~/Downloads/input.csv')
CKPT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'swap_proposer.pt')


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


def load_grouped_moves(pattern):
    groups = defaultdict(list)
    n_skipped_no_group = 0
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                m = json.loads(line)
                if m['mode'] not in ('boundary_swap', 'random_swap'):
                    continue
                if 'group_id' not in m:
                    n_skipped_no_group += 1
                    continue
                groups[m['group_id']].append(m)
    return groups, n_skipped_no_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--moves-glob', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                          'ga_cargo_packing', 'results', 'beam_moves_*.jsonl'))
    p.add_argument('--n-epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--margin', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    groups, n_skipped = load_grouped_moves(args.moves_glob)
    if n_skipped:
        print(f'Skipped {n_skipped} moves logged before group_id was added (no way to pair them)')

    # Split by GROUP, not by pair, before building pairs -- two pairs drawn
    # from the same parent group (e.g. (mi,mj) and (mi,mk)) are not
    # independent, so a pair-level random split can put both in different
    # splits while sharing a move. Splitting by group first guarantees no
    # validation pair's underlying moves were ever seen (in any pairing)
    # during training -- a real train/val boundary, not just deduplicated
    # pairs.
    group_ids = list(groups.keys())
    g = torch.Generator().manual_seed(args.seed)
    perm_groups = torch.randperm(len(group_ids), generator=g).tolist()
    n_val_groups = max(1, len(group_ids) // 5)
    val_group_ids = {group_ids[i] for i in perm_groups[:n_val_groups]}
    train_group_ids = {group_ids[i] for i in perm_groups[n_val_groups:]}

    def build_pairs(gids):
        pairs = []
        for group_id in gids:
            moves = groups[group_id]
            for i in range(len(moves)):
                for j in range(i + 1, len(moves)):
                    mi, mj = moves[i], moves[j]
                    if mi['delta'] == mj['delta']:
                        continue
                    pairs.append((mi, mj) if mi['delta'] < mj['delta'] else (mj, mi))
                    # pairs[-1] = (better_move, worse_move)
        return pairs

    train_pairs = build_pairs(train_group_ids)
    val_pairs = build_pairs(val_group_ids)

    print(f'{len(groups)} parent groups ({len(train_group_ids)} train / {len(val_group_ids)} val), '
          f'{sum(len(v) for v in groups.values())} moves, '
          f'{len(train_pairs)} train pairs, {len(val_pairs)} val pairs (group-disjoint)')
    if len(train_pairs) < 40 or len(val_pairs) < 10:
        raise RuntimeError(f'Only {len(train_pairs)} train / {len(val_pairs)} val pairs found -- need '
                            f'more beam search rounds (with the new group_id logging) to generate '
                            f'enough training data.')

    k_value, ulds_df, pkgs_df = parse_input_csv(INPUT_PATH)
    economy_df = pkgs_df[pkgs_df['Type'] != 'Priority'].reset_index(drop=True)
    pid_to_i = {pid: i for i, pid in enumerate(economy_df['Package_ID'])}
    n_items = len(economy_df)

    avg_uld_volume = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).mean()
    avg_uld_weight = ulds_df['Weight_Limit'].mean()
    feats_np = build_package_features(economy_df, avg_uld_volume, avg_uld_weight)
    feats_np, feat_mean, feat_std = normalize_features(feats_np)
    global_feats_np = build_global_features(
        n_ulds=len(ulds_df),
        total_remaining_volume=(ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum(),
        total_remaining_weight=ulds_df['Weight_Limit'].sum(),
        k_value=k_value,
    )
    global_feats_np, gmean, gstd = normalize_features(global_feats_np.reshape(1, -1))

    def move_to_tensors(m):
        ia, ib = pid_to_i[m['pid_a']], pid_to_i[m['pid_b']]
        return feats_np[ia], feats_np[ib], m['pos_a'] / n_items, m['pos_b'] / n_items

    def build_tensors(pairs):
        win_a, win_b, win_pa, win_pb = [], [], [], []
        lose_a, lose_b, lose_pa, lose_pb = [], [], [], []
        n_skipped_unknown_pid = 0
        for better, worse in pairs:
            if better['pid_a'] not in pid_to_i or better['pid_b'] not in pid_to_i or \
               worse['pid_a'] not in pid_to_i or worse['pid_b'] not in pid_to_i:
                n_skipped_unknown_pid += 1
                continue
            fa, fb, pa, pb = move_to_tensors(better)
            win_a.append(fa); win_b.append(fb); win_pa.append(pa); win_pb.append(pb)
            fa, fb, pa, pb = move_to_tensors(worse)
            lose_a.append(fa); lose_b.append(fb); lose_pa.append(pa); lose_pb.append(pb)
        if n_skipped_unknown_pid:
            print(f'Skipped {n_skipped_unknown_pid} pairs referencing unknown package IDs')
        n = len(win_a)
        return {
            'win_a': torch.tensor(np.array(win_a), dtype=torch.float32),
            'win_b': torch.tensor(np.array(win_b), dtype=torch.float32),
            'win_pa': torch.tensor(win_pa, dtype=torch.float32).unsqueeze(1),
            'win_pb': torch.tensor(win_pb, dtype=torch.float32).unsqueeze(1),
            'lose_a': torch.tensor(np.array(lose_a), dtype=torch.float32),
            'lose_b': torch.tensor(np.array(lose_b), dtype=torch.float32),
            'lose_pa': torch.tensor(lose_pa, dtype=torch.float32).unsqueeze(1),
            'lose_pb': torch.tensor(lose_pb, dtype=torch.float32).unsqueeze(1),
            'global_feat': torch.tensor(global_feats_np, dtype=torch.float32).repeat(n, 1),
            # MarginRankingLoss(x1, x2, y) = max(0, -y*(x1-x2)+margin); y=-1 means
            # x1 should end up LOWER than x2 -- x1=pred(winner) should be lower
            # (more negative delta = more promising) than x2=pred(loser).
            'target': -torch.ones(n),
            'n': n,
        }

    train_t = build_tensors(train_pairs)
    val_t = build_tensors(val_pairs)
    print(f'{train_t["n"]} train pairs / {val_t["n"]} val pairs after filtering (group-disjoint)')

    model = SwapProposer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    history = []
    best_val_loss = float('inf')
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        pred_win = model(train_t['win_a'], train_t['win_b'], train_t['global_feat'],
                          train_t['win_pa'], train_t['win_pb'])
        pred_lose = model(train_t['lose_a'], train_t['lose_b'], train_t['global_feat'],
                           train_t['lose_pa'], train_t['lose_pb'])
        loss = loss_fn(pred_win, pred_lose, train_t['target'])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_rank_acc = (pred_win < pred_lose).float().mean().item()
            val_pred_win = model(val_t['win_a'], val_t['win_b'], val_t['global_feat'],
                                  val_t['win_pa'], val_t['win_pb'])
            val_pred_lose = model(val_t['lose_a'], val_t['lose_b'], val_t['global_feat'],
                                   val_t['lose_pa'], val_t['lose_pb'])
            val_loss = loss_fn(val_pred_win, val_pred_lose, val_t['target']).item()
            # Pairwise ranking accuracy: does the model score the TRUE winner lower?
            val_rank_acc = (val_pred_win < val_pred_lose).float().mean().item()

        history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss,
                         'train_rank_acc': train_rank_acc, 'val_rank_acc': val_rank_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'feat_mean': feat_mean, 'feat_std': feat_std, 'gmean': gmean, 'gstd': gstd,
                'avg_uld_volume': avg_uld_volume, 'avg_uld_weight': avg_uld_weight,
                'best_epoch': epoch, 'best_val_loss': best_val_loss, 'best_val_rank_acc': val_rank_acc,
                'n_train_pairs': train_t['n'], 'n_val_pairs': val_t['n'],
            }, CKPT_OUT)

        if epoch % 25 == 0 or epoch == args.n_epochs - 1:
            print(f'epoch {epoch:4d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  '
                  f'train_rank_acc={train_rank_acc:.3f}  val_rank_acc={val_rank_acc:.3f}  '
                  f'best_val={best_val_loss:.4f}')

    history_path = os.path.join(os.path.dirname(CKPT_OUT), 'swap_proposer_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f'\nSaved best checkpoint to {CKPT_OUT}')
    print(f'Saved full per-epoch training history to {history_path}')


if __name__ == '__main__':
    main()
