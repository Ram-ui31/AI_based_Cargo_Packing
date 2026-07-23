"""
pretrain_priority_consolidation.py -- dedicated supervised pretraining stage
for the Priority-ULD consolidation decision, to run BEFORE the RL+contrastive
fine-tune (train_ga_rl_ppo_contrastive.py).

Why this stage exists: in the RL+contrastive loop, the K-weighted
consolidation-imitation loss (_consolidation_imitation_loss) is one of six
competing loss terms (policy, entropy, capacity, hinge, spread, contrastive)
updated jointly every step, and each (instance, K) pool entry only gets
sampled ~2-5 times across a few hundred updates -- verified on
contrastive_v6/v7 that imitation_loss never converged (plateaued ~3-4.5
mean cross-entropy the entire run, no downward trend even after tripling
training length), and a same-instance sweep with the K>=500 heuristic
disabled still trailed it by ~10% mean cost after four different loss-design
attempts (v4-v7).

This stage isolates the SAME imitation objective with none of that
competition: pure forward pass (no rollout, no packer.pack() call -- the
teacher target comes directly from _consolidate_priority_by_capacity, not
from an actual placement) + cross-entropy + backward, so a full epoch over
200 instances x 5 K values costs ~1-2s in the packer/rollout loop's
equivalent unit of work costs. That lets this run 50+ epochs (thousands of
exposures per instance) in the time one RL fine-tune run took, teaching the
network the actual combinatorial "which few ULDs should hold ALL Priority"
target thoroughly before RL fine-tuning ever touches it.

Deliberately K-UNIFORM (not K-weighted) during this stage, unlike the RL
loop's imitation term: _consolidate_priority_by_capacity's target doesn't
depend on K at all, and the goal here is for the network to represent that
mapping accurately regardless of what K happens to be fed in as input --
the RL+contrastive fine-tune stage afterward is what teaches LOW-K deviation
from this pretrained baseline (the earlier-validated "more spread but less
delay" tradeoff), via its own K-weighted imitation term on top of an
already-competent starting point instead of a near-random one.

Usage:
    python scripts/pretrain_priority_consolidation.py \
        --data-root ~/Desktop/good_data \
        --old-il-weights checkpoints/il/transformer_imitation_ga.pt \
        --save-path checkpoints/il/transformer_imitation_priority_pretrained.pt \
        --n-epochs 60
"""
from __future__ import annotations
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.config import DEVICE, MAX_SAFE_PKGS, MAX_SAFE_ULDS
from src.rl.data_utils import build_tensors, needs_chunking
from src.rl.train_rl import _consolidate_priority_by_capacity
from train_ga_rl_graft import load_grafted_model

K_VALUES = [100, 500, 1000, 3000, 5000]


def _load_pool(data_dir, split, n_instances):
    meta = pd.read_csv(os.path.join(data_dir, split, 'metadata.csv'))
    if n_instances is not None:
        meta = meta.head(n_instances).reset_index(drop=True)
    pool = []
    for _, row in meta.iterrows():
        tag = row['instance']
        u_path = os.path.join(data_dir, split, f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, split, f'{tag}_packages.csv')
        if not (os.path.exists(u_path) and os.path.exists(p_path)):
            continue
        ulds_df, pkgs_df = pd.read_csv(u_path), pd.read_csv(p_path)
        if needs_chunking(pkgs_df, ulds_df, MAX_SAFE_PKGS, MAX_SAFE_ULDS):
            continue
        teacher_assignment, _, _ = _consolidate_priority_by_capacity(pkgs_df, ulds_df)
        if not teacher_assignment:
            continue
        pkg_id_list   = pkgs_df['Package_ID'].tolist()
        uld_id_to_idx = {uid: i for i, uid in enumerate(ulds_df['ULD_ID'].tolist())}
        pool.append(dict(tag=tag, ulds_df=ulds_df, pkgs_df=pkgs_df,
                          teacher_assignment=teacher_assignment,
                          pkg_id_list=pkg_id_list, uld_id_to_idx=uld_id_to_idx,
                          n_ulds=len(ulds_df), n_pkgs=len(pkgs_df)))
    return pool


def _step(model, entry, k, device, optimizer=None):
    ulds_df, pkgs_df = entry['ulds_df'], entry['pkgs_df']
    n_ulds, n_pkgs   = entry['n_ulds'], entry['n_pkgs']
    t = build_tensors(pkgs_df, ulds_df, device, k)
    logits = model.forward(
        t['uld_feats'], t['pkg_feats'], t['key_padding_mask'],
        torch.tensor([n_ulds], device=device), t['dim_mask'], t['priority_mask'],
        t['tightness'], t['k_feat'],
    ).squeeze(0)
    priority_mask_i = t['priority_mask'].squeeze(0)
    is_priority = ~priority_mask_i[:n_pkgs]
    prio_positions = is_priority.nonzero(as_tuple=True)[0]

    targets, valid_positions = [], []
    for pos in prio_positions.tolist():
        pid = entry['pkg_id_list'][pos]
        uid = entry['teacher_assignment'].get(pid)
        if uid is not None and uid in entry['uld_id_to_idx']:
            valid_positions.append(pos)
            targets.append(entry['uld_id_to_idx'][uid])
    if not targets:
        return None, None

    valid_positions_t = torch.tensor(valid_positions, device=device)
    targets_t = torch.tensor(targets, device=device, dtype=torch.long)
    selected_logits = logits[valid_positions_t][:, :n_ulds]
    loss = F.cross_entropy(selected_logits, targets_t)
    acc = (selected_logits.argmax(dim=-1) == targets_t).float().mean().item()

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--old-il-weights', required=True)
    ap.add_argument('--save-path', required=True)
    ap.add_argument('--n-epochs', type=int, default=60)
    ap.add_argument('--n-train-instances', type=int, default=200)
    ap.add_argument('--n-val-instances', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    old_il_weights = os.path.abspath(os.path.expanduser(args.old_il_weights))
    save_path = os.path.abspath(os.path.expanduser(args.save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f'Device: {DEVICE}')
    print('Loading train/val pools (teacher targets precomputed once)...')
    train_pool = _load_pool(data_root, 'synthetic_train', args.n_train_instances)
    val_pool   = _load_pool(data_root, 'synthetic_test', args.n_val_instances)
    print(f'Train pool: {len(train_pool)} instances x {len(K_VALUES)} K values '
          f'= {len(train_pool) * len(K_VALUES)} pairs/epoch')
    print(f'Val pool  : {len(val_pool)} instances x {len(K_VALUES)} K values')

    model = load_grafted_model(old_il_weights, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    rng = random.Random(args.seed)
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        train_losses, train_accs = [], []
        order = list(train_pool)
        rng.shuffle(order)
        for entry in order:
            for k in K_VALUES:
                loss, acc = _step(model, entry, k, DEVICE, optimizer=optimizer)
                if loss is not None:
                    train_losses.append(loss)
                    train_accs.append(acc)

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for entry in val_pool:
                for k in K_VALUES:
                    loss, acc = _step(model, entry, k, DEVICE, optimizer=None)
                    if loss is not None:
                        val_losses.append(loss)
                        val_accs.append(acc)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        mean_train_acc  = float(np.mean(train_accs))  if train_accs  else float('nan')
        mean_val_loss   = float(np.mean(val_losses))  if val_losses  else float('nan')
        mean_val_acc    = float(np.mean(val_accs))    if val_accs    else float('nan')

        marker = ''
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'val_acc': mean_val_acc, 'val_loss': mean_val_loss},
                       save_path)
            marker = '  [saved]'
        else:
            patience_counter += 1

        print(f'epoch {epoch:3d}  train_loss={mean_train_loss:.4f} train_acc={mean_train_acc:.4f}  '
              f'val_loss={mean_val_loss:.4f} val_acc={mean_val_acc:.4f} (best={best_val_acc:.4f}){marker}')

        if patience_counter >= args.patience:
            print(f'\n[Early stop] no val_acc improvement for {args.patience} epochs.')
            break

    print(f'\nDone. Best val_acc={best_val_acc:.4f}. Checkpoint saved to {save_path}')


if __name__ == '__main__':
    main()
