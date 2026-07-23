"""
train_centrifuge_proposer.py -- trains CentrifugeEvictProposer on the
labeled data from generate_centrifuge_data.py.

Split is at INSTANCE (group) level, not example level -- learned this the
hard way with SwapProposer earlier in this project: splitting at the pair/
example level let examples from the same underlying context leak across
train/val, inflating reported accuracy (89% -> honest ~65-77% once fixed).
All examples from one instance share highly correlated context (same
package pool, same ULD set), so instance-level split is the only split
that gives an honest read on generalization to genuinely unseen problems.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from centrifuge_proposer import CentrifugeEvictProposer, pkg_to_feat_vec, PKG_FEATURE_DIM


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class CentrifugeDataset(Dataset):
    def __init__(self, records, pkg_mean=None, pkg_std=None, uld_mean=None, uld_std=None,
                 glob_mean=None, glob_std=None):
        self.records = records
        all_pkg_feats = []
        for r in records:
            all_pkg_feats.append(pkg_to_feat_vec(r['evict_pkg']))
            for c in r['container_pkgs']:
                all_pkg_feats.append(pkg_to_feat_vec(c))
            for c in r['unplaced_pool']:
                all_pkg_feats.append(pkg_to_feat_vec(c))
        all_pkg_feats = np.array(all_pkg_feats, dtype=np.float32) if all_pkg_feats else np.zeros((1, PKG_FEATURE_DIM), dtype=np.float32)
        self.pkg_mean = pkg_mean if pkg_mean is not None else all_pkg_feats.mean(axis=0)
        self.pkg_std = pkg_std if pkg_std is not None else all_pkg_feats.std(axis=0) + 1e-6

        uld_feats = np.array([[r['uld']['length'], r['uld']['width'], r['uld']['height'], r['uld']['weight_limit']]
                               for r in records], dtype=np.float32)
        self.uld_mean = uld_mean if uld_mean is not None else uld_feats.mean(axis=0)
        self.uld_std = uld_std if uld_std is not None else uld_feats.std(axis=0) + 1e-6

        glob_feats = np.array([[r['k_value'], len(r['container_pkgs']), len(r['unplaced_pool'])]
                                for r in records], dtype=np.float32)
        self.glob_mean = glob_mean if glob_mean is not None else glob_feats.mean(axis=0)
        self.glob_std = glob_std if glob_std is not None else glob_feats.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        container_feats = np.array([pkg_to_feat_vec(c) for c in r['container_pkgs']], dtype=np.float32) \
            if r['container_pkgs'] else np.zeros((1, PKG_FEATURE_DIM), dtype=np.float32)
        pool_feats = np.array([pkg_to_feat_vec(c) for c in r['unplaced_pool']], dtype=np.float32) \
            if r['unplaced_pool'] else np.zeros((1, PKG_FEATURE_DIM), dtype=np.float32)
        evict_feat = np.array(pkg_to_feat_vec(r['evict_pkg']), dtype=np.float32)
        uld_feat = np.array([r['uld']['length'], r['uld']['width'], r['uld']['height'], r['uld']['weight_limit']], dtype=np.float32)
        glob_feat = np.array([r['k_value'], len(r['container_pkgs']), len(r['unplaced_pool'])], dtype=np.float32)

        container_feats = (container_feats - self.pkg_mean) / self.pkg_std
        pool_feats = (pool_feats - self.pkg_mean) / self.pkg_std
        evict_feat = (evict_feat - self.pkg_mean) / self.pkg_std
        uld_feat = (uld_feat - self.uld_mean) / self.uld_std
        glob_feat = (glob_feat - self.glob_mean) / self.glob_std

        return {
            'container_feats': torch.from_numpy(container_feats),
            'pool_feats': torch.from_numpy(pool_feats),
            'evict_feat': torch.from_numpy(evict_feat),
            'uld_feat': torch.from_numpy(uld_feat),
            'glob_feat': torch.from_numpy(glob_feat),
            'net_gain': torch.tensor(r['net_gain'], dtype=torch.float32),
            'group_key': r['instance'] + '|' + '_'.join(f'{v:.1f}' for v in r['uld'].values()),
        }


def collate(batch):
    def pad_stack(key):
        tensors = [b[key] for b in batch]
        max_n = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_n, PKG_FEATURE_DIM)
        mask = torch.ones(len(tensors), max_n, dtype=torch.bool)  # True = PAD
        for i, t in enumerate(tensors):
            n = t.shape[0]
            padded[i, :n] = t
            mask[i, :n] = False
        return padded, mask

    container_feats, container_mask = pad_stack('container_feats')
    pool_feats, pool_mask = pad_stack('pool_feats')
    evict_feat = torch.stack([b['evict_feat'] for b in batch])
    uld_feat = torch.stack([b['uld_feat'] for b in batch])
    glob_feat = torch.stack([b['glob_feat'] for b in batch])
    net_gain = torch.stack([b['net_gain'] for b in batch])
    group_keys = [b['group_key'] for b in batch]
    return container_feats, container_mask, pool_feats, pool_mask, evict_feat, uld_feat, glob_feat, net_gain, group_keys


def precision_at_1(preds, labels, group_keys):
    """For each group (container), does the model's top-1 predicted
    candidate have the actual highest true net_gain in that group?
    The metric that matters for deployment: rank candidates, verify only
    the top few for real."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, gk in enumerate(group_keys):
        groups[gk].append(i)
    hits, total = 0, 0
    for gk, idxs in groups.items():
        if len(idxs) < 2:
            continue
        pred_best = max(idxs, key=lambda i: preds[i])
        true_best = max(idxs, key=lambda i: labels[i])
        total += 1
        if pred_best == true_best:
            hits += 1
    return hits / max(total, 1), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/centrifuge_train.jsonl')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='checkpoints/centrifuge_proposer.pt')
    p.add_argument('--history-out', type=str, default='data/centrifuge_proposer_history.json')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(args.data)
    instances = sorted(set(r['instance'] for r in records))
    random.Random(args.seed).shuffle(instances)
    n_val_inst = max(1, int(len(instances) * args.val_frac))
    val_instances = set(instances[:n_val_inst])
    train_records = [r for r in records if r['instance'] not in val_instances]
    val_records = [r for r in records if r['instance'] in val_instances]
    print(f'{len(instances)} instances total, {len(val_instances)} held out for val (instance-level split).')
    print(f'Train examples: {len(train_records)}  Val examples: {len(val_records)}')
    print(f'Train positive rate: {np.mean([r["net_gain"]>0 for r in train_records]):.3f}  '
          f'Val positive rate: {np.mean([r["net_gain"]>0 for r in val_records]):.3f}')

    train_ds = CentrifugeDataset(train_records)
    val_ds = CentrifugeDataset(val_records, pkg_mean=train_ds.pkg_mean, pkg_std=train_ds.pkg_std,
                                uld_mean=train_ds.uld_mean, uld_std=train_ds.uld_std,
                                glob_mean=train_ds.glob_mean, glob_std=train_ds.glob_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = CentrifugeEvictProposer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=50.0)  # robust to net_gain outliers, matches typical delay_cost scale

    history = []
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            cf, cm, pf, pm, ef, uf, gf, y, gk = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            opt.zero_grad()
            pred = model(cf, cm, pf, pm, ef, uf, gf)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_preds, all_labels, all_gks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                cf, cm, pf, pm, ef, uf, gf, y, gk = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                pred = model(cf, cm, pf, pm, ef, uf, gf)
                loss = loss_fn(pred, y)
                val_losses.append(loss.item())
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                all_gks.extend(gk)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_win_acc = np.mean((all_preds > 0) == (all_labels > 0))
        p_at_1, n_groups = precision_at_1(all_preds, all_labels, all_gks)
        # correlation as an additional ranking-quality signal
        corr = np.corrcoef(all_preds, all_labels)[0, 1] if len(all_preds) > 1 else 0.0

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f'Epoch {epoch:3d}: train_loss={train_loss:.1f} val_loss={val_loss:.1f} '
              f'val_win_acc={val_win_acc:.3f} val_p@1={p_at_1:.3f} (n_groups={n_groups}) val_corr={corr:.3f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                         'val_win_acc': float(val_win_acc), 'val_p_at_1': float(p_at_1),
                         'val_corr': float(corr)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'pkg_mean': train_ds.pkg_mean, 'pkg_std': train_ds.pkg_std,
                'uld_mean': train_ds.uld_mean, 'uld_std': train_ds.uld_std,
                'glob_mean': train_ds.glob_mean, 'glob_std': train_ds.glob_std,
                'epoch': epoch, 'val_loss': val_loss,
            }, args.out)

    with open(args.history_out, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nBest val_loss={best_val_loss:.1f}. Saved checkpoint to {args.out}, history to {args.history_out}')


if __name__ == '__main__':
    main()
