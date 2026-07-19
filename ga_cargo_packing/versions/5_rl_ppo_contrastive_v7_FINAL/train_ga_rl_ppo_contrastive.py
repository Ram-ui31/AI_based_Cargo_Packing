"""
train_ga_rl_ppo_contrastive.py -- PPO training with a contrastive cross-K
loss term (src/rl/train_rl_ppo_contrastive.py) on top of the same 200-
instances x 5-K-values training pool used by train_ga_rl_ppo_multik.py.

Supersedes train_ga_rl_ppo_multik.py's approach: multi-K training data alone
removed the correlational confound but did not produce causal K-sensitivity
(same-instance K-sweep came back perfectly flat). This adds the piece that
was missing -- an explicit loss term contrasting the SAME instance's
predicted spread at two different K values, directly rewarding the network
for using K as a causal control variable.

Does NOT modify train_rl_ppo.py, train_rl_ppo_contrastive.py's own
dependencies, or any other existing file -- monkey-patches
train_rl_ppo_contrastive's `_load_train_pool` (imported from train_rl_ppo,
same function referenced there) for this process only, exactly like
train_ga_rl_ppo_multik.py did.

Usage:
    python scripts/train_ga_rl_ppo_contrastive.py \
        --data-root ~/Desktop/good_data \
        --old-il-weights checkpoints/il/transformer_imitation_ga.pt \
        --save-dir checkpoints/rl_ppo_contrastive \
        --n-instances 200
"""
from __future__ import annotations
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.config import (
    DEVICE, RL_LR, RL_ENTROPY_COEF, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    RL_HINGE_COEF, RL_SPREAD_COEF, MAX_SAFE_PKGS, MAX_SAFE_ULDS,
)
from src.rl.data_utils import needs_chunking
import src.rl.train_rl_ppo as ppo_module
from src.rl.train_rl_ppo_contrastive import train_rl_ppo_contrastive
from train_ga_rl_graft import load_grafted_model, build_k_map  # reused unchanged

K_VALUES = [100, 500, 1000, 3000, 5000]


def _load_train_pool_multik(data_dir, k_values_map_dict, max_instances, max_safe_pkgs, max_safe_ulds):
    """Same pool construction as train_ga_rl_ppo_multik.py: n_instances
    underlying instances, each paired with all 5 K values."""
    n_instances = max_instances or 200
    train_meta = pd.read_csv(os.path.join(data_dir, 'synthetic_train', 'metadata.csv'))

    pool = {}
    n_used = 0
    n_skipped = 0
    for _, row in train_meta.iterrows():
        if n_used >= n_instances:
            break
        tag = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv')
        if not (os.path.exists(u_path) and os.path.exists(p_path)):
            continue
        ulds_df, pkgs_df = pd.read_csv(u_path), pd.read_csv(p_path)
        if needs_chunking(pkgs_df, ulds_df, max_safe_pkgs, max_safe_ulds):
            n_skipped += 1
            continue
        for k in K_VALUES:
            pool[f'{tag}__K{k}'] = (ulds_df, pkgs_df, k)
        n_used += 1

    print(f'Train pool (multi-K): {n_used} underlying instances x {len(K_VALUES)} K values '
          f'= {len(pool)} pool entries ({n_skipped} oversized instances skipped).')
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--old-il-weights', required=True)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--n-instances', type=int, default=200)
    ap.add_argument('--n-updates', type=int, default=200)
    ap.add_argument('--episodes-per-update', type=int, default=8)
    ap.add_argument('--ppo-epochs', type=int, default=4)
    ap.add_argument('--clip-eps', type=float, default=0.2)
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--val-instances', type=int, default=40)
    ap.add_argument('--patience', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--priority-drop-penalty', type=float, default=2000.0)
    ap.add_argument('--hinge-coef', type=float, default=RL_HINGE_COEF)
    ap.add_argument('--spread-coef', type=float, default=RL_SPREAD_COEF)
    ap.add_argument('--contrastive-coef', type=float, default=8.0)
    ap.add_argument('--contrastive-margin-slope', type=float, default=3.0)
    ap.add_argument('--imitation-coef', type=float, default=0.0)
    ap.add_argument('--resume-from', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    old_il_weights = os.path.abspath(os.path.expanduser(args.old_il_weights))
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    assert os.path.exists(old_il_weights), f'old IL checkpoint not found: {old_il_weights}'

    rl_save = os.path.join(save_dir, 'transformer_rl_ppo_contrastive.pt')
    rl_log  = os.path.join(save_dir, 'rl_ppo_contrastive_training_log.csv')

    print(f'Device        : {DEVICE}')
    print(f'Data root     : {data_root}')
    print(f'Old IL weights: {old_il_weights}')
    print(f'Save dir      : {save_dir}')
    print(f'Training pool : {args.n_instances} instances x {K_VALUES} K values')
    print(f'Contrastive   : coef={args.contrastive_coef}  margin_slope={args.contrastive_margin_slope}  '
          f'imitation_coef={args.imitation_coef}')

    model = load_grafted_model(old_il_weights, DEVICE)

    initial_best_val_cost = float('inf')
    if args.resume_from:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_from))
        assert os.path.exists(resume_path), f'resume checkpoint not found: {resume_path}'
        rckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        rstate = rckpt['model_state_dict'] if isinstance(rckpt, dict) and 'model_state_dict' in rckpt else rckpt
        model.load_state_dict(rstate, strict=True)
        if isinstance(rckpt, dict):
            initial_best_val_cost = rckpt.get('val_rl_cost_penalized', rckpt.get('val_rl_cost', float('inf')))
        print(f'Resumed policy weights from {resume_path} (carrying over best_val_cost={initial_best_val_cost})')

    k_values_map_dict = build_k_map(data_root, args.seed)
    print(f'Assigned K values to {len(k_values_map_dict)} instances (validation only).')

    from src.rl.rl_packer_adapter import RLPackerAdapter
    packer = RLPackerAdapter()
    print(f'Packer      : rl_packer ({packer.weights_path})')

    ppo_module._load_train_pool = _load_train_pool_multik

    history = train_rl_ppo_contrastive(
        model=model,
        data_dir=data_root,
        n_updates=args.n_updates,
        episodes_per_update=args.episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        lr=RL_LR,
        entropy_coef=RL_ENTROPY_COEF,
        hinge_coef=args.hinge_coef,
        spread_coef=args.spread_coef,
        contrastive_coef=args.contrastive_coef,
        contrastive_margin_slope=args.contrastive_margin_slope,
        imitation_coef=args.imitation_coef,
        lambda_weight_penalty=RL_LAMBDA_WEIGHT_PENALTY,
        lambda_volume_penalty=RL_LAMBDA_VOLUME_PENALTY,
        val_every=args.val_every,
        val_instances=args.val_instances,
        patience=args.patience,
        save_path=rl_save,
        log_path=rl_log,
        device=DEVICE,
        temperature=RL_TEMPERATURE,
        max_instances=args.n_instances,
        packer=packer,
        max_safe_pkgs=MAX_SAFE_PKGS,
        max_safe_ulds=MAX_SAFE_ULDS,
        k_values_map_dict=k_values_map_dict,
        priority_drop_penalty=args.priority_drop_penalty,
        initial_best_val_cost=initial_best_val_cost,
        seed=args.seed,
    )

    print('\nTraining complete.')
    print(f'Best checkpoint saved to: {rl_save}')
    print(f'Training log saved to   : {rl_log}')


if __name__ == '__main__':
    main()
