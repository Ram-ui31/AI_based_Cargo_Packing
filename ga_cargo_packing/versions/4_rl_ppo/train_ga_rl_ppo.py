"""
train_ga_rl_ppo.py -- PPO + per-K EMA baseline training, ported from
cargoism/git/model_b(c)/training/train_assignment.py's mechanics (see
src/rl/train_rl_ppo.py for the full training loop).

This is a SEPARATE entry point from scripts/train_ga_rl_graft.py (the
existing, working REINFORCE pipeline that produced checkpoints/rl_modelb_balanced/,
which currently beats the original K-blind model). Running this script does
NOT touch that checkpoint, its training script, or any of src/rl/{model,
data_utils,reward,train_rl}.py -- it only imports from them (plus reuses
load_grafted_model/build_k_map from train_ga_rl_graft.py directly rather
than duplicating them). If this experiment doesn't pan out, the working
REINFORCE pipeline is completely unaffected.

Usage:
    python scripts/train_ga_rl_ppo.py \
        --data-root /Users/ramupadhyay/Desktop/good_data \
        --old-il-weights ../checkpoints/il/transformer_imitation_ga.pt \
        --save-dir ../checkpoints/rl_ppo
"""
from __future__ import annotations
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rl.config import (
    DEVICE, RL_LR, RL_ENTROPY_COEF, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    RL_HINGE_COEF, RL_SPREAD_COEF,
)
from src.rl.train_rl_ppo import train_rl_ppo
from train_ga_rl_graft import load_grafted_model, build_k_map  # reused unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--old-il-weights', required=True,
                     help='the ORIGINAL K-blind IL checkpoint (checkpoints/il/transformer_imitation_ga.pt)')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--n-updates', type=int, default=200)
    ap.add_argument('--episodes-per-update', type=int, default=8)
    ap.add_argument('--ppo-epochs', type=int, default=4)
    ap.add_argument('--clip-eps', type=float, default=0.2)
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--val-instances', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--max-instances', type=int, default=None,
                     help='cap train instances (smoke-testing only)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--priority-drop-penalty', type=float, default=2000.0)
    ap.add_argument('--hinge-coef', type=float, default=RL_HINGE_COEF)
    ap.add_argument('--spread-coef', type=float, default=RL_SPREAD_COEF)
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

    rl_save = os.path.join(save_dir, 'transformer_rl_ppo.pt')
    rl_log  = os.path.join(save_dir, 'rl_ppo_training_log.csv')

    print(f'Device        : {DEVICE}')
    print(f'Data root     : {data_root}')
    print(f'Old IL weights: {old_il_weights}')
    print(f'Save dir      : {save_dir}')

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
    print(f'Assigned K values to {len(k_values_map_dict)} instances.')

    from src.rl.rl_packer_adapter import RLPackerAdapter
    packer = RLPackerAdapter()
    print(f'Packer      : rl_packer ({packer.weights_path})')

    history = train_rl_ppo(
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
        lambda_weight_penalty=RL_LAMBDA_WEIGHT_PENALTY,
        lambda_volume_penalty=RL_LAMBDA_VOLUME_PENALTY,
        val_every=args.val_every,
        val_instances=args.val_instances,
        patience=args.patience,
        save_path=rl_save,
        log_path=rl_log,
        device=DEVICE,
        temperature=RL_TEMPERATURE,
        max_instances=args.max_instances,
        packer=packer,
        k_values_map_dict=k_values_map_dict,
        priority_drop_penalty=args.priority_drop_penalty,
        initial_best_val_cost=initial_best_val_cost,
        seed=args.seed,
    )
    print(history.tail())


if __name__ == '__main__':
    main()
