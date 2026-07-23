"""
train_ga_rl.py — RL fine-tuning driver, starting from the IL checkpoint
trained on GA-generated labels (see scripts/train_ga_il.py).
Mirrors cargoism/git/rl_over_il_h1h2/scripts/train_h1h2_rl.py, with two
differences:
  1. Defaults to RLPackerAdapter (rl_packer's learned placement policy) as
     the packer, per the user's requirement, rather than EPIPacker being the
     default and rl_packer being opt-in. --use-py3dbp / --use-epi remain
     available for comparison runs.
  2. K-value assignment prefers good_data's own metadata_with_K.csv (already
     built to the 200x5 train / 20x5 test spec) over re-deriving a shuffled
     split, falling back to the shuffle-based build_k_map only if that file
     is missing.

Auxiliary losses (spread / capacity-violation / priority-drop) are already
implemented in the copied src/{reward,train_rl}.py -- see that module's
compute_packing_cost, rl_capacity_violation_penalty, and train_rl's
priority_drop_penalty param.

Usage:
    python scripts/train_ga_rl.py \
        --data-root /Users/ramupadhyay/Desktop/good_data \
        --il-weights ../checkpoints/il/transformer_imitation_ga.pt \
        --save-dir ../checkpoints/rl
"""
from __future__ import annotations
import argparse
import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.rl.config import (
    DEVICE,
    RL_EPOCHS, RL_LR, RL_ENTROPY_COEF, RL_KL_COEF,
    RL_EVAL_EVERY, RL_PATIENCE, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
)
from src.rl.model import TransformerClusterer
from src.rl.packer import EPIPacker
from src.rl.train_rl import train_rl

K_VALUES = [100, 500, 1000, 3000, 5000]


def build_k_map(data_root, seed):
    """
    Prefer good_data/synthetic_train/metadata_with_K.csv, which already
    assigns K per the 200-per-K (out of 1000 train instances) spec. Falls
    back to a fresh shuffled split (same logic as train_h1h2_rl.py's
    build_k_map) only if that file doesn't exist.

    train_rl()'s train_cache/test_cache are both keyed by the bare
    'instance_NNN' tag, and synthetic_test's tags are a name-reused subset
    of synthetic_train's -- NOT distinct instances. A single flat
    k_values_map_dict is looked up with that same bare tag in both the train
    and eval loops, so K must come from one source (synthetic_train) that
    synthetic_test's same-named tags inherit, not be assigned independently
    per split.
    """
    meta_with_k_path = os.path.join(data_root, 'synthetic_train', 'metadata_with_K.csv')
    if os.path.exists(meta_with_k_path):
        meta = pd.read_csv(meta_with_k_path)
        return dict(zip(meta['instance'], meta['K']))

    rng = random.Random(seed)
    meta = pd.read_csv(os.path.join(data_root, 'synthetic_train', 'metadata.csv'))
    instances = meta['instance'].tolist()
    rng.shuffle(instances)
    per_k = len(instances) // len(K_VALUES)
    k_map = {}
    for i, k_val in enumerate(K_VALUES):
        for tag in instances[i * per_k:(i + 1) * per_k]:
            k_map[tag] = k_val
    return k_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--il-weights', required=True)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--n-epochs', type=int, default=RL_EPOCHS)
    ap.add_argument('--use-py3dbp', action='store_true',
                     help='use pd3Packer (py3dbp) instead of rl_packer')
    ap.add_argument('--use-epi', action='store_true',
                     help='use the built-in EPIPacker instead of rl_packer')
    ap.add_argument('--max-instances', type=int, default=None,
                     help='cap train instances (smoke-testing only)')
    ap.add_argument('--n-rl-samples', type=int, default=4,
                     help='stochastic rollouts averaged per instance per epoch before computing '
                          'advantage -- reduces single-sample REINFORCE variance at the cost of '
                          'roughly n_rl_samples x more compute per epoch')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume-from', default=None,
                     help='RL checkpoint to continue training the policy from '
                          '(il-weights is still loaded separately as the frozen KL/baseline reference)')
    ap.add_argument('--priority-drop-penalty', type=float, default=2000.0,
                     help='training-signal-only penalty per unplaced Priority package '
                          '(Delay_Cost=0 for Priority, so the official cost formula gives '
                          'zero incentive to avoid dropping them without this)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root  = os.path.abspath(os.path.expanduser(args.data_root))
    il_weights = os.path.abspath(os.path.expanduser(args.il_weights))
    save_dir   = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    assert os.path.exists(il_weights), f'IL checkpoint not found: {il_weights}'

    rl_save           = os.path.join(save_dir, 'transformer_rl_ga.pt')
    rl_baseline_cache = os.path.join(save_dir, 'il_baseline_cache_ga.pkl')
    rl_log            = os.path.join(save_dir, 'rl_training_log_ga.csv')

    print(f'Device      : {DEVICE}')
    print(f'Data root   : {data_root}')
    print(f'IL weights  : {il_weights}')
    print(f'Save dir    : {save_dir}')

    model = TransformerClusterer().to(DEVICE)
    ckpt  = torch.load(il_weights, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    il_epoch    = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    il_val_loss = ckpt.get('val_loss', float('nan')) if isinstance(ckpt, dict) else float('nan')
    print(f'IL checkpoint loaded: epoch={il_epoch}, val_loss={il_val_loss}')

    il_model = copy.deepcopy(model).to(DEVICE)
    il_model.eval()
    for p in il_model.parameters():
        p.requires_grad_(False)

    initial_best_val_cost = float('inf')
    if args.resume_from:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_from))
        assert os.path.exists(resume_path), f'resume checkpoint not found: {resume_path}'
        rckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        rstate = rckpt['model_state_dict'] if isinstance(rckpt, dict) and 'model_state_dict' in rckpt else rckpt
        model.load_state_dict(rstate, strict=True)
        # Carry over the resumed checkpoint's own best cost so this run only
        # overwrites save_path with an ACTUALLY better result -- train_rl()
        # otherwise starts best_val_cost at inf and would happily save over
        # a good checkpoint with a worse one on its very first eval.
        if isinstance(rckpt, dict):
            initial_best_val_cost = rckpt.get('val_rl_cost_penalized', rckpt.get('val_rl_cost', float('inf')))
        print(f'Resumed policy weights from {resume_path} '
              f'(epoch={rckpt.get("epoch","?")}, val_rl_cost={rckpt.get("val_rl_cost","?")}, '
              f'carrying over best_val_cost={initial_best_val_cost})')

    k_values_map_dict = build_k_map(data_root, args.seed)
    print(f'Assigned K values to {len(k_values_map_dict)} instances.')

    if args.use_epi:
        packer = EPIPacker()
        print('Packer      : EPIPacker')
    elif args.use_py3dbp:
        from src.rl.packer import pd3Packer
        packer = pd3Packer()
        print('Packer      : pd3Packer (py3dbp)')
    else:
        from src.rl.rl_packer_adapter import RLPackerAdapter
        packer = RLPackerAdapter()
        print(f'Packer      : rl_packer ({packer.weights_path})')

    history = train_rl(
        model                  = model,
        il_model               = il_model,
        data_dir               = data_root,
        n_epochs               = args.n_epochs,
        lr                     = RL_LR,
        entropy_coef           = RL_ENTROPY_COEF,
        kl_coef                = RL_KL_COEF,
        eval_every             = RL_EVAL_EVERY,
        patience               = RL_PATIENCE,
        save_path              = rl_save,
        log_path               = rl_log,
        il_baseline_cache_path = rl_baseline_cache,
        device                 = DEVICE,
        temperature            = RL_TEMPERATURE,
        max_instances          = args.max_instances,
        n_rl_samples           = args.n_rl_samples,
        packer                 = packer,
        lambda_weight_penalty  = RL_LAMBDA_WEIGHT_PENALTY,
        lambda_volume_penalty  = RL_LAMBDA_VOLUME_PENALTY,
        k_values_map_dict      = k_values_map_dict,
        priority_drop_penalty  = args.priority_drop_penalty,
        initial_best_val_cost  = initial_best_val_cost,
    )
    print(history.tail())


if __name__ == '__main__':
    main()
