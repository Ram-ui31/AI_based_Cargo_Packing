"""
train_ga_rl_graft.py -- K-aware RL fine-tuning, warm-started from the OLD
(K-blind) IL checkpoint instead of retraining IL from scratch.

Why this exists: retraining IL with K as an input (scripts/train_ga_il.py ->
checkpoints/il_k_aware/) produced a meaningfully worse IL model (val_loss
0.797 @ epoch 146, early-stopped) than the original K-blind IL run
(val_loss 0.760 @ epoch 233, checkpoints/il/transformer_imitation_ga.pt).
Two reasons: (1) K carries no signal for the IL objective -- GA labels are
generated without any K term, so k_proj's gradients are pure noise during
IL and can only hurt convergence, never help; (2) adding k_proj as a new
parameter shifts every subsequent draw in the seeded weight init, so the
whole transformer trunk started from a different (and here, worse) random
initialization than the original run.

Fix: graft a k_proj layer onto the ALREADY-CONVERGED old IL checkpoint
(strict=False load -- every other weight matches shape/name exactly) and
ZERO-INITIALIZE k_proj.weight, so the grafted model is mathematically
identical to the old K-blind model at t=0 (k_emb = k_proj(k_feat) = 0 for
any K). RL then fine-tunes from that strong, fully-converged trunk, and
k_proj only moves away from zero to the extent K actually helps reduce the
RL reward -- which is where K's signal actually lives (compute_packing_cost
includes K*n_priority_ulds; the GA/IL label objective never did).

Usage:
    python scripts/train_ga_rl_graft.py \
        --data-root /Users/ramupadhyay/Desktop/good_data \
        --old-il-weights ../checkpoints/il/transformer_imitation_ga.pt \
        --save-dir ../checkpoints/rl_graft_k_aware
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
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.rl.config import (
    DEVICE,
    RL_EPOCHS, RL_LR, RL_ENTROPY_COEF, RL_KL_COEF,
    RL_EVAL_EVERY, RL_PATIENCE, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    RL_HINGE_COEF, RL_SPREAD_COEF,
)
from src.rl.model import TransformerClusterer
from src.rl.train_rl import train_rl

K_VALUES = [100, 500, 1000, 3000, 5000]


def build_k_map(data_root, seed):
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


def load_grafted_model(old_il_weights_path, device):
    """
    Old (K-blind) checkpoint's state_dict is missing two things the current
    (dual K-injection) TransformerClusterer has: k_proj.weight (entirely new
    key) and an extra input column on output_head.0.weight (shape grew from
    (d_model//2, d_model) to (d_model//2, d_model+1) for the direct K-concat
    path -- see model.py's forward()). A plain strict=False load handles a
    missing KEY fine but raises on a shape MISMATCH for a key present in
    both, so output_head.0.weight needs manual surgery: transplant the old
    weights into the first d_model columns and zero the new column, so the
    K-concat path contributes exactly 0 at t=0 -- same "start behaviorally
    identical to the old model" philosophy as k_proj's zero-init below,
    generalized to both K-injection paths.
    """
    model = TransformerClusterer().to(device)
    ckpt = torch.load(old_il_weights_path, map_location='cpu', weights_only=False)
    state = dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)

    old_head_w = state.pop('output_head.0.weight')             # (d_model//2, d_model)
    new_head_w = model.output_head[0].weight.data.clone()      # (d_model//2, d_model+1)
    new_head_w[:, :old_head_w.shape[1]] = old_head_w
    new_head_w[:, old_head_w.shape[1]:] = 0.0
    state['output_head.0.weight'] = new_head_w

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert list(missing) == ['k_proj.weight'], (
        f'expected only k_proj.weight to be missing from the old checkpoint, got: {missing}'
    )
    assert not unexpected, f'unexpected keys in old checkpoint not present in current model: {unexpected}'

    with torch.no_grad():
        nn.init.zeros_(model.k_proj.weight)

    il_epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    il_val_loss = ckpt.get('val_loss', float('nan')) if isinstance(ckpt, dict) else float('nan')
    print(f'Grafted k_proj + output_head K-column (zero-init) onto old IL checkpoint: '
          f'epoch={il_epoch}, val_loss={il_val_loss}')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--old-il-weights', required=True,
                     help='the ORIGINAL K-blind IL checkpoint (checkpoints/il/transformer_imitation_ga.pt), '
                          'NOT the K-aware retrain -- this is the whole point of grafting')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--n-epochs', type=int, default=RL_EPOCHS)
    ap.add_argument('--max-instances', type=int, default=None,
                     help='cap train instances (smoke-testing only)')
    ap.add_argument('--n-rl-samples', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume-from', default=None,
                     help='RL checkpoint to continue training the policy from '
                          '(old-il-weights is still loaded separately as the frozen KL/baseline reference)')
    ap.add_argument('--priority-drop-penalty', type=float, default=2000.0)
    ap.add_argument('--hinge-coef', type=float, default=RL_HINGE_COEF,
                     help='weight on the feasibility hinge loss (ported from model_b) -- '
                          'penalizes preferring NONE over a dimensionally-feasible ULD for Economy packages')
    ap.add_argument('--spread-coef', type=float, default=RL_SPREAD_COEF,
                     help='weight on the K-scaled differentiable soft-spread loss (ported from model_b)')
    ap.add_argument('--instances-per-epoch', type=int, default=None,
                     help='sample this many train instances per epoch (stratified across K), instead of '
                          'all of them -- trades broader-per-epoch coverage for several times more '
                          'epochs (gradient updates) per hour of wall-clock time. None = use all.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    old_il_weights = os.path.abspath(os.path.expanduser(args.old_il_weights))
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    assert os.path.exists(old_il_weights), f'old IL checkpoint not found: {old_il_weights}'

    rl_save = os.path.join(save_dir, 'transformer_rl_ga.pt')
    rl_baseline_cache = os.path.join(save_dir, 'il_baseline_cache_ga.pkl')
    rl_log = os.path.join(save_dir, 'rl_training_log_ga.csv')

    print(f'Device        : {DEVICE}')
    print(f'Data root     : {data_root}')
    print(f'Old IL weights: {old_il_weights}')
    print(f'Save dir      : {save_dir}')

    model = load_grafted_model(old_il_weights, DEVICE)

    # Frozen KL/baseline reference is the SAME grafted (zero-init k_proj)
    # starting point -- at t=0 it's behaviorally identical to the old
    # K-blind model, so the IL baseline cost used for the REINFORCE
    # advantage is exactly what the old model would have produced.
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
        if isinstance(rckpt, dict):
            initial_best_val_cost = rckpt.get('val_rl_cost_penalized', rckpt.get('val_rl_cost', float('inf')))
        print(f'Resumed policy weights from {resume_path} '
              f'(epoch={rckpt.get("epoch","?")}, val_rl_cost={rckpt.get("val_rl_cost","?")}, '
              f'carrying over best_val_cost={initial_best_val_cost})')

    k_values_map_dict = build_k_map(data_root, args.seed)
    print(f'Assigned K values to {len(k_values_map_dict)} instances.')

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
        max_instances           = args.max_instances,
        n_rl_samples           = args.n_rl_samples,
        packer                 = packer,
        lambda_weight_penalty  = RL_LAMBDA_WEIGHT_PENALTY,
        lambda_volume_penalty  = RL_LAMBDA_VOLUME_PENALTY,
        hinge_coef             = args.hinge_coef,
        spread_coef            = args.spread_coef,
        k_values_map_dict      = k_values_map_dict,
        priority_drop_penalty  = args.priority_drop_penalty,
        initial_best_val_cost  = initial_best_val_cost,
        instances_per_epoch    = args.instances_per_epoch,
    )
    print(history.tail())


if __name__ == '__main__':
    main()
