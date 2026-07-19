"""
train_ga_il.py — IL training driver using the Genetic Algorithm economy-split
solver (GALabellerAdapter, wrapping src.ga.GALabeller) as the label source.
Mirrors cargoism/git/good-il-over-greedy(c)/scripts/train_h1h2_il.py exactly,
swapped to the GA labeller.

Usage:
    python scripts/train_ga_il.py \
        --data-root /Users/ramupadhyay/Desktop/good_data \
        --cache ../cache/ga_cache.pkl \
        --save-dir ../checkpoints/il
"""
from __future__ import annotations
import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.il.config import (
    DEVICE, N_EPOCHS, BATCH_SIZE, LR, PATIENCE,
    LAMBDA_WEIGHT_PENALTY, LAMBDA_VOLUME_PENALTY,
)
from src.il.model import TransformerClusterer
from src.il.labeller import GALabellerAdapter
from src.il.train_il import train_il


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--cache', default=None, help='pickle from scripts/precompute_ga_cache.py')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ga-pop-size', type=int, default=16)
    ap.add_argument('--ga-max-generations', type=int, default=20)
    ap.add_argument('--ga-patience', type=int, default=6)
    ap.add_argument('--ga-time-budget-seconds', type=float, default=90.0,
                     help='only matters on a cache miss (live GA solve fallback)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    save_dir  = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    train_dir  = os.path.join(data_root, 'synthetic_train')
    test_dir   = os.path.join(data_root, 'synthetic_test')
    train_meta = os.path.join(train_dir, 'metadata.csv')
    test_meta  = os.path.join(test_dir, 'metadata.csv')
    for p in [train_dir, test_dir, train_meta, test_meta]:
        assert os.path.exists(p), f'missing: {p}'

    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, 'rb') as f:
            cache = pickle.load(f)
        print(f'Loaded GA label cache: {len(cache)} entries from {args.cache}')
    else:
        print('No cache found -- GALabeller will solve every chunk live (slow).')

    labeller = GALabellerAdapter(
        cache=cache, pop_size=args.ga_pop_size,
        max_generations=args.ga_max_generations, patience=args.ga_patience,
        time_budget_seconds=args.ga_time_budget_seconds,
    )

    il_save = os.path.join(save_dir, 'transformer_imitation_ga.pt')
    il_log  = os.path.join(save_dir, 'il_training_log.csv')

    print(f'Device    : {DEVICE}')
    print(f'Train dir : {train_dir}')
    print(f'Test dir  : {test_dir}')
    print(f'Save path : {il_save}')

    model = TransformerClusterer().to(DEVICE)

    history = train_il(
        model,
        train_dir       = train_dir,
        test_dir        = test_dir,
        train_meta_path = train_meta,
        test_meta_path  = test_meta,
        labeller        = labeller,
        n_epochs        = args.n_epochs,
        batch_size      = BATCH_SIZE,
        lr              = LR,
        patience        = PATIENCE,
        save_path       = il_save,
        log_path        = il_log,
        device          = DEVICE,
        lambda_weight   = LAMBDA_WEIGHT_PENALTY,
        lambda_volume   = LAMBDA_VOLUME_PENALTY,
    )
    print(history.tail())


if __name__ == '__main__':
    main()
