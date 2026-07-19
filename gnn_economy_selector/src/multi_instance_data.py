"""
multi_instance_data.py -- loads instances from good_data/synthetic_train
(~1000 varied instances) and good_data/synthetic_test (~100), plus the
real 400-package benchmark (~/Downloads/input.csv), through one unified
interface for multi-instance training.

This is the data generalization fix vs. tonight's single-instance work:
package/ULD counts vary across instances (86-375 packages in the training
set), and features must be computed FRESH per instance (never cached from
one specific instance into a checkpoint) for the model to actually
transfer across instances.
"""
from __future__ import annotations
import os
import random

import pandas as pd

GOOD_DATA_ROOT = os.path.expanduser('~/Desktop/good_data')
REAL_BENCHMARK_PATH = os.path.expanduser('~/Downloads/input.csv')


def load_synthetic_instance(split, instance_name):
    """split: 'synthetic_train' or 'synthetic_test'. Returns (k_value, ulds_df, pkgs_df)."""
    base = os.path.join(GOOD_DATA_ROOT, split)
    pkgs_df = pd.read_csv(os.path.join(base, f'{instance_name}_packages.csv'))
    ulds_df = pd.read_csv(os.path.join(base, f'{instance_name}_ulds.csv'))
    meta = pd.read_csv(os.path.join(base, 'metadata_with_K.csv'))
    k_value = float(meta.loc[meta['instance'] == instance_name, 'K'].iloc[0])
    return k_value, ulds_df, pkgs_df


def load_real_benchmark():
    """The specific real 400-package instance this whole session has targeted."""
    with open(REAL_BENCHMARK_PATH) as f:
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


class InstanceSampler:
    """Samples training instances: mostly from synthetic_train (for
    generalization), with the real benchmark oversampled at `benchmark_frac`
    so the model doesn't drift away from what we're actually being scored
    on (29,203 target)."""

    def __init__(self, benchmark_frac=0.15, seed=0):
        meta = pd.read_csv(os.path.join(GOOD_DATA_ROOT, 'synthetic_train', 'metadata_with_K.csv'))
        self.train_instances = meta['instance'].tolist()
        self.benchmark_frac = benchmark_frac
        self.rng = random.Random(seed)
        self._benchmark_cache = None

    def sample(self):
        if self.rng.random() < self.benchmark_frac:
            if self._benchmark_cache is None:
                self._benchmark_cache = load_real_benchmark()
            return 'REAL_BENCHMARK', self._benchmark_cache
        name = self.rng.choice(self.train_instances)
        return name, load_synthetic_instance('synthetic_train', name)
