"""
features.py -- shared feature extraction for the real 400-package instance,
used identically by train.py (building training tensors) and
infer_and_validate.py (scoring at inference time) so features never drift
between training and inference.
"""
from __future__ import annotations

import numpy as np


def build_package_features(economy_df, avg_uld_volume, avg_uld_weight):
    """Returns (n_items, 9) float32 array: length, width, height, volume,
    weight, delay_cost, value_density, volume_frac, weight_frac."""
    vol = (economy_df['Length'] * economy_df['Width'] * economy_df['Height']).to_numpy(dtype=np.float32)
    weight = economy_df['Weight'].to_numpy(dtype=np.float32)
    delay = economy_df['Delay_Cost'].to_numpy(dtype=np.float32)
    value_density = delay / np.clip(vol, 1, None)
    volume_frac = vol / avg_uld_volume
    weight_frac = weight / avg_uld_weight
    feats = np.stack([
        economy_df['Length'].to_numpy(dtype=np.float32),
        economy_df['Width'].to_numpy(dtype=np.float32),
        economy_df['Height'].to_numpy(dtype=np.float32),
        vol, weight, delay, value_density, volume_frac, weight_frac,
    ], axis=1)
    return feats


def build_global_features(n_ulds, total_remaining_volume, total_remaining_weight, k_value):
    return np.array([n_ulds, total_remaining_volume, total_remaining_weight, k_value], dtype=np.float32)


def normalize_features(feats, mean=None, std=None):
    """Per-column standardization. Returns (normalized_feats, mean, std) --
    pass mean/std back in at inference time to match training exactly."""
    if mean is None:
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mean) / std, mean, std
