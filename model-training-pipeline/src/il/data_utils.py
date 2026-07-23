import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    MAX_N_ULDS, MAX_N_PKGS, MAX_SEQ_LEN,
    ULD_FEAT_DIM, PKG_FEAT_DIM,
    MAX_ULD_DIM, MAX_ULD_WEIGHT, MAX_PKG_DIM, MAX_PKG_WEIGHT,
    MAX_DELAY_COST, MAX_TIGHTNESS,
    MAX_SAFE_PKGS, MAX_SAFE_ULDS,
    IGNORE_INDEX,
)
from .labeller import DEFAULT_LABELLER


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# tightness is computed per-instance and stored separately — NOT part of the
# raw feature arrays, so ULD_FEAT_DIM and PKG_FEAT_DIM stay at 7 and 9.
# The model injects tightness via a dedicated nn.Linear(1, d_model) layer.
# ─────────────────────────────────────────────────────────────────────────────

def compute_tightness(packages_df, ulds_df):
    """
    Scalar in [0, 1] representing how tight the n-1 ULD configuration is.
    tightness_raw = max(weight_tightness, volume_tightness) where:
      weight_tightness = total_pkg_weight / sum_of_(n-1)_largest_ULD_weight_limits
      volume_tightness = total_pkg_volume / sum_of_(n-1)_largest_ULD_volumes
    Values > 1.0 mean the shipment cannot fit in n-1 ULDs on that metric.
    Normalised to [0, 1] by dividing by MAX_TIGHTNESS.
    """
    n_ulds    = len(ulds_df)
    n_minus_1 = max(n_ulds - 1, 1)

    wt_limits  = sorted(ulds_df['Weight_Limit'].tolist(), reverse=True)[:n_minus_1]
    uld_vols   = sorted(
        (ulds_df['Length']*ulds_df['Width']*ulds_df['Height']).tolist(), reverse=True
    )[:n_minus_1]

    total_wt  = float(packages_df['Weight'].sum())
    total_vol = float((packages_df['Length']*packages_df['Width']*packages_df['Height']).sum())

    wt_tight  = total_wt  / max(sum(wt_limits), 1.0)
    vol_tight = total_vol / max(sum(uld_vols),  1.0)
    raw       = max(wt_tight, vol_tight)
    return min(raw / MAX_TIGHTNESS, 1.0)


def extract_uld_features(ulds_df):
    """Returns (MAX_N_ULDS, ULD_FEAT_DIM) numpy array. Tightness NOT included here."""
    feat = np.zeros((MAX_N_ULDS, ULD_FEAT_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(ulds_df.iterrows()):
        if i >= MAX_N_ULDS: break
        vol = (row['Length'] * row['Width'] * row['Height']) / (MAX_ULD_DIM**3)
        feat[i] = [
            row['Length']       / MAX_ULD_DIM,
            row['Width']        / MAX_ULD_DIM,
            row['Height']       / MAX_ULD_DIM,
            row['Weight_Limit'] / MAX_ULD_WEIGHT,
            vol,
            i / MAX_N_ULDS,
            0.0,   # reserved — keeps dim=7 stable
        ]
    return feat


def extract_pkg_features(packages_df):
    """Returns (MAX_N_PKGS, PKG_FEAT_DIM) numpy array. Tightness NOT included here."""
    feat = np.zeros((MAX_N_PKGS, PKG_FEAT_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS: break
        vol          = (row['Length']*row['Width']*row['Height']) / (MAX_PKG_DIM**3)
        is_priority  = 1.0 if row['Type'] == 'Priority' else 0.0
        delay        = row.get('Delay_Cost', 0) / MAX_DELAY_COST
        cost_density = min(delay / (vol + 1e-8), 1.0)
        feat[i] = [
            row['Length'] / MAX_PKG_DIM,
            row['Width']  / MAX_PKG_DIM,
            row['Height'] / MAX_PKG_DIM,
            row['Weight'] / MAX_PKG_WEIGHT,
            vol,
            is_priority,
            delay,
            cost_density,
            0.0,   # reserved — keeps dim=9 stable
        ]
    return feat


def compute_static_masks(packages_df, ulds_df):
    """
    dim_mask[i,j]    = True if package i physically fits in ULD j (sorted dim check)
    priority_mask[i] = True if package i is Economy (allowed to go to NONE)
    Both padded to (MAX_N_PKGS, MAX_N_ULDS) and (MAX_N_PKGS,).
    """
    uld_list = ulds_df.to_dict('records')

    dim_mask = np.zeros((MAX_N_PKGS, MAX_N_ULDS), dtype=bool)
    for i, (_, pkg) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS: break
        pd_ = sorted([pkg['Length'], pkg['Width'], pkg['Height']])
        for j, uld in enumerate(uld_list):
            if j >= MAX_N_ULDS: break
            ud_ = sorted([uld['Length'], uld['Width'], uld['Height']])
            if all(pd_[k] <= ud_[k] for k in range(3)):
                dim_mask[i, j] = True

    pm = np.zeros(MAX_N_PKGS, dtype=bool)
    for i, (_, row) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS: break
        pm[i] = (row['Type'] == 'Economy')

    return dim_mask, pm


def build_tensors(packages_df, ulds_df, device):
    """
    Build all model-input tensors for one instance (batch size = 1).
    Tightness is returned as a separate scalar tensor — the model adds it
    via a dedicated projection, not as part of the raw feature vectors.

    K is deliberately NOT part of this model's inputs -- the GA labels this
    IL model imitates are generated without any reference to K, so K
    carries no signal for this stage (see model.py's docstring).
    K-awareness only enters at RL fine-tuning (src/rl/data_utils.py).
    """
    n_ulds = len(ulds_df)
    n_pkgs = len(packages_df)

    uld_feat = extract_uld_features(ulds_df)   # (MAX_N_ULDS, 7)
    pkg_feat = extract_pkg_features(packages_df)   # (MAX_N_PKGS, 9)
    dim_mask, priority_mask = compute_static_masks(packages_df, ulds_df)

    tightness = compute_tightness(packages_df, ulds_df)   # scalar float

    # Padding mask: True = padding (ignore)
    key_mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.bool)
    key_mask[0, n_ulds:MAX_N_ULDS]          = True
    key_mask[0, MAX_N_ULDS + n_pkgs:]       = True

    return {
        'uld_feats':        torch.tensor(uld_feat, dtype=torch.float32).unsqueeze(0).to(device),
        'pkg_feats':        torch.tensor(pkg_feat, dtype=torch.float32).unsqueeze(0).to(device),
        'key_padding_mask': key_mask.to(device),
        'dim_mask':         torch.tensor(dim_mask, dtype=torch.bool).unsqueeze(0).to(device),
        'priority_mask':    torch.tensor(priority_mask, dtype=torch.bool).unsqueeze(0).to(device),
        'tightness':        torch.tensor([[tightness]], dtype=torch.float32).to(device),  # (1,1)
        'n_ulds':           n_ulds,
        'n_pkgs':           n_pkgs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING — making the pipeline safe for instances bigger than the model's
# trained capacity (MAX_N_PKGS packages, MAX_N_ULDS ULDs).
#
# WHY THIS EXISTS
# The Transformer's input tensors have fixed shape (MAX_N_PKGS, MAX_N_ULDS)
# baked into the checkpoint (positional embeddings, output head, etc.). That
# shape can't change without retraining from scratch. But real shipments can
# easily have more than 300 packages or more than 6 ULDs on a given day, so
# the *pipeline* needs to keep working correctly even when an instance is
# bigger than the model — instead of either:
#   (a) crashing (e.g. indexing a (300,) array with i=350), or
#   (b) silently truncating with `if i >= MAX_N_PKGS: break` and pretending
#       the dropped packages don't exist.
#
# STRATEGY
# - Too many ULDs  -> split the ULDs into groups of <= MAX_N_ULDS. Each group
#   is solved as an independent sub-instance against ALL the packages still
#   unassigned so far, then we move to the next ULD group with the leftover
#   packages. This mirrors how a real ops team would work: fill what you can
#   into the ULDs you can see at once, then move on to the rest.
# - Too many packages -> split the packages into chunks of <= MAX_N_PKGS.
#   Each chunk is solved against the (possibly chunked) ULDs in turn, and
#   each ULD's running weight/volume usage carries over between package
#   chunks, so capacity limits are respected globally, not just per-chunk.
# - Both at once -> nested: outer loop over ULD groups, inner loop over
#   package chunks within that ULD group.
#
# This keeps the *model* untouched (still only ever sees <= MAX_N_PKGS
# packages and <= MAX_N_ULDS ULDs per forward pass) while making the
# *pipeline* correct for arbitrarily large real-world instances.
# ─────────────────────────────────────────────────────────────────────────────

def chunk_dataframe(df, chunk_size):
    """Split a DataFrame into a list of DataFrames, each with <= chunk_size rows."""
    if len(df) <= chunk_size:
        return [df.reset_index(drop=True)]
    return [df.iloc[i:i + chunk_size].reset_index(drop=True)
            for i in range(0, len(df), chunk_size)]


def needs_chunking(packages_df, ulds_df, max_pkgs=None, max_ulds=None):
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds
    return len(packages_df) > max_pkgs or len(ulds_df) > max_ulds


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
#
# Instances bigger than the model's capacity (more than MAX_N_PKGS packages,
# or more than MAX_N_ULDS ULDs) are split into multiple sub-instances via
# chunk_dataframe() at load time, instead of being silently truncated. Each
# sub-instance is labelled independently and exposed as its own dataset row,
# so __len__ can grow past len(meta) when oversized instances are present,
# and every package the heuristic would have placed is actually present in
# the training data.
# ─────────────────────────────────────────────────────────────────────────────

class ClusteringDataset(Dataset):
    """
    Loads instances from disk, generates greedy labels via the Labeller,
    and returns tensors ready for the training loop.

    label_map : {Package_ID: ULD_ID | 'NONE'} -> int label
        ULD_ID maps to its 0-based index in ulds_df (within its sub-instance)
        'NONE' maps to MAX_N_ULDS
        Padding packages -> IGNORE_INDEX (= -100)

    Oversized instances (more than MAX_N_PKGS packages and/or more than
    MAX_N_ULDS ULDs) are split into multiple (packages_chunk, ulds_chunk)
    sub-instances at load time — see _expand_index(). Each chunk fits the
    model's fixed input shape, so nothing is ever dropped or truncated.
    """

    def __init__(self, data_dir, meta_path, labeller=None, device='cpu'):
        self.data_dir  = data_dir
        self.meta      = pd.read_csv(meta_path)
        self.labeller  = labeller or DEFAULT_LABELLER
        self.device    = device
        self._cache    = {}   # row idx -> tensors+labels, loaded lazily
        self._index    = self._expand_index()

    def _expand_index(self):
        """
        Build a flat list of (tag, pkg_chunk_idx, uld_chunk_idx) tuples, one
        per sub-instance the dataset will actually yield. Most instances
        produce exactly one entry; oversized ones produce several.
        """
        index = []
        for _, row in self.meta.iterrows():
            tag = row['instance']
            u_path = os.path.join(self.data_dir, f'{tag}_ulds.csv')
            p_path = os.path.join(self.data_dir, f'{tag}_packages.csv')
            n_ulds = len(pd.read_csv(u_path, usecols=[0]))
            n_pkgs = len(pd.read_csv(p_path, usecols=[0]))
            n_uld_chunks = max(1, math.ceil(n_ulds / MAX_SAFE_ULDS))
            n_pkg_chunks = max(1, math.ceil(n_pkgs / MAX_SAFE_PKGS))
            for ui in range(n_uld_chunks):
                for pi in range(n_pkg_chunks):
                    index.append((tag, pi, ui))
        return index

    def __len__(self):
        return len(self._index)

    def _load(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        tag, pkg_chunk_idx, uld_chunk_idx = self._index[idx]
        u_path   = os.path.join(self.data_dir, f'{tag}_ulds.csv')
        p_path   = os.path.join(self.data_dir, f'{tag}_packages.csv')
        full_ulds_df = pd.read_csv(u_path)
        full_pkgs_df = pd.read_csv(p_path)

        uld_chunks = chunk_dataframe(full_ulds_df, MAX_SAFE_ULDS)
        pkg_chunks = chunk_dataframe(full_pkgs_df, MAX_SAFE_PKGS)
        ulds_df    = uld_chunks[uld_chunk_idx]
        pkgs_df    = pkg_chunks[pkg_chunk_idx]

        # Namespace by data_dir: 'instance_000' is reused independently by both
        # synthetic_train and synthetic_test, so the bare tag is NOT a unique
        # labeller-cache key across the two ClusteringDataset instances that
        # train_il() builds (they share one Labeller object).
        split_tag = f'{os.path.basename(os.path.normpath(self.data_dir))}/{tag}'
        assignment = self.labeller.label(
            pkgs_df, ulds_df,
            tag=split_tag, pkg_chunk_idx=pkg_chunk_idx, uld_chunk_idx=uld_chunk_idx,
        )

        # Build label tensor (indices are local to this ULD chunk)
        uld_id_to_idx = {row['ULD_ID']: j for j, (_, row) in enumerate(ulds_df.iterrows())}
        label_arr     = np.full(MAX_N_PKGS, IGNORE_INDEX, dtype=np.int64)
        for i, (_, pkg) in enumerate(pkgs_df.iterrows()):
            if i >= MAX_N_PKGS: break   # unreachable: chunks are already <= MAX_N_PKGS
            uid = assignment.get(pkg['Package_ID'], 'NONE')
            label_arr[i] = MAX_N_ULDS if uid == 'NONE' else uld_id_to_idx.get(uid, MAX_N_ULDS)

        tensors = build_tensors(pkgs_df, ulds_df, self.device)
        tensors['labels']  = torch.tensor(label_arr, dtype=torch.long)
        tensors['n_ulds']  = len(ulds_df)
        tensors['n_pkgs']  = len(pkgs_df)
        tensors['tag']     = (f'{tag}__uldchunk{uld_chunk_idx}_pkgchunk{pkg_chunk_idx}'
                              if len(uld_chunks) > 1 or len(pkg_chunks) > 1 else tag)

        self._cache[idx] = tensors
        return tensors

    def __getitem__(self, idx):
        return self._load(idx)


def collate_fn(batch):
    """Stack variable-n_ulds instances into a batch."""
    out  = {}
    for k in ['uld_feats','pkg_feats','key_padding_mask',
            'dim_mask','priority_mask','tightness']:
        # These tensors from __getitem__ are already (1, ...) or (1,1) for tightness
        out[k] = torch.cat([b[k] for b in batch], dim=0)

    # labels are (MAX_N_PKGS,) from __getitem__, so they need stacking
    out['labels'] = torch.stack([b['labels'] for b in batch], dim=0)

    out['n_ulds_batch'] = torch.tensor([b['n_ulds'] for b in batch], dtype=torch.long)
    out['n_pkgs_list']  = [b['n_pkgs'] for b in batch]
    return out
