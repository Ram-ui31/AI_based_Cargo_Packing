import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import (
    MAX_N_ULDS, MAX_N_PKGS, MAX_SEQ_LEN, N_ULD_CLASSES,
    ULD_FEAT_DIM, PKG_FEAT_DIM,
    MAX_ULD_DIM, MAX_ULD_WEIGHT, MAX_PKG_DIM, MAX_PKG_WEIGHT,
    MAX_DELAY_COST, MAX_TIGHTNESS,
    MAX_SAFE_PKGS, MAX_SAFE_ULDS,
    DEVICE,
)


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_tightness(packages_df, ulds_df):
    n_ulds    = len(ulds_df)
    n_minus_1 = max(n_ulds - 1, 1)
    wt_limits = sorted(ulds_df['Weight_Limit'].tolist(), reverse=True)[:n_minus_1]
    uld_vols  = sorted(
        (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist(), reverse=True
    )[:n_minus_1]
    total_wt  = float(packages_df['Weight'].sum())
    total_vol = float((packages_df['Length'] * packages_df['Width'] * packages_df['Height']).sum())
    wt_tight  = total_wt  / max(sum(wt_limits), 1.0)
    vol_tight = total_vol / max(sum(uld_vols),  1.0)
    return min(max(wt_tight, vol_tight) / MAX_TIGHTNESS, 1.0)


def extract_uld_features(ulds_df):
    feat = np.zeros((MAX_N_ULDS, ULD_FEAT_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(ulds_df.iterrows()):
        if i >= MAX_N_ULDS:
            break
        vol = float(row['Length']) * float(row['Width']) * float(row['Height'])
        feat[i] = [
            row['Length']       / MAX_ULD_DIM,
            row['Width']        / MAX_ULD_DIM,
            row['Height']       / MAX_ULD_DIM,
            row['Weight_Limit'] / MAX_ULD_WEIGHT,
            vol / (MAX_ULD_DIM ** 3),
            i / MAX_N_ULDS,
            0.0,   # reserved slot — keeps dim=7 stable
        ]
    return feat


def extract_pkg_features(packages_df):
    feat = np.zeros((MAX_N_PKGS, PKG_FEAT_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS:
            break
        vol          = float(row['Length']) * float(row['Width']) * float(row['Height'])
        is_priority  = 1.0 if row['Type'] == 'Priority' else 0.0
        delay        = float(row.get('Delay_Cost', 0)) / MAX_DELAY_COST
        cost_density = min(delay / (vol / (MAX_PKG_DIM ** 3) + 1e-8), 1.0)
        feat[i] = [
            row['Length'] / MAX_PKG_DIM,
            row['Width']  / MAX_PKG_DIM,
            row['Height'] / MAX_PKG_DIM,
            row['Weight'] / MAX_PKG_WEIGHT,
            vol / (MAX_PKG_DIM ** 3),
            is_priority,
            delay,
            cost_density,
            0.0,   # reserved slot — keeps dim=9 stable
        ]
    return feat


def compute_static_masks(packages_df, ulds_df):
    uld_list = ulds_df.to_dict('records')
    dm = np.zeros((MAX_N_PKGS, MAX_N_ULDS), dtype=bool)
    for i, (_, pkg) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS:
            break
        pd_ = sorted([pkg['Length'], pkg['Width'], pkg['Height']])
        for j, uld in enumerate(uld_list):
            if j >= MAX_N_ULDS:
                break
            ud_ = sorted([uld['Length'], uld['Width'], uld['Height']])
            if all(pd_[k] <= ud_[k] for k in range(3)):
                dm[i, j] = True
    pm = np.zeros(MAX_N_PKGS, dtype=bool)
    for i, (_, row) in enumerate(packages_df.iterrows()):
        if i >= MAX_N_PKGS:
            break
        pm[i] = (row['Type'] == 'Economy')
    return dm, pm


def build_tensors(packages_df, ulds_df, device):
    """Build model input tensors for one instance. Tightness is a separate (1,1) tensor."""
    uld_f = extract_uld_features(ulds_df)
    pkg_f = extract_pkg_features(packages_df)
    dm, pm = compute_static_masks(packages_df, ulds_df)
    tightness = compute_tightness(packages_df, ulds_df)

    n_ulds = len(ulds_df)
    n_pkgs = len(packages_df)
    key_mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.bool)
    key_mask[0, n_ulds:MAX_N_ULDS]     = True
    key_mask[0, MAX_N_ULDS + n_pkgs:]  = True

    dm_pad = np.zeros((MAX_N_PKGS, MAX_N_ULDS), dtype=bool)
    pm_pad = np.zeros(MAX_N_PKGS, dtype=bool)
    dm_pad[:n_pkgs, :n_ulds] = dm[:n_pkgs, :n_ulds]
    pm_pad[:n_pkgs]           = pm[:n_pkgs]

    return {
        'uld_feats':        torch.tensor(uld_f, dtype=torch.float32).unsqueeze(0).to(device),
        'pkg_feats':        torch.tensor(pkg_f, dtype=torch.float32).unsqueeze(0).to(device),
        'key_padding_mask': key_mask.to(device),
        'dim_mask':         torch.tensor(dm_pad, dtype=torch.bool).unsqueeze(0).to(device),
        'priority_mask':    torch.tensor(pm_pad, dtype=torch.bool).unsqueeze(0).to(device),
        'tightness':        torch.tensor([[tightness]], dtype=torch.float32).to(device),
    }


def load_instance_cache(data_dir, split, meta_df):
    """Load all ULD/package CSVs for a split into a {tag: (ulds_df, pkgs_df)} dict."""
    import os
    cache = {}
    for _, row in meta_df.iterrows():
        tag    = row['instance']
        u_path = os.path.join(data_dir, split, f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, split, f'{tag}_packages.csv')
        if os.path.exists(u_path) and os.path.exists(p_path):
            cache[tag] = (pd.read_csv(u_path), pd.read_csv(p_path))
    return cache


# ── Chunking utilities ────────────────────────────────────────────────────────
# The Transformer has fixed-shape input tensors baked into the checkpoint
# (positional embeddings, output head). Real instances can exceed MAX_N_PKGS
# or MAX_N_ULDS, so we chunk rather than truncate or crash.

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


# ── IL baseline sampling ──────────────────────────────────────────────────────

@torch.no_grad()
def il_sample_assignment(il_model, tensors, n_ulds, n_pkgs,
                          packages_df, ulds_df,
                          n_samples=8, temperature=1.0):
    """
    Draw n_samples independent assignments from the IL model's distribution.

    PRECONDITION: n_pkgs <= MAX_N_PKGS and n_ulds <= MAX_N_ULDS.
    For larger instances use il_sample_assignment_safe().

    Returns:
        assignments : list of n_samples {Package_ID: ULD_ID|'NONE'} dicts
        il_logits   : (MAX_N_PKGS, N_ULD_CLASSES) raw IL logits
        weight_used : list[float] per-ULD weight used (last sample)
        volume_used : list[float] per-ULD volume used (last sample)
    """
    il_model.eval()
    device = tensors['uld_feats'].device

    il_logits_full = il_model.forward(
        tensors['uld_feats'], tensors['pkg_feats'],
        tensors['key_padding_mask'],
        torch.tensor([n_ulds], device=device),
        tensors['dim_mask'], tensors['priority_mask'],
        tensors['tightness'],
    ).squeeze(0)
    il_logits = il_logits_full

    pkg_ids       = packages_df['Package_ID'].tolist()
    uld_ids       = ulds_df['ULD_ID'].tolist()
    uld_wt_limits = ulds_df['Weight_Limit'].tolist()
    pkg_weights   = packages_df['Weight'].tolist()
    uld_volumes   = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist()
    pkg_volumes   = (packages_df['Length'] * packages_df['Width'] * packages_df['Height']).tolist()

    assignments = []
    for _ in range(n_samples):
        weight_used = [0.0] * n_ulds
        volume_used = [0.0] * n_ulds
        assignment  = {}
        for i in range(min(n_pkgs, MAX_N_PKGS)):
            lg = il_logits[i].clone()
            is_priority = (lg[MAX_N_ULDS].item() < -1e8)
            for j in range(n_ulds):
                if weight_used[j] + pkg_weights[i] > uld_wt_limits[j]:
                    lg[j] = -1e9
            for j in range(n_ulds):
                if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                    lg[j] = -1e9

            def _relative_overflow(j):
                wt_over  = max(0.0, (weight_used[j] + pkg_weights[i] - uld_wt_limits[j])
                               / max(uld_wt_limits[j], 1e-6))
                vol_over = max(0.0, (volume_used[j] + pkg_volumes[i] - uld_volumes[j])
                               / max(uld_volumes[j], 1e-6))
                return max(wt_over, vol_over)

            if is_priority and all(lg[j].item() < -1e8 for j in range(n_ulds)):
                fallback     = min(range(n_ulds), key=_relative_overflow)
                lg[fallback] = 0.0

            probs  = F.softmax(lg / temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()
            if is_priority and action == MAX_N_ULDS:
                action = min(range(n_ulds), key=_relative_overflow)

            assignment[pkg_ids[i]] = ('NONE' if action == MAX_N_ULDS
                                      else uld_ids[action])
            if action < n_ulds:
                weight_used[action] += pkg_weights[i]
                volume_used[action] += pkg_volumes[i]
        assignments.append(assignment)

    return assignments, il_logits, weight_used, volume_used


def il_sample_assignment_safe(il_model, packages_df, ulds_df, device,
                               n_samples=8, temperature=1.0,
                               max_pkgs=None, max_ulds=None):
    """
    Chunk-safe version of il_sample_assignment() for instances of any size.

    Returns: list of n_samples {Package_ID: ULD_ID|'NONE'} dicts.
    """
    device   = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    if not needs_chunking(packages_df, ulds_df, max_pkgs, max_ulds):
        tensors = build_tensors(packages_df, ulds_df, device)
        assignments, _, _, _ = il_sample_assignment(
            il_model, tensors, len(ulds_df), len(packages_df),
            packages_df, ulds_df, n_samples=n_samples, temperature=temperature,
        )
        return assignments

    assignments       = [dict() for _ in range(n_samples)]
    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks        = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break
        n_ulds_here = len(uld_chunk)
        for pkg_chunk in chunk_dataframe(remaining_pkgs_df, max_pkgs):
            tensors = build_tensors(pkg_chunk, uld_chunk, device)
            chunk_assignments, _, _, _ = il_sample_assignment(
                il_model, tensors, n_ulds_here, len(pkg_chunk),
                pkg_chunk, uld_chunk, n_samples=n_samples, temperature=temperature,
            )
            for s in range(n_samples):
                assignments[s].update(chunk_assignments[s])

        unresolved_ids = set(remaining_pkgs_df['Package_ID'])
        for s in range(n_samples):
            unresolved_ids &= {pid for pid in remaining_pkgs_df['Package_ID']
                               if assignments[s].get(pid, 'NONE') == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(unresolved_ids)
        ].reset_index(drop=True)

    return assignments


def actions_to_assignment(actions, n_pkgs, packages_df, ulds_df):
    """Convert int action tensor -> {Package_ID: ULD_ID or 'NONE'}."""
    pkg_ids = packages_df['Package_ID'].tolist()
    uld_ids = ulds_df['ULD_ID'].tolist()
    return {pkg_ids[i]: ('NONE' if actions[i].item() == MAX_N_ULDS
                         else uld_ids[actions[i].item()])
            for i in range(n_pkgs)}
