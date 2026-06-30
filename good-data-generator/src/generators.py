import os

import numpy as np
import pandas as pd

from .config import (
    ULD_L_MAX, ULD_W_MAX, ULD_H_MAX, ULD_DIM_MIN, ULD_DIM_STEP,
    N_ULDS_MIN, N_ULDS_MAX,
    PKG_DIM_MIN, PKG_DIM_MAX, PKG_WT_MIN, PKG_WT_MAX,
    DELAY_MIN, DELAY_MAX,
    PRIO_RATIO_MIN, PRIO_RATIO_MAX,
    TARGET_VOL_RATIO, TARGET_WT_RATIO, RATIO_NOISE,
    PRIO_SHARE_BASE, PRIO_SHARE_NOISE, PRIO_SHARE_SPREAD,
)
from .sampling import (
    _L_POOL, _W_POOL, _H_POOL, _WT_POOL,
    _avg_pkg_vol, _avg_pkg_wt, _scale_weights, _scale_dims_to_vol,
)


# ─────────────────────────────────────────────────────────────────────────────
# ULD GENERATOR
#
# Samples L, W, H independently from their allowed pools,
# sorts descending to enforce L >= W >= H,
# then re-clamps each axis to its ceiling after sorting.
# ─────────────────────────────────────────────────────────────────────────────

def generate_ulds(n_ulds, rng):
    rows = []
    for i in range(n_ulds):
        l = int(rng.choice(_L_POOL))
        w = int(rng.choice(_W_POOL))
        h = int(rng.choice(_H_POOL))

        # Sort descending → largest becomes L
        dl, dw, dh = sorted([l, w, h], reverse=True)

        # Re-clamp each axis to its ceiling (sorting may move a large W into L slot)
        dl = min(dl, ULD_L_MAX)
        dw = min(dw, ULD_W_MAX)
        dh = min(dh, ULD_H_MAX)

        # Snap back to multiple of 5 after clamping, keep above floor
        dl = max(ULD_DIM_MIN, (dl // ULD_DIM_STEP) * ULD_DIM_STEP)
        dw = max(ULD_DIM_MIN, (dw // ULD_DIM_STEP) * ULD_DIM_STEP)
        dh = max(ULD_DIM_MIN, (dh // ULD_DIM_STEP) * ULD_DIM_STEP)

        wt = int(rng.choice(_WT_POOL))
        rows.append({
            'ULD_ID':       f'U{i + 1}',
            'Length':       dl,
            'Width':        dw,
            'Height':       dh,
            'Weight_Limit': wt,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PACKAGE GENERATOR
#
# Per-axis dim ceiling  =  min(PKG_DIM_MAX, min ULD dim on that axis)
# This GUARANTEES no package dim > any ULD dim on that axis.
#
# Package count is derived as max(vol-implied, wt-implied) so that both
# target_vol and target_wt are reachable after scaling.
#
# Priority and Economy packages are generated as two separate pools, each
# scaled to its OWN volume/weight target:
#   - Priority pool  -> scaled so sum(priority_vol)/uld_vol and
#                        sum(priority_wt)/uld_wt land close to PRIO_SHARE_BASE
#                        (jittered, but the two stay close to each other —
#                        matching input.csv).
#   - Economy pool   -> sized so the overall totals (priority + economy)
#                        still hit TARGET_VOL_RATIO / TARGET_WT_RATIO.
# This guarantees both the overall ratio constraints AND the priority-share
# constraint hold simultaneously.
# ─────────────────────────────────────────────────────────────────────────────

def generate_packages(ulds_df, rng):
    uld_vol = float((ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum())
    uld_wt  = float(ulds_df['Weight_Limit'].sum())

    # Per-axis dim ceiling: smallest ULD dim on that axis, capped at PKG_DIM_MAX
    max_l = max(PKG_DIM_MIN, min(PKG_DIM_MAX, int(ulds_df['Length'].min())))
    max_w = max(PKG_DIM_MIN, min(PKG_DIM_MAX, int(ulds_df['Width'].min())))
    max_h = max(PKG_DIM_MIN, min(PKG_DIM_MAX, int(ulds_df['Height'].min())))

    # ── Overall targets ────────────────────────────────────────────────────────
    vr = TARGET_VOL_RATIO * rng.uniform(1 - RATIO_NOISE, 1 + RATIO_NOISE)
    wr = TARGET_WT_RATIO  * rng.uniform(1 - RATIO_NOISE, 1 + RATIO_NOISE)
    target_vol_total = vr * uld_vol
    target_wt_total  = wr * uld_wt

    # ── Priority-share targets ────────────────────────────────────────────────
    # One shared base draw keeps the two ratios correlated/close, then each
    # axis (vol vs wt) gets its own small jitter so instances vary a bit.
    share_base = PRIO_SHARE_BASE * rng.uniform(1 - PRIO_SHARE_NOISE, 1 + PRIO_SHARE_NOISE)
    share_vol  = np.clip(share_base + rng.uniform(-PRIO_SHARE_SPREAD, PRIO_SHARE_SPREAD), 0.05, 0.95)
    share_wt   = np.clip(share_base + rng.uniform(-PRIO_SHARE_SPREAD, PRIO_SHARE_SPREAD), 0.05, 0.95)

    target_prio_vol = share_vol * uld_vol
    target_prio_wt  = share_wt  * uld_wt

    # Economy gets whatever is left of the overall total after priority's share
    # (floored a little above zero so economy always has a non-trivial target)
    target_econ_vol = max(target_vol_total - target_prio_vol, 0.05 * uld_vol)
    target_econ_wt  = max(target_wt_total  - target_prio_wt,  0.05 * uld_wt)

    # ── Package counts ────────────────────────────────────────────────────────
    avg_vol = _avg_pkg_vol(max_l, max_w, max_h)
    avg_wt  = _avg_pkg_wt()

    ratio  = rng.uniform(PRIO_RATIO_MIN, PRIO_RATIO_MAX)

    n_prio_from_vol = int(round(target_prio_vol / avg_vol))
    n_prio_from_wt  = int(round(target_prio_wt  / avg_wt))
    n_prio = max(1, max(n_prio_from_vol, n_prio_from_wt))

    n_econ_from_vol = int(round(target_econ_vol / avg_vol))
    n_econ_from_wt  = int(round(target_econ_wt  / avg_wt))
    n_econ_floor    = max(1, int(round(n_prio * (1 - ratio) / ratio)))  # keep priority ratio in range
    n_econ = max(n_econ_floor, max(n_econ_from_vol, n_econ_from_wt))

    n_pkgs = n_prio + n_econ

    # ── Priority pool: sample raw dims/weights, scale to its own targets ──────
    prl = rng.randint(PKG_DIM_MIN, max_l + 1, size=n_prio).astype(float)
    prw = rng.randint(PKG_DIM_MIN, max_w + 1, size=n_prio).astype(float)
    prh = rng.randint(PKG_DIM_MIN, max_h + 1, size=n_prio).astype(float)
    prl, prw, prh = _scale_dims_to_vol(prl, prw, prh, target_prio_vol, max_l, max_w, max_h)

    praw_wt = rng.randint(PKG_WT_MIN, PKG_WT_MAX + 1, size=n_prio).astype(float)
    pr_wt   = _scale_weights(praw_wt, target_prio_wt, PKG_WT_MIN, PKG_WT_MAX)

    # ── Economy pool: same approach, own targets ───────────────────────────────
    erl = rng.randint(PKG_DIM_MIN, max_l + 1, size=n_econ).astype(float)
    erw = rng.randint(PKG_DIM_MIN, max_w + 1, size=n_econ).astype(float)
    erh = rng.randint(PKG_DIM_MIN, max_h + 1, size=n_econ).astype(float)
    erl, erw, erh = _scale_dims_to_vol(erl, erw, erh, target_econ_vol, max_l, max_w, max_h)

    eraw_wt = rng.randint(PKG_WT_MIN, PKG_WT_MAX + 1, size=n_econ).astype(float)
    e_wt    = _scale_weights(eraw_wt, target_econ_wt, PKG_WT_MIN, PKG_WT_MAX)

    # ── Delay costs (Priority = 0, Economy = sampled) ─────────────────────────
    prio_delays = [0] * n_prio
    econ_delays = list(rng.randint(DELAY_MIN, DELAY_MAX + 1, size=n_econ))

    # ── Combine + shuffle ──────────────────────────────────────────────────────
    rl = np.concatenate([prl, erl])
    rw = np.concatenate([prw, erw])
    rh = np.concatenate([prh, erh])
    wt = np.concatenate([pr_wt, e_wt])
    types  = ['Priority'] * n_prio + ['Economy'] * n_econ
    delays = prio_delays + econ_delays

    order = rng.permutation(n_pkgs)
    rows = [
        {
            'Package_ID': f'P-{rank + 1}',
            'Length':     int(rl[i]),
            'Width':      int(rw[i]),
            'Height':     int(rh[i]),
            'Weight':     int(wt[i]),
            'Type':       types[i],
            'Delay_Cost': int(delays[i]),
        }
        for rank, i in enumerate(order)
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_instance(seed):
    rng    = np.random.RandomState(seed)
    n_ulds = int(rng.randint(N_ULDS_MIN, N_ULDS_MAX + 1))

    ulds_df = generate_ulds(n_ulds, rng)
    pkgs_df = generate_packages(ulds_df, rng)

    uld_vol  = float((ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).sum())
    uld_wt   = float(ulds_df['Weight_Limit'].sum())
    pkg_vol  = float((pkgs_df['Length'] * pkgs_df['Width'] * pkgs_df['Height']).sum())
    pkg_wt   = float(pkgs_df['Weight'].sum())

    is_prio   = pkgs_df['Type'] == 'Priority'
    n_prio    = int(is_prio.sum())
    n_pkgs    = len(pkgs_df)

    prio_vol  = float((pkgs_df.loc[is_prio, 'Length'] *
                        pkgs_df.loc[is_prio, 'Width']  *
                        pkgs_df.loc[is_prio, 'Height']).sum())
    prio_wt   = float(pkgs_df.loc[is_prio, 'Weight'].sum())

    meta = {
        'instance':           None,
        'seed':                seed,
        'n_ulds':              n_ulds,
        'n_packages':          n_pkgs,
        'n_priority':          n_prio,
        'n_economy':           n_pkgs - n_prio,
        'priority_ratio':      round(n_prio / n_pkgs, 4),
        'vol_ratio':           round(pkg_vol / uld_vol, 4),
        'wt_ratio':            round(pkg_wt  / uld_wt,  4),
        'prio_vol_uld_ratio':  round(prio_vol / uld_vol, 4),   # sum(priority_vol) / sum(uld_vol)
        'prio_wt_uld_ratio':   round(prio_wt  / uld_wt,  4),   # sum(priority_wt)  / sum(uld_wt)
        'uld_total_vol':       int(uld_vol),
        'uld_total_wt':        int(uld_wt),
        'pkg_total_vol':       int(pkg_vol),
        'pkg_total_wt':        int(pkg_wt),
    }
    return ulds_df, pkgs_df, meta


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_instances, output_dir, base_seed, split_name, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    meta_rows = []

    for i in range(n_instances):
        seed = base_seed + i
        tag  = f'instance_{i:03d}'

        ulds_df, pkgs_df, meta = generate_instance(seed)
        meta['instance'] = tag

        ulds_df.to_csv(os.path.join(output_dir, f'{tag}_ulds.csv'),     index=False)
        pkgs_df.to_csv(os.path.join(output_dir, f'{tag}_packages.csv'), index=False)
        meta_rows.append(meta)

        if verbose and (i == 0 or (i + 1) % 200 == 0 or i + 1 == n_instances):
            print(f'  [{split_name}] {i+1:4d}/{n_instances}  '
                  f'seed={seed}  n_ulds={meta["n_ulds"]}  '
                  f'n_pkgs={meta["n_packages"]:3d}  '
                  f'prio_ratio={meta["priority_ratio"]:.3f}  '
                  f'vol_r={meta["vol_ratio"]:.3f}  '
                  f'wt_r={meta["wt_ratio"]:.3f}  '
                  f'prio_vol/uld={meta["prio_vol_uld_ratio"]:.3f}  '
                  f'prio_wt/uld={meta["prio_wt_uld_ratio"]:.3f}')

    cols    = ['instance'] + [c for c in meta_rows[0] if c != 'instance']
    meta_df = pd.DataFrame(meta_rows)[cols]
    meta_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    return meta_df
