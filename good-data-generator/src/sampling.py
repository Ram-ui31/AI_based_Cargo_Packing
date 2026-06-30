import math

import numpy as np

from .config import (
    ULD_DIM_MIN, ULD_L_MAX, ULD_W_MAX, ULD_H_MAX,
    ULD_WT_MIN, ULD_WT_MAX, ULD_WT_STEP,
    PKG_DIM_MIN, PKG_WT_MIN, PKG_WT_MAX,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pool(lo, hi, step=5):
    """All multiples of step in [lo, hi]."""
    return np.arange(math.ceil(lo / step) * step, hi + 1, step)


_L_POOL  = _pool(ULD_DIM_MIN, ULD_L_MAX)
_W_POOL  = _pool(ULD_DIM_MIN, ULD_W_MAX)
_H_POOL  = _pool(ULD_DIM_MIN, ULD_H_MAX)
_WT_POOL = _pool(ULD_WT_MIN,  ULD_WT_MAX, step=ULD_WT_STEP)


def _avg_pkg_vol(max_l, max_w, max_h):
    """Expected volume of one package given per-axis ceilings (uniform draw)."""
    return ((PKG_DIM_MIN + max_l) / 2 *
            (PKG_DIM_MIN + max_w) / 2 *
            (PKG_DIM_MIN + max_h) / 2)


def _avg_pkg_wt():
    return (PKG_WT_MIN + PKG_WT_MAX) / 2


def _scale_weights(raw, target, lo, hi, iters=30):
    """
    Iterative proportional scaling of a weight array to sum to target,
    clamping each value to [lo, hi] at every iteration.
    Converges in <10 iterations in practice.
    """
    w = raw.copy().astype(float)
    for _ in range(iters):
        cur = w.sum()
        if abs(cur - target) < 0.5:
            break
        w = np.clip(w * (target / cur), lo, hi)
    return np.round(w).astype(int)


def _scale_dims_to_vol(rl, rw, rh, target_vol, max_l, max_w, max_h):
    """Uniformly (cube-root) scale a set of dims toward target_vol, clamping per-axis."""
    rl = rl.astype(float).copy()
    rw = rw.astype(float).copy()
    rh = rh.astype(float).copy()
    cur_vol = (rl * rw * rh).sum()
    if cur_vol > 0:
        sc = (target_vol / cur_vol) ** (1.0 / 3.0)
        rl = np.clip(np.round(rl * sc).astype(int), PKG_DIM_MIN, max_l)
        rw = np.clip(np.round(rw * sc).astype(int), PKG_DIM_MIN, max_w)
        rh = np.clip(np.round(rh * sc).astype(int), PKG_DIM_MIN, max_h)
    else:
        rl, rw, rh = rl.astype(int), rw.astype(int), rh.astype(int)
    return rl, rw, rh
