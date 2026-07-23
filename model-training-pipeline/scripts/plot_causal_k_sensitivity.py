"""
plot_causal_k_sensitivity.py -- regenerates results/contrastive_causal_k_sensitivity.png
and results/model_learned_k_adaptation.png with the contrastive_v7 checkpoint,
the current best (soft_spread_loss_ipr + economy-only entropy + K-gap-scaled
contrastive margin + K-weighted consolidation-imitation loss), replacing the
stale plots from earlier attempts that either showed a flat, non-causal
spread curve (spread/hinge coefficient tuning, multi-K training data alone,
the first contrastive pass with the old saturating soft_spread_loss) or
were superseded by a strictly-better checkpoint under production evaluation
(v3 through v6 -- see checkpoints/rl_ppo_contrastive_v7/ for the full
lineage in training log form).

Note: this same-instance K-sweep runs through the PRODUCTION pipeline
(rl_assign_argmax_safe), which still gates Priority-ULD assignment through
the _consolidate_priority_by_capacity heuristic once K >= 500 -- the model's
own predictions only govern that decision below K=500. A same-instance sweep
with the heuristic disabled entirely (diagnostic only, not the shipped
config) still trails the heuristic by ~10% mean cost in aggregate even with
v7, though several individual instances now match or beat it -- the model
has NOT fully replaced the heuristic despite the fixes below; it has gotten
meaningfully closer, especially in the K<500 range the heuristic never
touches.

Same-instance K-sweep methodology throughout: for each sampled instance,
run rl_assign_argmax_safe + RLPackerAdapter at every K in K_VALUES and
record spread (n_priority_ulds) and total cost. A flat line across K means
the model learned a K-correlated constant bias, not a causal response --
this is exactly the test that debunked the earliest attempts.

Usage:
    python scripts/plot_causal_k_sensitivity.py \
        --checkpoint checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt \
        --n-instances 20
"""
from __future__ import annotations
import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.rl.config import DEVICE
from src.rl.model import TransformerClusterer
from src.rl.train_rl import rl_assign_argmax_safe
import src.rl.train_rl as _tr
from src.rl.reward import compute_packing_cost
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.adaptive_assign import rl_assign_argmax_adaptive

# ── Palette (matches eval/generate_plots.py's house style) ────────────────
SURFACE       = '#fcfcfb'
INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'
SERIES_COLOR  = '#e34948'
BAND_ALPHA    = 0.15

K_VALUES = [100, 500, 1000, 3000, 5000]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BASELINE)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def sweep(model, packer, data_dir, sample_tags, device, disable_heuristic=False):
    """
    Default mode uses rl_assign_argmax_adaptive -- the current production
    pipeline, which computes both {heuristic, model} candidates per
    (instance, K) and keeps whichever is actually cheaper, rather than the
    old fixed PRIORITY_CONSOLIDATION_MIN_K=500 threshold (verified: that
    fixed threshold picked the actually-cheaper option only ~50% of the
    time at K>=500, since the true crossover point is instance-specific).

    disable_heuristic : if True, forces the PURE model path at every K (no
        heuristic candidate considered at all, not even as a losing
        candidate) -- a DIAGNOSTIC-ONLY setting showing what the model
        itself has learned in isolation, distinct from both the adaptive
        production default and the older fixed-threshold behavior.
    """
    rows = []
    if disable_heuristic:
        orig_min_k = _tr.PRIORITY_CONSOLIDATION_MIN_K
        _tr.PRIORITY_CONSOLIDATION_MIN_K = max(K_VALUES) + 1
        try:
            for tag in sample_tags:
                ulds_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv'))
                pkgs_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv'))
                for k in K_VALUES:
                    asgn = rl_assign_argmax_safe(model, pkgs_df, ulds_df, device, k)
                    placements, _ = packer.pack(asgn, pkgs_df, ulds_df)
                    cost, delay, spread_cost, n_prio, up, ue = compute_packing_cost(placements, pkgs_df, k)
                    rows.append(dict(instance=tag, K=k, spread=n_prio, cost=cost,
                                      priority_dropped=len(up), economy_dropped=len(ue)))
        finally:
            _tr.PRIORITY_CONSOLIDATION_MIN_K = orig_min_k
    else:
        for tag in sample_tags:
            ulds_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv'))
            pkgs_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv'))
            for k in K_VALUES:
                _, placements, cost, _total_unfit, _chosen = rl_assign_argmax_adaptive(
                    model, pkgs_df, ulds_df, device, k, packer)
                _, delay, spread_cost, n_prio, up, ue = compute_packing_cost(placements, pkgs_df, k)
                rows.append(dict(instance=tag, K=k, spread=n_prio, cost=cost,
                                  priority_dropped=len(up), economy_dropped=len(ue)))
    return pd.DataFrame(rows)


def plot_spread_vs_k(df, out_path, title, color=SERIES_COLOR, subtitle_extra=''):
    means = df.groupby('K')['spread'].mean().reindex(K_VALUES)
    stds  = df.groupby('K')['spread'].std().reindex(K_VALUES)

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style_axes(ax)

    x = np.arange(len(K_VALUES))
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)
    ax.plot(x, means.values, color=color, linewidth=2, marker='o', markersize=8,
            solid_capstyle='round', zorder=3)
    for xi, yi in zip(x, means.values):
        ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=9, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel('K (spread-cost coefficient)')
    ax.set_ylabel('Mean priority spread (n ULDs)')
    n = df['instance'].nunique()
    fig.suptitle(title, color=INK_PRIMARY, fontsize=12, x=0.02, y=0.98, ha='left')
    ax.set_title(f'Same-instance K-sweep, n={n} instances, band = +/-1 std{subtitle_extra}',
                 color=INK_MUTED, fontsize=9, loc='left', pad=12)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/rl_ppo_contrastive_v7/transformer_rl_ppo_contrastive.pt')
    ap.add_argument('--data-root', default=os.path.expanduser('~/Desktop/good_data'))
    ap.add_argument('--n-instances', type=int, default=20)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='results/contrastive_causal_k_sensitivity.png')
    ap.add_argument('--disable-heuristic', action='store_true',
                     help='Diagnostic only: bypass _consolidate_priority_by_capacity at every K, '
                          'showing the model\'s own raw predictions instead of the production pipeline.')
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model = TransformerClusterer().to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    packer = RLPackerAdapter()

    train_meta = pd.read_csv(os.path.join(args.data_root, 'synthetic_train', 'metadata.csv'))
    random.seed(args.seed)
    sample_tags = random.sample(list(train_meta['instance']), args.n_instances)

    df = sweep(model, packer, args.data_root, sample_tags, DEVICE, disable_heuristic=args.disable_heuristic)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace('.png', '_data.csv'), index=False)

    total_dropped = df['priority_dropped'].sum()
    print(f'Priority packages dropped across the whole sweep: {total_dropped} (should be 0)')
    assert total_dropped == 0, 'contrastive_v7 dropped priority packages during the K-sweep -- investigate before trusting the plot'

    if args.disable_heuristic:
        plot_spread_vs_k(df, args.out,
                          'Model\'s OWN K-response, heuristic disabled (contrastive_v7, diagnostic)',
                          color='#2a78d6', subtitle_extra=' -- NOT the production pipeline')
    else:
        plot_spread_vs_k(df, args.out,
                          'Priority spread vs K -- adaptive selection (contrastive_v7)')


if __name__ == '__main__':
    main()
