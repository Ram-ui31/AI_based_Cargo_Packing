"""
plot_adaptive_method_choice.py -- shows how often rl_assign_argmax_adaptive
picks the heuristic's Priority-consolidation answer vs the model's own, per
K value, across a sample of instances.

Why this graph: the whole point of adaptive selection (src/rl/
adaptive_assign.py) is that there is NO fixed crossover K where one side
always wins -- the old PRIORITY_CONSOLIDATION_MIN_K=500 threshold picked the
actually-cheaper option only ~50% of the time at K>=500 when checked
directly. This plot makes that visible: if the split were a clean step
function (all heuristic above some K, all model below), that would suggest
a threshold COULD have worked, just not at 500. If it's mixed at every K
(the actual finding), that's direct visual evidence a fixed threshold could
never have worked at any single value.

Usage:
    python scripts/plot_adaptive_method_choice.py \
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
from src.rl.rl_packer_adapter import RLPackerAdapter
from src.rl.adaptive_assign import rl_assign_argmax_adaptive

# ── Palette (matches eval/generate_plots.py's house style) ────────────────
SURFACE       = '#fcfcfb'
INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'
COLOR = {'heuristic': '#2a78d6', 'model': '#e34948'}  # fixed categorical order, never cycled

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


def sweep_choices(model, packer, data_dir, sample_tags, device):
    rows = []
    for tag in sample_tags:
        ulds_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv'))
        pkgs_df = pd.read_csv(os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv'))
        for k in K_VALUES:
            _, _, cost, _, chosen = rl_assign_argmax_adaptive(model, pkgs_df, ulds_df, device, k, packer)
            rows.append(dict(instance=tag, K=k, chosen=chosen, cost=cost))
    return pd.DataFrame(rows)


def plot_method_choice(df, out_path):
    n = df['instance'].nunique()
    counts = df.groupby(['K', 'chosen']).size().unstack(fill_value=0).reindex(K_VALUES)
    for label in ('heuristic', 'model'):
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[['heuristic', 'model']]
    fractions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style_axes(ax)

    x = np.arange(len(K_VALUES))
    width = 0.34
    for i, label in enumerate(('heuristic', 'model')):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, counts[label], width=width, color=COLOR[label],
                       label=label, zorder=3)
        for xi, bar, cnt, frac in zip(x, bars, counts[label], fractions[label]):
            if cnt > 0:
                ax.annotate(f'{int(cnt)}\n({frac:.0%})', (xi + offset, bar.get_height()),
                            textcoords='offset points', xytext=(0, 4),
                            ha='center', fontsize=8.5, color=INK_PRIMARY, linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel('K (spread-cost coefficient)')
    ax.set_ylabel(f'Instances choosing this method (of {n})')
    ax.set_ylim(0, n * 1.22)
    ax.legend(frameon=False, loc='upper right', labelcolor=INK_SECONDARY, fontsize=9)

    fig.suptitle('Which candidate does adaptive selection actually pick?', color=INK_PRIMARY,
                 fontsize=12, x=0.02, y=0.98, ha='left')
    ax.set_title(f'Same-instance K-sweep, n={n} instances -- mixed at every K, not a clean step function',
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
    ap.add_argument('--out', default='results/adaptive_method_choice.png')
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

    df = sweep_choices(model, packer, args.data_root, sample_tags, DEVICE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace('.png', '_data.csv'), index=False)

    print(df.groupby(['K', 'chosen']).size().unstack(fill_value=0))
    plot_method_choice(df, args.out)


if __name__ == '__main__':
    main()
