"""
generate_plots.py — three comparison plots from results/comparison_ga_il_rl.csv
(see generate_comparison_data.py):
    1. cost_vs_k.png       — mean cost per K, GA vs RL
    2. cost_histogram.png  — cost distribution, GA vs IL vs RL
    3. spread_vs_k.png     — mean spread per K, GA vs RL

Palette and mark choices follow the project's dataviz skill: fixed categorical
hue order (never cycled), one axis, thin marks, legend always present for
>=2 series, recessive gridlines, direct labels on bars instead of relying on
color alone.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Palette (light mode, static PNG) ──────────────────────────────────────
SURFACE       = '#fcfcfb'
INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'

COLOR = {'GA': '#2a78d6', 'IL': '#4a3aa7', 'RL': '#e34948'}  # fixed categorical order, never cycled
K_ORDER = [100, 500, 1000, 3000, 5000]


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


def plot_cost_vs_k(df, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style_axes(ax)

    means = df[df['method'].isin(['GA', 'RL'])].groupby(['K', 'method'])['cost'].mean().unstack()
    means = means.reindex(K_ORDER)

    x = np.arange(len(K_ORDER))
    width = 0.34
    for i, method in enumerate(['GA', 'RL']):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means[method], width=width, color=COLOR[method],
                      label=method, zorder=3, edgecolor=SURFACE, linewidth=0.5)
        for rect, val in zip(bars, means[method]):
            ax.annotate(f'{val:,.0f}', (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                       textcoords='offset points', xytext=(0, 3), ha='center',
                       fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in K_ORDER])
    ax.set_ylabel('Mean cost  (K × spread + delay cost)')
    ax.set_title('Cost vs K — GA labels vs RL fine-tuned', color=INK_PRIMARY,
                 fontsize=13, fontweight='bold', loc='left', pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    legend = ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_cost_histogram(df, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style_axes(ax)

    all_costs = df['cost']
    bins = np.linspace(all_costs.min(), all_costs.max(), 22)

    for method in ['GA', 'IL', 'RL']:
        vals = df[df['method'] == method]['cost']
        ax.hist(vals, bins=bins, color=COLOR[method], alpha=0.5, label=method,
               edgecolor=COLOR[method], linewidth=1.2, zorder=3)

    ax.set_xlabel('Cost  (K × spread + delay cost)')
    ax.set_ylabel('Number of test instances')
    ax.set_title('Cost distribution — GA vs IL vs RL', color=INK_PRIMARY,
                 fontsize=13, fontweight='bold', loc='left', pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.legend(frameon=False, loc='upper right', fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_spread_vs_k(df, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style_axes(ax)

    means = df[df['method'].isin(['GA', 'RL'])].groupby(['K', 'method'])['spread'].mean().unstack()
    means = means.reindex(K_ORDER)

    x = np.arange(len(K_ORDER))
    width = 0.34
    for i, method in enumerate(['GA', 'RL']):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means[method], width=width, color=COLOR[method],
                      label=method, zorder=3, edgecolor=SURFACE, linewidth=0.5)
        for rect, val in zip(bars, means[method]):
            ax.annotate(f'{val:.2f}', (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                       textcoords='offset points', xytext=(0, 3), ha='center',
                       fontsize=8, color=INK_SECONDARY)

    # Spread values sit in a narrow band (~3.0-3.7) -- zero-based bars stay
    # correct (never truncate a bar axis, that exaggerates the differences),
    # but extra headroom above the tallest bar is needed so the legend box
    # doesn't collide with the K=100 group's value labels.
    ax.set_ylim(0, means.to_numpy().max() * 1.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in K_ORDER])
    ax.set_ylabel('Mean spread  (# ULDs holding Priority)')
    ax.set_title('Spread vs K — GA labels vs RL fine-tuned', color=INK_PRIMARY,
                 fontsize=13, fontweight='bold', loc='left', pad=14)
    ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='results/comparison_ga_il_rl.csv')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    plot_cost_vs_k(df, os.path.join(out_dir, 'cost_vs_k.png'))
    plot_cost_histogram(df, os.path.join(out_dir, 'cost_histogram.png'))
    plot_spread_vs_k(df, os.path.join(out_dir, 'spread_vs_k.png'))


if __name__ == '__main__':
    main()
