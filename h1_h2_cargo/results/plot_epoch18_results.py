"""
plot_epoch18_results.py — renders the h1_h2 vs IL vs RL (epoch 18) comparison
as two static PNGs, from the evaluation data already computed by
rl_fineuning_over_il/eval/evaluate_h1h2.py (eval_h1h2_vs_il_vs_rl.csv).

Usage:
    python results/plot_epoch18_results.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_THIS_DIR, '..', '..', 'rl_fineuning_over_il', 'eval')

COLORS = {'h1_h2': '#2a78d6', 'IL': '#1baf7a', 'RL': '#eda100'}
LABELS = {'h1_h2': 'h1_h2 heuristic', 'IL': 'IL (imitation)', 'RL': 'RL (epoch 18, fine-tuned)'}
METHODS = ['h1_h2', 'IL', 'RL']
K_VALUES = [100, 500, 1000, 3000, 5000]


def plot_cost_vs_k(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    x = np.arange(len(K_VALUES))
    width = 0.26

    for i, m in enumerate(METHODS):
        means, stds = [], []
        for k in K_VALUES:
            sub = df[(df.K == k) & (df.method == m)]
            means.append(sub.cost.mean())
            stds.append(sub.cost.std())
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                       label=LABELS[m], color=COLORS[m], edgecolor='white', linewidth=0.6)
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in K_VALUES])
    ax.set_ylabel('Mean total cost per instance')
    ax.set_title('Cost vs K — h1_h2 heuristic vs IL vs RL (epoch 18)\nAll priority packages shipped, weight/volume/overlap constraints verified', fontsize=11)
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#e1e0d9', zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Saved -> {out_path}')


def plot_cost_histogram(df, out_path):
    all_costs = df['cost'].values
    edges = np.linspace(all_costs.min(), all_costs.max(), 17)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), dpi=150, sharex=True)
    for ax, m in zip(axes, METHODS):
        vals = df[df.method == m]['cost'].values
        ax.hist(vals, bins=edges, color=COLORS[m], edgecolor='white', linewidth=0.6)
        ax.set_ylabel(LABELS[m], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#e1e0d9', zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(vals.mean(), color='#333333', linestyle='--', linewidth=1)
        ax.text(vals.mean(), ax.get_ylim()[1] * 0.9, f' mean={vals.mean():.0f}', fontsize=8, color='#333333')

    axes[-1].set_xlabel('Total cost per instance')
    axes[0].set_title('Distribution of per-instance cost — h1_h2 vs IL vs RL (epoch 18)\nSame bins/x-scale across all three, 100 test instances each', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Saved -> {out_path}')


def main():
    df = pd.read_csv(os.path.join(_EVAL_DIR, 'eval_h1h2_vs_il_vs_rl.csv'))
    plot_cost_vs_k(df, os.path.join(_THIS_DIR, 'cost_vs_k_epoch18.png'))
    plot_cost_histogram(df, os.path.join(_THIS_DIR, 'cost_histogram_epoch18.png'))


if __name__ == '__main__':
    main()
