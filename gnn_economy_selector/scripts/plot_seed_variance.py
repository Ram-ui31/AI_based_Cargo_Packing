"""
plot_seed_variance.py -- shows real-cost spread across 4 random seeds of
the IDENTICAL config-B architecture, next to the spread across DIFFERENT
architectures (B/C/D) -- the point: is the seed-to-seed noise floor as
large as the differences we were attributing to architecture choice?

Usage:
    python scripts/plot_seed_variance.py
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'seed_variance.png')

SEED_RESULTS = {
    'B (unseeded, original)': 30937,
    'B seed=0': 31345,
    'B seed=1': 31120,
    'B seed=2': 31686,
}
ARCH_RESULTS = {
    'B (best)': 30937,
    'C': 31118,
    'D': 31798,
}
CURRENT_BEST = 30475
COMPETITOR_TARGET = 29203


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    ax = axes[0]
    names = list(SEED_RESULTS.keys())
    vals = list(SEED_RESULTS.values())
    ax.scatter(range(len(vals)), vals, s=100, color='#55A868', zorder=3)
    ax.plot(range(len(vals)), vals, color='#55A868', alpha=0.3, linestyle=':')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_title(f'Config B across 4 seeds\n(same architecture)\n'
                 f'range={max(vals)-min(vals):,}, std={np.std(vals):,.0f}')
    ax.set_ylabel('Real cost (lower is better)')

    ax = axes[1]
    names2 = list(ARCH_RESULTS.keys())
    vals2 = list(ARCH_RESULTS.values())
    ax.scatter(range(len(vals2)), vals2, s=100, color='#4C72B0', zorder=3)
    ax.plot(range(len(vals2)), vals2, color='#4C72B0', alpha=0.3, linestyle=':')
    ax.set_xticks(range(len(names2)))
    ax.set_xticklabels(names2, rotation=20, ha='right')
    ax.set_title(f'Different architectures\n(single seed each)\n'
                 f'range={max(vals2)-min(vals2):,}')

    for ax in axes:
        ax.axhline(CURRENT_BEST, color='black', linestyle='-', linewidth=1.2, label=f'Current best: {CURRENT_BEST:,}')
        ax.axhline(COMPETITOR_TARGET, color='red', linestyle='--', linewidth=1.2, label=f'Competitor: {COMPETITOR_TARGET:,}')
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Is architecture choice signal, or noise?')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved plot to {OUT_PNG}')


if __name__ == '__main__':
    main()
