"""
plot_sweep.py -- bar chart of LOSO-AUC per config from sweep.py, grouped
by data subset (formula-only vs all-scenes-including-random).

Usage:
    python scripts/plot_sweep.py
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SWEEP_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep_results.json')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sweep_results.png')


def main():
    with open(SWEEP_JSON) as f:
        results = json.load(f)

    names = [r['name'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = ['#4C72B0' if r['data'] == 'formula' else '#C44E52' for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, aucs, color=colors)
    ax.set_ylabel('LOSO-AUC (held-out placement prediction quality)')
    ax.set_title('PackageSetRanker sweep: leave-one-scene-out cross-validation')
    ax.set_ylim(0.5, max(aucs) + 0.05)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='random guessing')
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30, ha='right')

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#4C72B0', label='formula-only scenes (15)'),
        Patch(facecolor='#C44E52', label='all scenes incl. random (25)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved plot to {OUT_PNG}')


if __name__ == '__main__':
    main()
