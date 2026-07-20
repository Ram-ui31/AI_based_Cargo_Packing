"""
Comparison graphs: three independent classical heuristics from the
literature review (FFD, LAFF, BFD) vs. the external RL baseline
(Online-3D-BPP-DRL) vs. our best result (Cherry). Eclipse and Halley are
intentionally excluded -- this comparison is about classical-heuristic vs.
external-RL vs. our-best, not a rehash of the internal 3-model comparison.

Two charts, matching this project's established visual style (matplotlib,
white background, bold value labels, recessive gridlines):
  1. Real 400-package instance cost.
  2. Grand-average cost across the same 20 held-out synthetic instances.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'graphs'), exist_ok=True)

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                      'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# Fixed categorical order + colors (consistent hue per method across both
# charts, never reassigned by rank/value).
LABELS = ['FFD\n(literature)', 'LAFF\n(literature)', 'BFD\n(literature)',
          'Online-3D-BPP-DRL\n(external RL)', 'Cherry\n(ours, best)']
COLORS = ['#c9a86a', '#b98b52', '#a66f3b', '#8C8C8C', '#4c8bf5']

with open(os.path.join(HERE, 'results', 'summary.json')) as f:
    summary = json.load(f)

real_costs = [
    summary['ffd']['real_instance']['total_cost'],
    summary['laff']['real_instance']['total_cost'],
    summary['bfd']['real_instance']['total_cost'],
    35676,   # Online-3D-BPP-DRL, ~/Desktop/online-3d-bpp-benchmark/benchmark_result.json
    28409,   # Cherry, final result after centrifuge-evict refinement
]

grand_avgs = [
    summary['ffd']['sweep_20_instance']['grand_avg'],
    summary['laff']['sweep_20_instance']['grand_avg'],
    summary['bfd']['sweep_20_instance']['grand_avg'],
    15535.15,  # Online-3D-BPP-DRL, ~/Desktop/online-3d-bpp-benchmark/online_bpp_20instance_results.json
    9498.8,    # Cherry, cargoism/git/01-cherry/results/three_model_k_sweep.json
]


def bar_chart(values, title, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bars = ax.bar(LABELS, values, color=COLORS, width=0.6, zorder=3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:,.0f}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 8), ha='center',
                    fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylim(0, max(values) * 1.15)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'graphs', out_name), dpi=150)
    plt.close(fig)
    print(f'saved graphs/{out_name}')


bar_chart(real_costs,
          'Real 400-package instance: classical heuristics vs. RL baseline vs. Cherry',
          'Total packing cost', '01_real_instance_comparison.png')

bar_chart(grand_avgs,
          'Grand-average cost across 20 held-out synthetic instances\n(same instances, same K sweep, for all 5 methods)',
          'Average total packing cost\n(mean of the 5 per-K averages)', '02_grand_average_comparison.png')

# Individual per-model versions (grand-average only), swapping Cherry's slot
# for Eclipse / Halley respectively -- used in each model's own README.
OTHER_MODELS = {
    'eclipse': ('Eclipse\n(RL placement policy)', '#8f9bb3', 10631.1, '03_eclipse_grand_average_comparison.png'),
    'halley': ('Halley\n(GRPO economy order)', '#e0245e', 10540.0, '04_halley_grand_average_comparison.png'),
}

for key, (label, color, value, out_name) in OTHER_MODELS.items():
    labels_bak, colors_bak = LABELS[-1], COLORS[-1]
    LABELS[-1], COLORS[-1] = label, color
    values = grand_avgs[:-1] + [value]
    bar_chart(values,
              f'Grand-average cost across 20 held-out synthetic instances\n(same instances, same K sweep, classical heuristics vs. RL baseline vs. {key.capitalize()})',
              'Average total packing cost\n(mean of the 5 per-K averages)', out_name)
    LABELS[-1], COLORS[-1] = labels_bak, colors_bak

print('\nDone.')
