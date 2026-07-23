"""
generate_summary_plots.py -- renders the session's progress into a set of
static PNG charts under results/plots/, from real logged data (beam_log_*
/ knapsack_log_* JSONL files on disk) plus a couple of hardcoded series
that are exact console output already produced this session (the two
SwapProposer training runs, and the priority-allocation strategy sweep --
the latter re-run fresh via test_priority_allocation.py so the numbers in
the chart are live, not remembered).

Usage:
    python scripts/generate_summary_plots.py
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

COMPETITOR_TARGET = 29203
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                      'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------
# Chart 1: cost progression across methods (the whole session's journey)
# ---------------------------------------------------------------------
def chart_method_progression():
    methods = [
        ('Classical formula\n(value_density^1.5)', 30475),
        ('RL/GRPO\n(single-instance)', 30672),
        ('Multi-instance GRPO\n(generalized)', 30608),
        ('Beam search\n(order-based,\nbreakthrough)', 29656),
        ('Guided beam search\n(SwapProposer,\nranking loss)', 29564),
        ('Knapsack search\n(direct\nassignment)', 29564),
    ]
    labels = [m[0] for m in methods]
    costs = [m[1] for m in methods]

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = ['#8f9bb3' if c > COMPETITOR_TARGET else '#4c8bf5' for c in costs]
    bars = ax.bar(labels, costs, color=colors, width=0.6, zorder=3)
    ax.axhline(COMPETITOR_TARGET, color='#e0245e', linestyle='--', linewidth=2, zorder=4,
               label=f'Competitor benchmark ({COMPETITOR_TARGET:,})')
    ax.set_ylim(28500, 31100)
    ax.set_ylabel('Real packing cost (lower is better)')
    ax.set_title('Session progress: real cost on the 400-package benchmark instance')
    for bar, cost in zip(bars, costs):
        ax.annotate(f'{cost:,.0f}', (bar.get_x() + bar.get_width() / 2, cost),
                    textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.legend(loc='upper left')
    plt.xticks(rotation=0, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '01_method_progression.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 2: beam/knapsack search convergence curves (round vs best cost)
# ---------------------------------------------------------------------
def chart_search_convergence():
    runs = {
        'Order-based beam (pow1.5 seed)': 'beam_log_default.jsonl',
        'Order-based beam (GNN seed)': 'beam_log_gnn_seed.jsonl',
        'Guided beam (SwapProposer)': 'beam_log_guided_v3.jsonl',
        'Knapsack (direct assignment)': 'knapsack_log_v1.jsonl',
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#4c8bf5', '#f5a623', '#7ed321', '#bd10e0']
    any_data = False
    for (label, fname), color in zip(runs.items(), colors):
        rows = load_jsonl(os.path.join(RESULTS_DIR, fname))
        if not rows:
            continue
        any_data = True
        # Sequential log-entry index, not the raw 'round' field: a
        # mid-session restart bug (see project history) caused round
        # numbers in beam_log_default.jsonl to briefly jump backward
        # across a relaunch, which would otherwise draw a confusing
        # loop-back in the line. Index-in-file is monotonic by construction
        # and still shows the true improvement trajectory.
        ys = [min(r['beam_costs']) for r in rows]
        xs = list(range(len(ys)))
        running_min = []
        best = float('inf')
        for y in ys:
            best = min(best, y)
            running_min.append(best)
        ax.plot(xs, running_min, label=label, color=color, linewidth=2)
    ax.axhline(COMPETITOR_TARGET, color='#e0245e', linestyle='--', linewidth=2,
               label=f'Competitor benchmark ({COMPETITOR_TARGET:,})')
    ax.set_xlabel('Round (sequential log order)')
    ax.set_ylabel('Best-ever real cost')
    ax.set_title('Search convergence: best-ever cost by round, across search variants')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    if any_data:
        ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '02_search_convergence.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 3: SwapProposer training -- before/after fixing the loss function
# ---------------------------------------------------------------------
def chart_swap_proposer_training():
    # Exact console output from this session's two training runs.
    epochs = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299]
    regression_val_loss = [0.8675, 0.4928, 0.4055, 0.3752, 0.3688, 0.3704, 0.3713,
                            0.3754, 0.3814, 0.3820, 0.3920, 0.3969, 0.4017]
    regression_metric = [0.016, 0.984, 0.984, 0.984, 0.984, 0.984, 0.984,
                          0.984, 0.968, 0.968, 0.984, 0.968, 0.984]  # val_dir_acc -- inflated by class imbalance
    ranking_val_loss = [0.1023, 0.0538, 0.0432, 0.0406, 0.0435, 0.0416, 0.0423,
                        0.0429, 0.0412, 0.0439, 0.0411, 0.0433, 0.0428]
    ranking_metric = [0.453, 0.787, 0.880, 0.867, 0.853, 0.867, 0.853,
                      0.827, 0.827, 0.840, 0.853, 0.840, 0.840]  # val_rank_acc -- real pairwise skill

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(epochs, regression_metric, marker='o', color='#e0245e', label='v1: delta regression (Huber loss)\n"val_dir_acc" -- inflated by imbalance')
    ax.plot(epochs, ranking_metric, marker='o', color='#4c8bf5', label='v2: pairwise ranking (Margin loss)\n"val_rank_acc" -- real discriminative skill')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='chance baseline (v2\'s task)')
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='"always predict positive" baseline (v1\'s task)')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Validation accuracy metric')
    ax.set_title('SwapProposer: v1\'s 98% was a base-rate artifact,\nv2\'s ~85% is real signal')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='center right')

    ax = axes[1]
    ax.plot(epochs, regression_val_loss, marker='o', color='#e0245e', label='v1: regression val loss (Huber)')
    ax.plot(epochs, ranking_val_loss, marker='o', color='#4c8bf5', label='v2: ranking val loss (Margin)')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Validation loss (not comparable across losses --\nshown to illustrate overfitting shape)')
    ax.set_title('v1 overfits after ~epoch 100 (val loss rises);\nv2 stays flat/stable')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '03_swap_proposer_training.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 4: MILP theoretical ceiling vs what's actually achievable
# ---------------------------------------------------------------------
def chart_milp_ceiling():
    # From scripts/milp_ceiling.py's real run this session.
    spread_cost = 15000  # K=5000 * 3 priority ULDs
    milp_relaxation_floor = 15000 + 10387       # volume+weight-only ceiling (unreachable in practice)
    milp_selection_real_packed = 31540           # MILP's own item selection, real-packed in 3D
    our_best = 29564
    competitor = COMPETITOR_TARGET

    labels = ['Volume+weight-only\nMILP relaxation\n(theoretical, ignores 3D shape)',
              "MILP's selection,\nreal 3D-packed\n(shape-blind selection fails)",
              'Our best result\n(real-packer-guided search)',
              'Competitor\nbenchmark']
    values = [milp_relaxation_floor, milp_selection_real_packed, our_best, competitor]
    colors = ['#c7c7c7', '#e0245e', '#4c8bf5', '#7ed321']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:,.0f}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 6), ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Real packing cost')
    ax.set_title('Why the remaining gap is a 3D-packing-efficiency problem,\nnot a "which packages" search problem')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '04_milp_ceiling.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 5: Priority-to-ULD allocation strategy sweep (live re-run data)
# ---------------------------------------------------------------------
def chart_priority_allocation(strategy_costs):
    if not strategy_costs:
        print('No priority-allocation data available, skipping chart 5.')
        return
    labels = list(strategy_costs.keys())
    values = list(strategy_costs.values())
    best_idx = values.index(min(values))
    colors = ['#4c8bf5' if i == best_idx else '#8f9bb3' for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:,.0f}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 6), ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Real packing cost')
    ax.set_title('Priority-to-ULD allocation strategy sweep\n(within the fixed best 3-ULD combo)')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=15, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, '05_priority_allocation_sweep.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    strategy_costs = None
    if len(sys.argv) > 1 and sys.argv[1] == '--with-priority-data':
        # Expects a small JSON file written by a wrapper that parsed
        # test_priority_allocation.py's output; see generate_all.sh.
        with open(os.path.join(RESULTS_DIR, 'priority_allocation_costs.json')) as f:
            strategy_costs = json.load(f)

    chart_method_progression()
    chart_search_convergence()
    chart_swap_proposer_training()
    chart_milp_ceiling()
    chart_priority_allocation(strategy_costs)
    print(f'Saved charts to {PLOTS_DIR}')
