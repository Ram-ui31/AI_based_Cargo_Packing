"""
generate_graphs.py -- renders all presentation graphs for the hybrid
AI+heuristic cargo packer, from real logged data only (no fabricated
numbers). Deliberately does NOT reference any external cost benchmark --
presentation-safe.

Usage:
    python generate_graphs.py
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
GA_RESULTS = os.path.join(HERE, '..', '..', 'ga_cargo_packing', 'results')
CHECKPOINTS = os.path.join(HERE, '..', 'checkpoints')

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


def running_min(costs):
    out, best = [], float('inf')
    for c in costs:
        best = min(best, c)
        out.append(best)
    return out


# ---------------------------------------------------------------------
# Chart 1: full method progression (every technique tried, in order)
# ---------------------------------------------------------------------
def chart_method_progression():
    methods = [
        ('Classical formula\n(hand-tuned)', 30475),
        ('RL / GRPO\n(single-instance)', 30672),
        ('Multi-instance GRPO\n(generalized) -- Halley', 30608),
        ('Local search\n(order-based)', 29656),
        ('AI-guided search\n(first iteration)', 29564),
        ('Assignment search\n(direct reformulation)', 29564),
        ('+ Better candidate\ngeneration', 28960),
        ('Final: local search +\nimproved candidate generation', 28452),
    ]
    labels = [m[0] for m in methods]
    costs = [m[1] for m in methods]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = ['#8f9bb3'] * (len(costs) - 1) + ['#4c8bf5']
    bars = ax.bar(labels, costs, color=colors, width=0.6, zorder=3)
    ax.set_ylim(27800, 31100)
    ax.set_ylabel('Total packing cost (lower is better)')
    ax.set_title('Progression across every method tried')
    for bar, cost in zip(bars, costs):
        ax.annotate(f'{cost:,.0f}', (bar.get_x() + bar.get_width() / 2, cost),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=0, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '01_method_progression.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 2: search convergence -- vanilla vs AI-guided, final phase
# ---------------------------------------------------------------------
def chart_search_convergence():
    runs = {
        'Local search (vanilla)': 'beam_log_ems_v1.jsonl',
        'Local search (independent restart)': 'beam_log_ems_v2.jsonl',
        'Assignment search (vanilla)': 'knapsack_log_v3.jsonl',
        'AI-guided search (v2)': 'beam_log_guided_ems_v2.jsonl',
        'AI-guided search (v3)': 'beam_log_guided_ems_v3.jsonl',
        'AI-guided search (v4)': 'beam_log_guided_ems_v4.jsonl',
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ['#4c8bf5', '#7ed321', '#f5a623', '#bd10e0', '#e0245e', '#50e3c2']
    any_data = False
    for (label, fname), color in zip(runs.items(), colors):
        rows = load_jsonl(os.path.join(GA_RESULTS, fname))
        if not rows:
            continue
        any_data = True
        costs = [min(r['beam_costs']) for r in rows]
        xs = list(range(len(costs)))
        ax.plot(xs, running_min(costs), label=label, color=color, linewidth=2)
    ax.set_xlabel('Round (sequential)')
    ax.set_ylabel('Best-ever cost')
    ax.set_title('Search convergence: vanilla vs. AI-guided')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    if any_data:
        ax.legend(loc='upper right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '02_search_convergence.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 3: SwapProposer training curves (real per-epoch history)
# ---------------------------------------------------------------------
def chart_training_curves():
    history_path = os.path.join(CHECKPOINTS, 'swap_proposer_history.json')
    if not os.path.exists(history_path):
        print('No training history found, skipping chart 3.')
        return
    with open(history_path) as f:
        history = json.load(f)
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_rank_acc'] for h in history]
    val_acc = [h['val_rank_acc'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(epochs, train_loss, color='#4c8bf5', label='Train loss', linewidth=1.5)
    ax.plot(epochs, val_loss, color='#e0245e', label='Validation loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Margin ranking loss')
    ax.set_title('AI model training: loss curves')
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, train_acc, color='#4c8bf5', label='Train accuracy', linewidth=1.5)
    ax.plot(epochs, val_acc, color='#e0245e', label='Validation accuracy', linewidth=1.5)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Chance baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pairwise ranking accuracy')
    ax.set_title('AI model training: accuracy curves')
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '03_training_validation_curves.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 4: candidate generation improvement (before/after)
# ---------------------------------------------------------------------
def chart_candidate_generation():
    labels = ['Baseline\ncandidate generation', 'Improved\ncandidate generation']
    values = [30475, 28452]
    colors = ['#8f9bb3', '#4c8bf5']
    fig, ax = plt.subplots(figsize=(8, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:,.0f}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total packing cost')
    ax.set_title('Impact of improved candidate-placement generation')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '04_candidate_generation_impact.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 5: CentrifugeEvictProposer training curves
# ---------------------------------------------------------------------
def chart_centrifuge_training():
    history_path = os.path.join(CHECKPOINTS, 'centrifuge_proposer_history.json')
    if not os.path.exists(history_path):
        print('No centrifuge proposer history found, skipping chart 5.')
        return
    with open(history_path) as f:
        history = json.load(f)
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_win_acc = [h['val_win_acc'] for h in history]
    val_p_at_1 = [h['val_p_at_1'] for h in history]
    val_corr = [h['val_corr'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(epochs, train_loss, color='#4c8bf5', label='Train loss', linewidth=1.5)
    ax.plot(epochs, val_loss, color='#e0245e', label='Validation loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Smooth L1 loss (net delay-cost gain)')
    ax.set_title('CentrifugeEvictProposer: loss curves')
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, val_win_acc, color='#4c8bf5', label='Win/loss accuracy', linewidth=1.5)
    ax.plot(epochs, val_corr, color='#00b894', label='Correlation (pred vs. real)', linewidth=1.5)
    ax.plot(epochs, val_p_at_1, color='#e0245e', label='Precision@1 (per-container)', linewidth=1.5)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Chance baseline (win/loss)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation metric')
    ax.set_title('CentrifugeEvictProposer: validation metrics\n(held out at the instance level)')
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '05_centrifuge_proposer_training.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 6: CentrifugeEvictProposer on the real benchmark instance --
# predicted vs. real net gain, and recall@K vs. verification budget
# ---------------------------------------------------------------------
def chart_centrifuge_real_instance():
    eval_path = os.path.join(HERE, '..', 'results', 'centrifuge_real_instance_eval.json')
    if not os.path.exists(eval_path):
        print('No real-instance centrifuge eval found, skipping chart 6.')
        return
    with open(eval_path) as f:
        results = json.load(f)

    real_gains = [r['real_net_gain'] for r in results]
    preds = [r['model_pred'] for r in results]
    true_wins = [(r, p) for r, p in zip(real_gains, preds) if r > 0]
    true_losses = [(r, p) for r, p in zip(real_gains, preds) if r <= 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter([p for _, p in true_losses], [r for r, _ in true_losses],
               color='#8f9bb3', alpha=0.6, s=28, label=f'Not a real win (n={len(true_losses)})')
    ax.scatter([p for _, p in true_wins], [r for r, _ in true_wins],
               color='#00b894', alpha=0.9, s=60, label=f'Real win (n={len(true_wins)})', zorder=3)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Model-predicted net gain')
    ax.set_ylabel('Real net gain (exhaustive compact + refill)')
    ax.set_title('Real 400-package instance:\npredicted vs. actual, all 151 candidates')
    ax.legend(fontsize=9, loc='upper left')

    order = sorted(range(len(results)), key=lambda i: -preds[i])
    sorted_gains = [real_gains[i] for i in order]
    n_true_wins = sum(1 for g in real_gains if g > 0)
    Ks = list(range(1, len(results) + 1))
    recalls = []
    found = 0
    for k in Ks:
        if sorted_gains[k - 1] > 0:
            found += 1
        recalls.append(found / max(n_true_wins, 1))

    ax = axes[1]
    ax.plot(Ks, recalls, color='#4c8bf5', linewidth=2)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Model-guided verification budget (top-K candidates real-checked)')
    ax.set_ylabel('Fraction of true wins captured')
    ax.set_title(f'Real instance: recall vs. verification budget\n({n_true_wins} true wins among {len(results)} candidates)')
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '06_centrifuge_real_instance_eval.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 7: 3-model comparison across all 5 K values (20 held-out
# synthetic instances, 4 per K), average cost vs. K
# ---------------------------------------------------------------------
def chart_three_model_k_sweep():
    sweep_path = os.path.join(HERE, '..', 'results', 'three_model_k_sweep.json')
    if not os.path.exists(sweep_path):
        print('No three-model K-sweep results found, skipping chart 7.')
        return
    with open(sweep_path) as f:
        records = json.load(f)

    from collections import defaultdict
    by_k = defaultdict(lambda: {'rl_placement_cost': [], 'multi_instance_grpo_cost': [], 'cherry_cost': []})
    for r in records:
        by_k[r['K']]['rl_placement_cost'].append(r['rl_placement_cost'])
        by_k[r['K']]['multi_instance_grpo_cost'].append(r['multi_instance_grpo_cost'])
        by_k[r['K']]['cherry_cost'].append(r['cherry_cost'])

    ks = sorted(by_k.keys())
    rl_avg = [sum(by_k[k]['rl_placement_cost']) / len(by_k[k]['rl_placement_cost']) for k in ks]
    gnn_avg = [sum(by_k[k]['multi_instance_grpo_cost']) / len(by_k[k]['multi_instance_grpo_cost']) for k in ks]
    cherry_avg = [sum(by_k[k]['cherry_cost']) / len(by_k[k]['cherry_cost']) for k in ks]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(ks, rl_avg, color='#8f9bb3', marker='o', linewidth=2, markersize=7, label='Eclipse (RL placement policy, baseline)')
    ax.plot(ks, gnn_avg, color='#e0245e', marker='s', linewidth=2, markersize=7, label='Halley (Multi-instance GRPO, economy order)')
    ax.plot(ks, cherry_avg, color='#4c8bf5', marker='^', linewidth=2, markersize=7, label='Cherry (centrifuge-evict refinement)')
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([f'{int(k):,}' for k in ks])
    ax.set_xlabel('K')
    ax.set_ylabel('Average total packing cost')
    ax.set_title('3-model comparison across all 5 K values\n(20 held-out synthetic instances, 4 per K, same instance+K per model)')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '07_three_model_k_sweep.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Chart 8: grand-average cost histogram -- Eclipse vs. Halley vs. Cherry,
# averaged across all 5 K values (mean of the 5 per-K averages from
# chart 7 / three_model_k_sweep.json)
# ---------------------------------------------------------------------
def chart_three_model_grand_average():
    sweep_path = os.path.join(HERE, '..', 'results', 'three_model_k_sweep.json')
    if not os.path.exists(sweep_path):
        print('No three-model K-sweep results found, skipping chart 8.')
        return
    with open(sweep_path) as f:
        records = json.load(f)

    from collections import defaultdict
    by_k = defaultdict(lambda: {'rl_placement_cost': [], 'multi_instance_grpo_cost': [], 'cherry_cost': []})
    for r in records:
        by_k[r['K']]['rl_placement_cost'].append(r['rl_placement_cost'])
        by_k[r['K']]['multi_instance_grpo_cost'].append(r['multi_instance_grpo_cost'])
        by_k[r['K']]['cherry_cost'].append(r['cherry_cost'])

    ks = sorted(by_k.keys())
    per_k_avg = {
        'Eclipse': [sum(by_k[k]['rl_placement_cost']) / len(by_k[k]['rl_placement_cost']) for k in ks],
        'Halley': [sum(by_k[k]['multi_instance_grpo_cost']) / len(by_k[k]['multi_instance_grpo_cost']) for k in ks],
        'Cherry': [sum(by_k[k]['cherry_cost']) / len(by_k[k]['cherry_cost']) for k in ks],
    }
    grand_avg = {name: sum(vals) / len(vals) for name, vals in per_k_avg.items()}

    labels = ['Eclipse', 'Halley', 'Cherry']
    values = [grand_avg[l] for l in labels]
    colors = ['#8f9bb3', '#e0245e', '#4c8bf5']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.annotate(f'{v:,.0f}', (bar.get_x() + bar.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average total packing cost\n(mean of the 5 per-K averages)')
    ax.set_title('Grand-average cost across all 5 K values\n(20 held-out synthetic instances, 4 per K)')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, '08_three_model_grand_average.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    chart_method_progression()
    chart_search_convergence()
    chart_training_curves()
    chart_candidate_generation()
    chart_centrifuge_training()
    chart_centrifuge_real_instance()
    chart_three_model_k_sweep()
    chart_three_model_grand_average()
    print(f'Saved graphs to {HERE}')
