"""
plot_full_comparison.py -- bar chart of every real-validated cost tried
across the whole session (classical selection formulas, ILP/GA, Priority-
ULD combos, and GNN ranker attempts), against the current best (30,475)
and the competitor's target (29,203).

Usage:
    python scripts/plot_full_comparison.py
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'full_comparison.png')

# (label, cost, category)
RESULTS = [
    ('value_density (orig)', 30822, 'classical: value-density family'),
    ('pow1.3', 30742, 'classical: value-density family'),
    ('pow1.4', 30601, 'classical: value-density family'),
    ('pow1.5 (BEST)', 30475, 'classical: value-density family'),
    ('pow1.6', 31023, 'classical: value-density family'),
    ('pow1.7', 30641, 'classical: value-density family'),
    ('pow2.0', 31430, 'classical: value-density family'),
    ('wpow1.0', 32221, 'classical: weight/joint density'),
    ('wpow1.5', 33017, 'classical: weight/joint density'),
    ('joint_pow1.0', 31811, 'classical: weight/joint density'),
    ('joint_pow1.3', 31371, 'classical: weight/joint density'),
    ('joint_pow1.5', 31664, 'classical: weight/joint density'),
    ('joint_pow1.7', 30942, 'classical: weight/joint density'),
    ('joint_pow2.0', 31562, 'classical: weight/joint density'),
    ('ascending_volume', 31207, 'classical: weight/joint density'),
    ('worst_fit', 34034, 'ULD-ordering / packing variants'),
    ('multi_restart', 30567, 'ULD-ordering / packing variants'),
    ('knapsack (best of 5)', 30972, 'ILP / genetic algorithm'),
    ('GA (nominal proxy)', 31525, 'ILP / genetic algorithm'),
    ('GA (real-simplified proxy)', 32236, 'ILP / genetic algorithm'),
    ('best alt. Priority combo', 30732, 'Priority-ULD allocation'),
    ('GNN: D (large, formula)', 31798, 'GNN ranker'),
    ('GNN: B (small, formula)', 30937, 'GNN ranker'),
    ('GNN: C (med, formula)', 31118, 'GNN ranker'),
    ('GNN: allscenes (+random)', 31084, 'GNN ranker'),
    ('GNN: bootstrap self-play (best)', 30787, 'GNN ranker'),
    ('RL: REINFORCE', 31081, 'RL fine-tune (real reward)'),
    ('RL: GRPO v1 (lr1e-4,g6)', 30920, 'RL fine-tune (real reward)'),
    ('RL: GRPO v2 (final)', 30672, 'RL fine-tune (real reward)'),
    ('RL: GRPO v3 [running]', 30672, 'RL fine-tune (real reward)'),
    ('RL: PPO v1 (ep4,clip0.2, worsening)', 31314, 'RL fine-tune (real reward)'),
    ('RL: PPO v2 (ep2,clip0.1,gentler)', 30991, 'RL fine-tune (real reward)'),
    ('RL: multi-instance v1 (300 rounds)', 32955, 'RL fine-tune (real reward)'),
    ('RL: multi-instance v3 (~2000 rounds) [running]', 30726, 'RL fine-tune (real reward)'),
    ('Halley: multi-instance GRPO (final, generalized)', 30608, 'RL fine-tune (real reward)'),
    ('MILP ceiling: exact volume+weight selection, real 3D-packed', 29972, 'Exact solver (relaxation)'),
    ('Local search (order-based, real-evaluated)', 29656, 'Local search / assignment search'),
    ('AI-guided search (SwapProposer, first iteration)', 29564, 'Local search / assignment search'),
    ('Assignment search (direct reformulation)', 29564, 'Local search / assignment search'),
    ('+ EMS candidate-generation fix', 28960, 'Local search / assignment search'),
    ('Eclipse: full ensemble (RL placement + heuristics)', 28452, 'Local search / assignment search'),
    ('Cherry: + centrifuge-evict refinement (BEST)', 28409, 'Local search / assignment search'),
]

CATEGORY_COLORS = {
    'classical: value-density family': '#4C72B0',
    'classical: weight/joint density': '#8172B2',
    'ULD-ordering / packing variants': '#937860',
    'ILP / genetic algorithm': '#DA8BC3',
    'Priority-ULD allocation': '#CCB974',
    'GNN ranker': '#55A868',
    'RL fine-tune (real reward)': '#C44E52',
    'Exact solver (relaxation)': '#8C8C8C',
    'Local search / assignment search': '#2E7D32',
}

CURRENT_BEST = 28409
COMPETITOR_TARGET = 29203


def main():
    results_sorted = sorted(RESULTS, key=lambda r: r[1])
    labels = [r[0] for r in results_sorted]
    costs = [r[1] for r in results_sorted]
    colors = [CATEGORY_COLORS[r[2]] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(11, 12))
    bars = ax.barh(labels, costs, color=colors)
    ax.axvline(CURRENT_BEST, color='black', linestyle='-', linewidth=1.5,
               label=f'Current best (Cherry): {CURRENT_BEST:,}')
    ax.axvline(COMPETITOR_TARGET, color='red', linestyle='--', linewidth=1.5,
               label=f'Competitor target: {COMPETITOR_TARGET:,}')
    ax.set_xlabel('Real cost (lower is better)')
    ax.set_title('Every real-validated strategy tried across the whole project')
    ax.set_xlim(28000, max(costs) + 500)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f'{cost:,.0f}', va='center', fontsize=8)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()]
    legend_elems.append(plt.Line2D([0], [0], color='black', lw=1.5, label=f'Current best (Cherry): {CURRENT_BEST:,}'))
    legend_elems.append(plt.Line2D([0], [0], color='red', lw=1.5, ls='--', label=f'Competitor target: {COMPETITOR_TARGET:,}'))
    ax.legend(handles=legend_elems, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'Saved plot to {OUT_PNG}')


if __name__ == '__main__':
    main()
