"""
generate_flowcharts.py -- renders standalone PNG flowcharts (overview and
detailed) of the hybrid AI+heuristic cargo packer, using plain matplotlib
boxes/arrows so the images work directly in any presentation tool (not
dependent on a mermaid renderer).

Usage:
    python generate_flowcharts.py
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))

COLOR_INPUT = '#8f9bb3'
COLOR_HEURISTIC = '#f5a623'
COLOR_AI = '#4c8bf5'
COLOR_HYBRID = '#7ed321'
COLOR_OUTPUT = '#50c878'
TEXT_COLOR = '#1a1a1a'


def box(ax, xy, w, h, text, color, fontsize=10, text_color='white'):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                            linewidth=1.5, edgecolor='#333333', facecolor=color, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=3, wrap=True)
    return (x + w / 2, y), (x + w / 2, y + h), (x, y + h / 2), (x + w, y + h / 2)


def arrow(ax, start, end, label=None, color='#555555'):
    a = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18,
                         linewidth=1.8, color=color, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=8.5, color=color, ha='left', va='center', style='italic')


# ---------------------------------------------------------------------
# Overview flowchart
# ---------------------------------------------------------------------
def overview_flowchart():
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 15)
    ax.axis('off')
    ax.set_title('Cherry -- Hybrid AI + Heuristic Cargo Packer -- Overview', fontsize=15, fontweight='bold', pad=10)

    w = 6
    x0 = 1
    steps = [
        ('Input: packages + ULDs', COLOR_INPUT),
        ('AI model: Priority-to-ULD\nassignment (trained network)', COLOR_AI),
        ('Priority packing\n(exhaustive, guaranteed placement)', COLOR_HEURISTIC),
        ('Economy package search\n(local search over ordering / assignment)', COLOR_HEURISTIC),
        ('Hybrid 3D packer\n(AI placement model + heuristics,\nbest-of-N per container)', COLOR_HYBRID),
        ('Cherry: AI-guided evict-centrifuge\nrefinement (CentrifugeEvictProposer\nranks candidates, top-K real-verified)', COLOR_AI),
        ('Final packing solution', COLOR_OUTPUT),
    ]
    h = 1.4
    gap = 0.6
    ys = []
    y = 13.2
    for _ in steps:
        ys.append(y)
        y -= (h + gap)

    centers = []
    for (text, color), y in zip(steps, ys):
        c = box(ax, (x0, y), w, h, text, color, fontsize=11)
        centers.append((x0 + w / 2, y, x0 + w / 2, y + h))

    for i in range(len(centers) - 1):
        top_of_next = centers[i + 1][3]
        bottom_of_cur = centers[i][1]
        arrow(ax, (x0 + w / 2, bottom_of_cur), (x0 + w / 2, top_of_next))

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'overview_flowchart.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Detailed flowchart
# ---------------------------------------------------------------------
def detailed_flowchart():
    fig, ax = plt.subplots(figsize=(15, 15.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16.5)
    ax.axis('off')
    ax.set_title('Cherry -- Hybrid AI + Heuristic Cargo Packer -- Detailed Pipeline', fontsize=16, fontweight='bold', pad=10)

    # Stage 1: input + priority assignment
    box(ax, (0.5, 14.5), 4, 1.3, 'Input CSV\n(packages, ULDs, K)', COLOR_INPUT, fontsize=9)
    box(ax, (5.2, 14.5), 4.3, 1.3, 'Priority Clusterer\n(trained neural network)', COLOR_AI, fontsize=9)
    box(ax, (10.2, 14.5), 5.3, 1.3, 'Priority -> ULD partition\n+ exhaustive Priority packing', COLOR_HEURISTIC, fontsize=9)
    arrow(ax, (4.5, 15.15), (5.2, 15.15))
    arrow(ax, (9.5, 15.15), (10.2, 15.15))

    # Stage 2: economy seed
    box(ax, (10.2, 12.5), 5.3, 1.3, 'Economy package seed order\n(value-density heuristic)', COLOR_HEURISTIC, fontsize=9)
    arrow(ax, (12.85, 14.5), (12.85, 13.8))

    # Stage 3: local search loop (big box)
    loop_x, loop_y, loop_w, loop_h = 0.5, 7.8, 15, 4.2
    ax.add_patch(FancyBboxPatch((loop_x, loop_y), loop_w, loop_h, boxstyle='round,pad=0.05,rounding_size=0.15',
                                 linewidth=2, edgecolor='#333333', facecolor='#f0f2f5', zorder=0))
    ax.text(loop_x + 0.2, loop_y + loop_h - 0.35, 'Local Search Loop (repeats each round)',
            fontsize=11, fontweight='bold', color='#333333')

    box(ax, (1, 9.7), 6, 1.2, 'Candidate moves\n(swap / relocate / block-shuffle)', COLOR_HEURISTIC, fontsize=8.5)
    box(ax, (8.3, 9.7), 6.5, 1.2, 'All candidates real-evaluated directly\n(no pre-screening in the winning run)', COLOR_HEURISTIC, fontsize=8.5)
    arrow(ax, (7, 10.3), (8.3, 10.3))

    box(ax, (1, 8.2), 13.8, 1.1, 'Hybrid 3D Packer -- REAL evaluation (per container, best-of-N):\n'
                                  'AI placement policy  |  heuristic strategies (contact-area, deepest-corner) x2 candidate-generation methods',
        COLOR_HYBRID, fontsize=8.5)
    arrow(ax, (11.5, 9.7), (7.9, 9.3))

    arrow(ax, (12.85, 12.5), (4, 10.9))  # seed order feeds into loop
    arrow(ax, (7.9, 8.2), (7.9, 7.0))  # loop feeds down toward decision

    ax.text(loop_x + 0.2, loop_y + 0.15,
            'A separate AI-guided variant (SwapProposer pre-screening) was also built and tested here,\n'
            'but did not demonstrate improvement over this direct approach -- see README for the full comparison.',
            fontsize=8, style='italic', color='#555555')

    # Stage 4: decision
    box(ax, (5.5, 5.3), 5, 1.3, 'Keep best-cost candidates\n(elitist selection, deduplicated)', COLOR_HEURISTIC, fontsize=9)
    arrow(ax, (7.9, 5.3), (7.9, 4.6))
    arrow(ax, (10.5, 5.95), (11.5, 7.8), color='#999999')
    ax.text(11.7, 6.8, 'repeat until\nplateau', fontsize=8.5, style='italic', color='#555555')

    # Stage 5: Cherry -- AI-guided evict-centrifuge refinement (applied after
    # local search plateaus, on the search's best solution)
    box(ax, (4.5, 3.0), 7, 1.3,
        'Cherry: CentrifugeEvictProposer-guided refinement\n'
        'evict -> compact -> refill, model ranks all candidates,\n'
        'only top-K real-verified (compact+refill), apply if profitable',
        COLOR_AI, fontsize=8.5)
    arrow(ax, (8, 3.0), (8, 2.05))
    ax.text(9.7, 2.5, 'repeat until no\nprofitable move found', fontsize=8, style='italic', color='#555555')
    ax.text(1.0, 2.5,
            'Real benchmark: 20 real\nchecks (top-10 x 2 rounds) vs.\n151 exhaustive -- same final\ncost, ~7.5x fewer checks.',
            fontsize=7.5, style='italic', color='#555555')

    box(ax, (5.5, 0.6), 5, 1.3, 'Final packing solution', COLOR_OUTPUT, fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'detailed_flowchart.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    overview_flowchart()
    detailed_flowchart()
    print(f'Saved flowcharts to {HERE}')
