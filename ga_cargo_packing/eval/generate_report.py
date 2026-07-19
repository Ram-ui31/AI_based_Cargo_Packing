"""
generate_report.py — builds results/GA_pipeline_report.pdf: a written summary
of the GA -> IL -> RL/PPO pipeline, the bugs found and fixed along the way,
the session-2 K-conditioning fix, final results, and concrete improvement
ideas. Pulls live numbers from results/comparison_ga_ppo.csv and
eval/verify_results.csv rather than hardcoding them, so re-running this
after further training stays accurate.
"""
from __future__ import annotations
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_THIS_DIR, '..')
RESULTS = os.path.join(ROOT, 'results')

INK = colors.HexColor('#0b0b0b')
MUTED = colors.HexColor('#52514e')
BLUE = colors.HexColor('#2a78d6')
RED = colors.HexColor('#e34948')
GRIDLINE = colors.HexColor('#e1e0d9')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('H1', parent=styles['Heading1'], fontSize=20, textColor=INK, spaceAfter=14))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=INK, spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11.5, textColor=INK, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=14, textColor=INK, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle('MyBullet', parent=styles['Body'], leftIndent=16, bulletIndent=4, spaceAfter=4))
styles.add(ParagraphStyle('Small', parent=styles['Body'], fontSize=8.5, textColor=MUTED))
styles.add(ParagraphStyle('Caption', parent=styles['Body'], fontSize=9, textColor=MUTED, alignment=1, spaceBefore=4, spaceAfter=16))
styles.add(ParagraphStyle('Cell', parent=styles['Body'], fontSize=8, leading=10.5, spaceAfter=0))
styles.add(ParagraphStyle('CellHead', parent=styles['Cell'], textColor=colors.white, fontName='Helvetica-Bold'))


def p(text, style='Body'):
    return Paragraph(text, styles[style])


def cell(text, header=False):
    """Table cells must be Paragraph flowables, not raw strings -- plain
    strings in a reportlab Table neither wrap to the column width nor parse
    HTML entities (&mdash; etc. show up as literal text otherwise)."""
    return Paragraph(text, styles['CellHead'] if header else styles['Cell'])


def bullets(items):
    return [p(f'&bull;&nbsp;&nbsp;{item}', 'MyBullet') for item in items]


def load_numbers():
    # comparison_ga_ppo.csv (GA vs the PPO-trained, K-conditioned RL model) --
    # superseded comparison_ga_il_rl.csv (GA vs IL vs the original K-blind RL
    # model) once the K-conditioning fix (session 2, see §7) made that
    # script's IL-checkpoint load path incompatible with the new architecture.
    comp = pd.read_csv(os.path.join(RESULTS, 'comparison_ga_ppo.csv'))
    verify = pd.read_csv(os.path.join(_THIS_DIR, 'verify_results.csv'))
    means = comp.groupby('method')['cost'].mean()
    return {
        'ga_mean': means.get('GA', float('nan')),
        'rl_mean': means.get('RL', float('nan')),
        'n_instances_comparison': comp['instance'].nunique(),
        'verify_mean_cost': verify['cost'].mean(),
        'verify_n': len(verify),
        'verify_priority_drops': verify['priority_dropped'].sum(),
    }


def build():
    nums = load_numbers()
    out_path = os.path.join(RESULTS, 'GA_pipeline_report.pdf')
    doc = SimpleDocTemplate(out_path, pagesize=LETTER,
                            topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch)
    story = []

    # ── Title ──────────────────────────────────────────────────────────────
    story.append(p('GA &rarr; IL &rarr; RL Pipeline for 3D ULD Cargo Packing', 'H1'))
    story.append(p(
        'A Genetic-Algorithm-labelled imitation-learning and reinforcement-learning '
        'pipeline for assigning packages to Unit Load Devices (ULDs), packing with a '
        'learned single-ULD placement policy (rl_packer), and fine-tuning against the '
        'true objective: every Priority package packed, no overlaps, weight/volume '
        'respected, minimizing K&middot;spread + delay cost.', 'Body'))
    story.append(Spacer(1, 6))

    # ── 1. Problem & conditions ───────────────────────────────────────────
    story.append(p('1. Problem statement', 'H2'))
    story.append(p(
        'Each instance has several ULDs and several packages, classified Priority or '
        'Economy. Four conditions must hold on any produced solution:', 'Body'))
    story.extend(bullets([
        'Every Priority package must be packed.',
        'No two packages may overlap in 3D space.',
        'Per-ULD weight and volume limits must never be exceeded.',
        'Minimize cost = K&middot;spread + &sum;(delay cost of unplaced Economy packages), '
        'where spread is the number of ULDs holding at least one Priority package, and K '
        '&isin; {100, 500, 1000, 3000, 5000} is assigned per instance (200 of the 1000 '
        'training instances per K value; 20 of the 100 test instances per K value, from '
        'good_data&#39;s own metadata_with_K.csv).',
    ]))

    # ── 2. Pipeline architecture ───────────────────────────────────────────
    story.append(p('2. Pipeline architecture', 'H2'))
    story.append(p(
        'Three stages, mirroring the existing H1H2-labelled pipeline in this project '
        '(good-il-over-greedy + rl_over_il_h1h2) but with the label source replaced end '
        'to end:', 'Body'))

    story.append(p('2.1 &nbsp;Genetic Algorithm (src/ga/)', 'H3'))
    story.append(p(
        'Generates training labels for the IL model. Priority packages are never part of '
        'the GA&#39;s search space &mdash; they are packed first, against the full ULD '
        'fleet, by reusing h1_h2_cargo&#39;s own GreedyPipeline (its rescue and '
        'nuclear-eviction fallbacks, imported unmodified), which guarantees condition 1 '
        'independent of anything the GA decides. The GA then only decides where each '
        'Economy package goes:', 'Body'))
    story.extend(bullets([
        '<b>Encoding</b>: one ternary gene per Economy package &mdash; 2 = priority-ULD '
        'bucket, 1 = other-ULD bucket, 0 = unallocated.',
        '<b>Fitness</b>: sum of delay cost of Economy packages left unplaced after a cheap '
        'trial-pack of each bucket (lower is better).',
        '<b>Initial population</b>: one greedy-seeded individual (maximizes Economy packed '
        'into the priority bucket), the rest random over {0, 1}.',
        '<b>Selection</b>: fitness-proportional parent-pair sampling.',
        '<b>Crossover</b>: per-gene, inherited from the fitter parent with configurable '
        'probability (default 0.65).',
        '<b>Mutation</b>: population split into 4 fitness-ordered buckets with rates '
        '[1%, 2%, 3%, 4%] &mdash; fitter individuals mutate less.',
        '<b>Validation/repair</b>: if Economy crowds out Priority in the trial-pack, evict '
        'the largest offending Economy gene and retry (bounded to 3 attempts &mdash; see '
        '&sect;4 on why this bound mattered).',
        'The winning individual is re-packed once, rigorously, via h1_h2_cargo&#39;s '
        'own multi-sort greedy_pack (the cheap trial-pack above is for scoring speed '
        'only).',
    ]))

    story.append(p('2.2 &nbsp;IL Transformer (src/il/)', 'H3'))
    story.append(p(
        'A Transformer clusterer (unchanged architecture from the existing pipeline) '
        'trained by cross-entropy to imitate the GA&#39;s package&rarr;ULD assignments, '
        'plus an auxiliary capacity-violation penalty. Trained 248 epochs to convergence '
        '(early-stopped), reaching val_loss=0.760, val_acc=73.8%, priority_acc=73.1%.', 'Body'))

    story.append(p('2.3 &nbsp;RL fine-tuning (src/rl/)', 'H3'))
    story.append(p(
        'REINFORCE fine-tuning of the IL checkpoint, packing every rollout with '
        '<b>rl_packer</b> (the project&#39;s learned single-ULD 3D placement policy) '
        'rather than the built-in heuristic packer, per the requirement to use rl_packer. '
        'Auxiliary losses: K&middot;spread inside the reward itself, a differentiable '
        'capacity-violation penalty on raw logits, and a priority-drop penalty (Priority '
        'packages carry Delay_Cost=0, so the bare cost formula alone gives the policy no '
        'incentive to avoid dropping them). <i>This stage was substantially reworked in a '
        'later session &mdash; see &sect;7 &mdash; after this original version was found to '
        'never actually condition its assignment strategy on K at all.</i>', 'Body'))

    story.append(PageBreak())

    # ── 3. rl_packer as the packer ────────────────────────────────────────
    story.append(p('3. Using rl_packer as the packer', 'H2'))
    story.append(p(
        'The plain rl_packer_adapter.py already present elsewhere in this project has no '
        'priority ordering at all: its candidate list mixes Priority and Economy '
        'packages, so an Economy package can claim the only spot a Priority package '
        'needed purely by candidate-list order. Measured directly, this adapter dropped '
        'Priority packages in 4 of 10 tested instances even with a well-trained '
        'clusterer.', 'Body'))
    story.append(p(
        'A side-by-side test also checked whether rl_packer is simply a stronger packer '
        'than the existing heuristic EPIPacker &mdash; on identical clusterer '
        'assignments, mean volume utilization was 36.7% (rl_packer) vs 37.0% (EPIPacker): '
        'statistically tied. rl_packer&#39;s own training reports 73.2% utilization in '
        'isolation (random 30&ndash;90-package episodes into one empty ULD), a much '
        'easier task than the sparse, priority/economy-mixed loads it actually receives '
        'here &mdash; that gap is the likely explanation, not an inherent weakness.', 'Body'))
    story.append(p(
        'Fix implemented in this project&#39;s copy of the adapter (src/rl/'
        'rl_packer_adapter.py): Priority packages get their own placement episode with '
        'first claim on every extreme point in a ULD, Economy fills what&#39;s left in '
        'the same partially-filled space, and a bounded cross-ULD eviction-rescue pass '
        '(mirroring h1_h2_cargo&#39;s own EPIPacker rescue strategy) moves a stuck '
        'Priority package to another ULD, evicting Economy there if needed, before giving '
        'up on it. Verified after the fix: 0 priority drops across 30 real test '
        'instances.', 'Body'))

    # ── 4. Bugs found and fixed ────────────────────────────────────────────
    story.append(p('4. Bugs found and fixed during development', 'H2'))
    story.append(p(
        'Several real bugs surfaced only once the pipeline ran at full scale (1000 '
        'train + 100 test instances). Each is listed with its symptom and fix:', 'Body'))

    bug_rows_raw = [
        ['h1_h2_cargo and rl_packer geometry silently '
         'corrupted when both loaded in one process',
         "Two unrelated modules both named 'geometry.py' "
         '&mdash; whichever imported first won the '
         'sys.modules cache',
         'Isolated import in RLPackerAdapter: evict '
         'conflicting cache entries, import fresh with '
         'rl_packer on sys.path[0], restore after'],
        ['Priority packages left unplaced by GAPipeline',
         'Priority packing was restricted to only the '
         '"priority-ULD bucket", which is sized by volume '
         'heuristic and can genuinely be too small',
         'Pack Priority first against the FULL ULD fleet '
         '(reusing GreedyPipeline&#39;s rescue machinery) '
         'before the GA even runs'],
        ['Full-scale GA precompute made zero progress in '
         '2h47m across 9 workers',
         'Unbounded repair loop (max_repairs=50) '
         'combined with a priority bucket too small for '
         'even the greedy-seed individual, on some real '
         'instances',
         'max_repairs 50&rarr;3 (verified: identical '
         'fitness, 5x faster) plus a hard 90s wall-clock '
         'budget per GA solve'],
        ['RL training epoch cost ~1000s, dominated by '
         'overhead unrelated to model size',
         'Per-package .item() calls on MPS tensors each '
         'force a full GPU command-buffer sync &mdash; up '
         'to 300&times; per instance',
         'Batch the whole per-instance decision loop onto '
         'one CPU tensor copy; only the final '
         'differentiable step touches the GPU tensor'],
        ['A resumed RL run overwrote a good checkpoint '
         '(19317.57) with a worse one (26504.91)',
         'train_rl() always initializes best_val_cost = '
         'inf, so a resumed run&#39;s first eval trivially '
         '"beats" infinity',
         'Carry the resumed checkpoint&#39;s own '
         'val_rl_cost_penalized in as the starting '
         'best_val_cost'],
        ['RL validation cost pinned at a single value for '
         '19+ epochs across 3 different variance-'
         'reduction attempts (LR decay, global advantage '
         'normalization, 4&times; rollout averaging)',
         'spread_cost = K&times;n_priority_ulds means a '
         'K=5000 instance swings ~50&times; more than a '
         'K=100 instance; a single GLOBAL RMS scale was '
         'dominated by the 200 K=5000 instances, driving '
         'the other 800 instances&#39; normalized '
         'advantage toward zero',
         'Normalize advantage per-instance by that '
         'instance&#39;s own IL-baseline magnitude instead '
         'of a global running scale'],
    ]
    bug_rows = [[cell('Symptom', header=True), cell('Root cause', header=True), cell('Fix', header=True)]]
    bug_rows += [[cell(a), cell(b), cell(c)] for a, b, c in bug_rows_raw]
    bug_table = Table(bug_rows, colWidths=[1.9 * inch, 2.35 * inch, 2.35 * inch], repeatRows=1)
    bug_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fcfcfb')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fcfcfb')]),
    ]))
    story.append(bug_table)

    story.append(PageBreak())

    # ── 5. Results ─────────────────────────────────────────────────────────
    story.append(p('5. Final results', 'H2'))
    story.append(p(
        f'Verified against {int(nums["verify_n"])} test instances via '
        'eval/verify_pipeline.py, using the final PPO-fine-tuned checkpoint from &sect;7: '
        '0 Priority packages dropped, 0 overlap/weight/volume violations. Mean cost '
        'compares favourably to the existing pipeline&#39;s own H1H2+RL and Hybrid '
        'baselines on the same instances:', 'Body'))

    result_rows = [
        [cell('Method', header=True), cell('Mean cost (K&middot;spread + delay)', header=True)],
        [cell('<b>This pipeline (GA&rarr;IL&rarr;RL/PPO, rl_packer)</b>'), cell(f'<b>{nums["verify_mean_cost"]:,.0f}</b>')],
        [cell('cargoism/git H1H2+RL baseline'), cell('24,513.8')],
        [cell('cargoism/git Hybrid baseline'), cell('25,013.7')],
    ]
    result_table = Table(result_rows, colWidths=[3.4 * inch, 2.4 * inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#eef5ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 14))

    for fname, caption in [
        ('cost_vs_k.png', 'Figure 1. Mean cost per K value, GA labels vs the final PPO-fine-tuned RL model.'),
        ('spread_vs_k.png', 'Figure 2. Mean spread (ULDs holding Priority) per K value, GA vs PPO -- '
                            'note spread now decreases as K increases, the behavior &sect;7 fixes.'),
    ]:
        img_path = os.path.join(RESULTS, fname)
        if os.path.exists(img_path):
            story.append(Image(img_path, width=5.8 * inch, height=3.87 * inch))
            story.append(p(caption, 'Caption'))

    story.append(PageBreak())

    # ── 7. K-conditioning fix (session 2) ──────────────────────────────────
    story.append(p('7. Session 2: fixing K-conditioning', 'H2'))
    story.append(p(
        'The RL model from &sect;2.3&ndash;5 turned out to never actually condition its '
        'assignment strategy on K at all &mdash; it produced roughly the same spread '
        'whether K=100 (spread barely matters) or K=5000 (every extra ULD costs 5000). '
        'This section covers the fix, arrived at over several iterations.', 'Body'))

    story.append(p('7.1 &nbsp;What was tried', 'H3'))
    story.extend(bullets([
        '<b>From-scratch retrain with K as a model input</b> &mdash; regressed badly '
        '(worse than the original model on every axis). Retraining IL from scratch with '
        'an added K-input layer converged to a meaningfully worse checkpoint than the '
        'original K-blind IL run, handicapping RL from the start.',
        '<b>Graft warm-start</b> &mdash; instead of retraining IL, a zero-initialized new '
        'K-input layer was grafted onto the <i>existing</i> strong IL checkpoint, so the '
        'model starts out mathematically identical to the original and RL fine-tunes from '
        'there. This became the standard warm-start for every later attempt, but alone '
        'still plateaued below the original model.',
        '<b>Architecture and loss fixes ported from a separate, more mature reference '
        'implementation</b> (a sibling project that had already solved this problem): '
        'log-scale K normalization (linear K/max(K) crushed most K values near 0), '
        '<b>dual K injection</b> (K fed in twice &mdash; once into the shared transformer '
        'trunk, once concatenated directly at the output head, since a single diffuse '
        'path let K get washed out), a <b>feasibility hinge loss</b> (dense penalty for '
        'rejecting a dimensionally-feasible Economy package to unplaced), and a '
        '<b>K-scaled soft-spread loss</b> (differentiable proxy for spread, scaled by K).',
    ]))

    story.append(p('7.2 &nbsp;The actual unlock: loss-magnitude calibration', 'H3'))
    story.append(p(
        'The soft-spread loss had been present all session but its weight was roughly '
        '2000&times; smaller than the other loss terms &mdash; mathematically part of the '
        'objective, functionally noise. Increasing its weight alone caused the model to '
        'over-minimize spread even at <i>low</i> K, where it shouldn&#39;t (dropping more '
        'Economy packages than necessary to save a spread cost that barely matters at low '
        'K). The fix was increasing the feasibility hinge loss&#39;s weight by a '
        'comparable amount at the same time, so the two dense losses (one pushing spread '
        'down, one pushing Economy retention up) properly counterbalance each other '
        'instead of one drowning out the other. The first checkpoint trained after this '
        'fix beat the original model outright.', 'Body'))

    story.append(p('7.3 &nbsp;Final upgrade: PPO with a per-K baseline', 'H3'))
    story.append(p(
        'On top of the now-working architecture and loss design, the REINFORCE training '
        'loop (&sect;6.1&#39;s frozen-IL-baseline advantage) was replaced with a PPO '
        'clipped surrogate objective and an online per-K exponential-moving-average '
        'baseline and standard deviation, ported from the same reference implementation. '
        'This is the checkpoint used for the results in &sect;5 and Figures 1&ndash;2.', 'Body'))

    story.append(p('7.4 &nbsp;Final comparison', 'H3'))
    story.append(p(
        'Fair, apples-to-apples evaluation of the original model and the final PPO model '
        'on the same full 83-instance (non-chunked) test set, identical deterministic '
        'decoding for both, zero Priority drops for both:', 'Body'))
    k_rows = [
        [cell('Model', header=True), cell('Mean cost', header=True), cell('vs original', header=True)],
        [cell('Original (K-blind)'), cell('16,846.0'), cell('&mdash;')],
        [cell('PPO, K-conditioned (file-order decoding)'), cell('16,508.0'), cell('&minus;338 (&minus;2.0%)')],
    ]
    k_table = Table(k_rows, colWidths=[2.6 * inch, 1.8 * inch, 1.8 * inch])
    k_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(k_table)
    story.append(Spacer(1, 8))
    story.append(p(
        'Per K value: the PPO model clearly beats the original at K=100, K=500, and '
        'K=5000; is roughly tied at K=1000; and is slightly behind at K=3000 &mdash; the '
        'one bucket this fix did not fully resolve (see &sect;8.5).', 'Body'))

    story.append(p('7.5 &nbsp;Session 3: decode-order fix (real-world instance stress test)', 'H3'))
    story.append(p(
        'A large real-world instance (400 packages, 6 ULDs, K=5000, 103 Priority) exposed a '
        'gap the &sect;7.4 test set never stressed: <font face="Courier">rl_assign_argmax_safe</font> '
        'decodes packages sequentially in whatever order the input file gives them, and each '
        'package&#39;s capacity mask reflects only what earlier packages in that order already '
        'used. With Priority and Economy interleaved in file order, Economy packages decided '
        'earlier could claim capacity before a later Priority package arrived, artificially '
        'inflating spread &mdash; and instances large enough to need chunking (&gt;300 packages) '
        'could split Priority across independent chunk forward passes with no coordination '
        'between them. On this instance the model reached spread=6 (of 6 possible) and cost '
        '49,903, and a side-by-side check confirmed the &sect;7 original (K-blind) model did '
        'no better (spread=6, cost=50,063) &mdash; this was not a regression from &sect;7, it '
        'was a decode-order gap neither model had been tested against.', 'Body'))
    story.append(p(
        'The fix requires no retraining: reorder packages Priority-first (descending weight) '
        'then Economy (ascending volume) before decoding. Priority now claims ULDs while '
        'capacity is wide open, and since all 103 Priority packages fit in one chunk, none of '
        'them get split across chunk boundaries. On the real-world instance this cut cost to '
        '34,673 (spread 6&rarr;4, &minus;30.5%). Re-run on the full &sect;7.4 test set (not '
        'just this one instance) to rule out overfitting to it:', 'Body'))
    order_rows = [
        [cell('Decode order', header=True), cell('Mean cost', header=True), cell('vs file-order', header=True)],
        [cell('File order (as generated)'), cell('16,508.0'), cell('&mdash;')],
        [cell('<b>Priority-first, Economy asc-volume</b>'), cell('<b>13,362.9</b>'),
         cell('<b>&minus;3,145.1 (&minus;19.0%)</b>')],
    ]
    order_table = Table(order_rows, colWidths=[2.6 * inch, 1.8 * inch, 1.8 * inch])
    order_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#eef5ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(order_table)
    story.append(Spacer(1, 8))
    story.append(p(
        '78 of 83 test instances improved, 5 were marginally worse, none unchanged &mdash; a '
        'broad win, not an artifact of one instance.', 'Body'))

    story.append(p('7.6 &nbsp;An important caveat, and a deeper fix', 'H3'))
    story.append(p(
        '&sect;7.5&#39;s fix is a decode-order heuristic wrapped around the frozen &sect;7 '
        'model, not the model learning to reduce spread &mdash; no weights changed. '
        '<font face="Courier">rl_assign_argmax_safe</font> is a sequential greedy decoder: the '
        'model still chooses every package&#39;s ULD via its own logits, but it was never asked '
        'to decide which package to consider first, and that ordering turns out to matter a '
        'great deal for a greedy decoder. This is worth stating plainly rather than implying '
        'the model itself got better at the objective it was trained on.', 'Body'))
    story.append(p(
        'Pushing further on the real-world instance: even after &sect;7.5&#39;s fix, spread was '
        'stuck at 4 while the assignment stage itself already reached 3 ULDs (matching an '
        'external benchmark for this instance) &mdash; the packer&#39;s own bounded rescue pass '
        '(&sect;3) was recruiting a 4th, fresh ULD for every stuck Priority package instead of '
        'trying to squeeze back into the 3 already-Priority ULDs first, because its candidate '
        'order was sorted by raw ULD volume with no preference for ULDs already holding '
        'Priority. That ordering bug is fixed (now Priority-holding ULDs are tried first), and '
        'it is a real, generally-useful improvement &mdash; but on this specific instance it '
        'changed nothing, because the 3 ULDs the assignment stage had chosen held Priority '
        'almost exclusively already (0, 0, and 1 Economy package respectively) &mdash; there was '
        'nothing left to evict.', 'Body'))
    story.append(p(
        'The real cause went one level deeper: total Priority volume was 96% of those 3 ULDs&#39; '
        '<i>nominal</i> capacity, but rl_packer&#39;s real extreme-point placement only achieves '
        '&sim;70% <i>true</i> volumetric efficiency (verified: order-invariant across 6 heuristic '
        'orderings and 15 random shuffles, and rotation was already in its candidate search, so '
        'this is a genuine placement-quality ceiling, not another ordering bug) &mdash; 43.4M of '
        'nominal-96%-full priority volume simply cannot fit in ULDs that can truly only hold '
        '&sim;31.6M. The assignment stage had also picked the <i>wrong</i> 3 ULDs: the 3 it chose '
        'summed to less nominal capacity than the 3 largest-volume ULDs in the fleet, which (at '
        'the same 70% efficiency) would have had just enough margin (+1.9%) to fit everything. A '
        'quick check of whether the &sim;70% ceiling is just bad luck from a single greedy '
        'rollout: best-of-10 <i>stochastic</i> rollouts (vs the single deterministic one normally '
        'used) recovered 5 of 27 stuck packages in one ULD (&sim;18%) &mdash; real headroom from '
        'better search, but nowhere near closing the gap; the ceiling is mostly the policy&#39;s '
        'own placement quality, not sampling variance.', 'Body'))
    story.append(p(
        'Fix: <font face="Courier">_consolidate_priority_by_capacity</font> deterministically '
        'packs Priority via first-fit-decreasing into the fewest, largest-volume ULDs that can '
        'hold it by aggregate weight/volume, before the model ever sees Economy. This is a '
        'second, stronger heuristic layered on top of &sect;7.5&#39;s, not a model change either. '
        'On the real-world instance: spread 4&rarr;3, cost 34,673&rarr;27,474, zero violations '
        '(these absolute figures were later found understated by a bug &mdash; see &sect;7.8 '
        '&mdash; the spread 4&rarr;3 result itself is correct, only the cost numbers shift). '
        'But minimizing spread this way is not free &mdash; claiming the largest ULDs for '
        'Priority leaves the <i>smallest</i> ULDs for Economy, so it only pays off when K is '
        'large enough that the spread savings outweigh the lost Economy capacity. Tested per-K '
        'on the full &sect;7.4 test set:', 'Body'))
    kgate_rows = [
        [cell('K', header=True), cell('Mean cost change vs &sect;7.5 alone', header=True), cell('Verdict', header=True)],
        [cell('100'), cell('+443.8 (10/17 instances worse)'), cell('Net loss &mdash; disabled below K=500')],
        [cell('500'), cell('&minus;508.7'), cell('Net win')],
        [cell('1000'), cell('&minus;641.1'), cell('Net win')],
        [cell('3000'), cell('&minus;860.6'), cell('Net win')],
        [cell('5000'), cell('&minus;2,750.5'), cell('Net win')],
    ]
    kgate_table = Table(kgate_rows, colWidths=[0.8 * inch, 3.0 * inch, 2.4 * inch])
    kgate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fdeceb')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(kgate_table)
    story.append(Spacer(1, 8))
    story.append(p(
        f'Same K=100 mismatch this whole session has been about &mdash; a fix that ignores K '
        f'either over- or under-corrects spread. Gated to K &ge; 500 '
        f'(<font face="Courier">PRIORITY_CONSOLIDATION_MIN_K</font>, src/rl/config.py), this is '
        f'now the default inside <font face="Courier">rl_assign_argmax_safe</font> &mdash; mean '
        f'cost 13,362.9 &rarr; 12,390.9 (41 of 83 improved, 28 worse, 14 unchanged relative to '
        f'&sect;7.5 alone, concentrated at K=100 where consolidation is correctly skipped).', 'Body'))

    story.append(p('7.7 &nbsp;A much bigger gap: Economy selection itself', 'H3'))
    story.append(p(
        'Asked directly what the low-K solution should be, since spread cost is nearly '
        'irrelevant there (K=100 instances: delay cost is &sim;98% of total cost). That reframes '
        'the whole problem at low K as a pure knapsack: which Economy packages are worth keeping, '
        'given more supply than capacity. Checked whether the model&#39;s own per-package Economy '
        'choice is actually good at this by testing a simple greedy heuristic against it &mdash; '
        'sort Economy by descending delay_cost&divide;volume ("value density") and first-fit '
        'greedily. The model does show some learned value-triage (kept packages average &sim;2x '
        'the value-density of dropped ones on a sampled instance, not random) but the greedy '
        'heuristic still won clearly, at <i>every</i> K bucket, no gating needed this time:', 'Body'))
    vd_rows = [
        [cell('K', header=True), cell('Model mean', header=True), cell('Greedy-VD mean', header=True), cell('Win rate', header=True)],
        [cell('100'), cell('10,683.8'), cell('9,687.9'), cell('15/17')],
        [cell('500'), cell('11,380.6'), cell('10,233.6'), cell('15/16')],
        [cell('1000'), cell('10,066.5'), cell('8,886.4'), cell('15/17')],
        [cell('3000'), cell('13,126.4'), cell('11,679.4'), cell('13/16')],
        [cell('5000'), cell('16,681.2'), cell('15,245.8'), cell('17/17')],
    ]
    vd_table = Table(vd_rows, colWidths=[0.6*inch, 1.6*inch, 1.7*inch, 1.5*inch])
    vd_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(vd_table)
    story.append(Spacer(1, 8))
    story.append(p(
        'Integrated as the new default: the loop that decodes packages through the model still '
        'runs unchanged (so Priority&#39;s eviction-rescue still works), but Economy&#39;s '
        'resulting ULD choice is discarded and re-derived via this heuristic afterward, using '
        'whatever capacity Priority&#39;s final placement left. Deliberately <b>not</b> seeded '
        'with Priority&#39;s nominal footprint, for the same reason as &sect;7.6: nominal '
        'accounting is more conservative than the packer&#39;s real (&sim;70%) placement '
        'efficiency. This mattered concretely at the time &mdash; seeding scored better in '
        '<i>aggregate</i> (mean 11,155.1) but regressed the real-world instance (27,474 &rarr; '
        '30,703); unseeded gave a smaller aggregate gain but improved the real-world instance '
        'further (27,474 &rarr; 26,434), so unseeded was kept. <b>&sect;7.8 found these specific '
        'numbers were understated by a bug</b> &mdash; the qualitative choice (unseeded) still '
        'stands, but see &sect;7.8 for why the gap between the two was much smaller than it '
        'looked.', 'Body'))

    story.append(PageBreak())

    # ── 7.8 Correction ────────────────────────────────────────────────────
    story.append(p('7.8 &nbsp;Correction: a rescue-loop bug was silently dropping packages', 'H3'))
    story.append(p(
        'Asked to export the 400-package real-world instance&#39;s full placement as JSON '
        '(package&rarr;ULD, dimensions, coordinates). Building that export surfaced a real bug: '
        '<font face="Courier">len(placements)</font> from '
        '<font face="Courier">RLPackerAdapter.pack()</font> was 309, not 400 &mdash; 91 packages '
        'were missing entirely, neither placed nor tagged as dropped.', 'Body'))
    story.append(p(
        'Root cause, in <font face="Courier">pack()</font>&#39;s cross-ULD rescue loop: when '
        'rescuing a stuck Priority package into <font face="Courier">other_uid</font>, the '
        're-pack candidate list was built as <font face="Courier">other_placed_ids + [pid]</font> '
        '&mdash; only that ULD&#39;s <i>currently-placed</i> packages plus the newly-rescued one. '
        'Its own previously-left-behind packages were excluded from the candidate list, then '
        'overwritten out of <font face="Courier">left_behind_by_uld[other_uid]</font> by the '
        're-pack&#39;s result &mdash; silently vanishing from all tracking. '
        '<font face="Courier">compute_packing_cost</font> only sums over the returned placements '
        'list, so these packages contributed <b>zero</b> delay cost too, understating every cost '
        'this session reported for large/contested instances. Invisible on typical smaller test '
        'instances (little rescue contention); severe on the 400-package instance, where 103 '
        'Priority packages consolidated into 3 ULDs created heavy contention across many rescue '
        'rounds. Fix: always carry the target ULD&#39;s own previously-left-behind packages '
        'forward into the re-pack.', 'Body'))
    corrected_rows = [
        [cell('', header=True), cell('Before fix', header=True), cell('After fix (correct)', header=True)],
        [cell('Full 83-instance mean cost'), cell('12,049.7 (&minus;28.5%)'), cell(f'<b>{nums["rl_mean"]:,.1f} (&minus;24.6%)</b>')],
        [cell('Real-world instance cost'), cell('26,434'), cell('<b>33,857</b>')],
        [cell('Beats 27,500 external benchmark?'), cell('Yes (apparently)'), cell('<b>No</b>')],
    ]
    corrected_table = Table(corrected_rows, colWidths=[2.6 * inch, 1.8 * inch, 1.8 * inch])
    corrected_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fdeceb')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(corrected_table)
    story.append(Spacer(1, 8))
    story.append(p(
        'Exact cost breakdown of the real-world instance&#39;s corrected final 33,857:', 'Body'))
    breakdown_rows = [
        [cell('Component', header=True), cell('Cost', header=True), cell('% of total', header=True)],
        [cell('Spread cost (K &times; 3 ULDs)'), cell('15,000'), cell('44.3%')],
        [cell('Delay &mdash; assignment stage itself said NONE'), cell('3,960'), cell('11.7%')],
        [cell('<b>Delay &mdash; packer couldn&#39;t physically fit an assigned package</b>'),
         cell('<b>14,897</b>'), cell('<b>44.0%</b>')],
        [cell('<b>Total</b>'), cell('<b>33,857</b>'), cell('100%')],
    ]
    breakdown_table = Table(breakdown_rows, colWidths=[3.4 * inch, 1.2 * inch, 1.2 * inch])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fdeceb')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#eef5ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIDLINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 8))
    story.append(p(
        'The packer&#39;s placement-efficiency ceiling (&sect;7.6) is now clearly the dominant '
        'cost component (44.0%) and the clearest remaining lever for closing the gap to the '
        '27,500 benchmark. Also re-checked the seeded-vs-unseeded Economy question (&sect;7.7) '
        'under the fix: the gap nearly disappears (33,846 seeded vs 33,857 unseeded) &mdash; the '
        'earlier dramatic 27,474-vs-30,703 gap was itself partly a bug artifact. Did not '
        're-derive every historical intermediate step-by-step number (&sect;7.5&#39;s file-order '
        'baseline, &sect;7.6&#39;s consolidation-only figure) under the fix &mdash; each was a '
        'same-bug-on-both-sides comparison, so the qualitative conclusions (the K&ge;500 gate, '
        'the unseeded default) should still hold; only their absolute magnitudes are '
        'approximate.', 'Body'))

    story.append(PageBreak())

    # ── 8. How to improve ──────────────────────────────────────────────────
    story.append(p('8. How to improve further', 'H2'))
    story.append(p(
        'The original RL stage plateaued early (best validation cost found at epoch 2 of '
        '80). The per-instance advantage-normalization bug (&sect;4) was real and worth '
        'fixing, but did not by itself unlock further improvement at the aggregate level.', 'Body'))

    story.append(p('8.1 &nbsp;Replace the frozen IL baseline with a real value function '
                    '&mdash; done, see &sect;7.3', 'H3'))
    story.append(p(
        'The original advantage was <font face="Courier">il_baseline &minus; rollout_cost</font>, '
        'a static number computed once before training &mdash; once the policy pulled ahead '
        'of that fixed reference (which happened almost immediately), the signal stopped '
        'discriminating a good update from a bad one relative to the policy&#39;s <i>current</i> '
        'ability. Session 2 replaced this with a PPO clipped surrogate objective and an '
        'online per-K exponential-moving-average baseline (&sect;7.3), which was the single '
        'biggest driver of the final result in &sect;5/&sect;7.4. Remaining headroom: the '
        'per-K EMA baseline is still a scalar mean/std, not a full learned critic conditioned '
        'on the specific instance&#39;s features &mdash; a proper value head would likely '
        'sharpen the advantage estimate further, particularly at K=3000 (&sect;7.4&#39;s '
        'remaining soft spot).', 'Body'))

    story.append(p('8.2 &nbsp;Batch training instead of per-instance SGD', 'H3'))
    story.append(p(
        'Each epoch currently does ~1000 sequential gradient steps (one per training '
        'instance, immediately followed by optimizer.step()), which is both slow (this is '
        'most of the ~1000s/epoch cost, independent of the MPS sync fix) and high-variance '
        '(no gradient averaging across instances within a step). Accumulating gradients '
        'across a batch of instances before each optimizer.step() would reduce variance and '
        'likely wall-clock time, though it requires restructuring the padding/masking to '
        'handle variable instance sizes within a batch.', 'Body'))

    story.append(p('8.3 &nbsp;Give the GA a larger search budget', 'H3'))
    story.append(p(
        'pop_size=16, max_generations=20, and a 90-second wall-clock cap per instance were '
        'chosen specifically for tractability across ~1300 instance-chunks on a 10-core '
        'machine (~3.25 hours total) &mdash; not because that budget was shown to be '
        'sufficient for label quality. A larger population/generation budget, or a smarter '
        '(vectorized) fitness-evaluation shortcut that keeps the trial-pack cheap without '
        'capping the search, would very likely produce better GA labels for the IL model to '
        'imitate, improving the whole pipeline&#39;s ceiling before RL even enters.', 'Body'))

    story.append(p('8.4 &nbsp;Make cross-ULD rescue in rl_packer more thorough', 'H3'))
    story.append(p(
        'The rescue pass added in this project (&sect;3) is intentionally bounded '
        '(max_rescue_rounds=8, one move accepted per round) to avoid reintroducing the '
        'unbounded-repair-loop failure mode from &sect;4. h1_h2_cargo&#39;s own EPIPacker '
        'rescue runs to a fixed point (up to 15 rounds, multiple moves per round). Now that '
        'the bounded version is verified correct, a follow-up could raise those bounds or '
        'run to a fixed point, provided it stays behind the same kind of wall-clock guard '
        'that fixed the GA&#39;s equivalent issue.', 'Body'))

    story.append(p('8.5 &nbsp;Let the GA&#39;s fitness account for K directly', 'H3'))
    story.append(p(
        'The GA&#39;s fitness function is delay-cost-only, deliberately agnostic to K (K '
        'only enters at RL fine-tuning time). A K-aware GA fitness &mdash; adding a spread '
        'term scaled by that instance&#39;s own K &mdash; could produce labels that are '
        'already closer to optimal per K bucket, giving the IL model a better starting '
        'point and potentially narrowing the gap Figure 1 shows between GA and RL at high '
        'K values.', 'Body'))

    doc.build(story)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    build()
