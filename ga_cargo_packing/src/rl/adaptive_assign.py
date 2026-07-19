"""
adaptive_assign.py -- replaces the fixed PRIORITY_CONSOLIDATION_MIN_K=500
global threshold with a per-instance, per-K adaptive choice.

Why: rl_assign_argmax_safe's `if k_value >= PRIORITY_CONSOLIDATION_MIN_K`
gate assumes a single crossover point (K=500) where "always fully
consolidate Priority via the heuristic" starts being cheaper than "let the
model's own learned placement decide" -- true in aggregate across the
dataset the threshold was originally tuned on, but NOT true instance by
instance. Directly measured on 8 sampled instances x 5 K values: the fixed
threshold picks the actually-cheaper option only 3/8 times below K=500 and
16/32 times (a coin flip) at K>=500. Some instances (e.g. one where the
model's own placement wins at EVERY K from 100 to 5000) should never use
the heuristic at all; others should switch sides at a K far from 500.

Fix: don't guess a threshold, and don't try to predict one either -- a
learned threshold-predictor has the exact same generalization risk as the
fixed constant it would replace (wrong on any instance whose true crossover
point the predictor didn't see in training). Instead, compute candidates
directly, pack each, and keep whichever is actually cheaper for THIS
specific (instance, K) pair -- a strict min-of-N, so more candidates can
only match or beat fewer, never do worse, and nothing needs to generalize
since every candidate is evaluated for real on the actual instance at hand.

Candidates:
    'heuristic' -- _consolidate_priority_by_capacity's minimal-ULD-count
        consolidation (forces PRIORITY_CONSOLIDATION_MIN_K below k_value).
    'model'     -- the model's own learned Priority placement (forces
        PRIORITY_CONSOLIDATION_MIN_K above k_value).
    'heuristic_plus_one' -- the SAME heuristic consolidation but with one
        extra ULD added to its candidate set beyond the minimal count that
        fits, trading a little more nominal spread for more packer headroom.
        Same reasoning that already motivated the low-K "more spread, less
        delay" tradeoff finding earlier this session, but applied as an
        explicit extra candidate rather than relying on the model to
        rediscover it from scratch -- rl_packer's real extreme-point
        placement achieves ~70% of nominal volume, not 100%, so the minimal-
        ULD consolidation sometimes leaves too little real margin for
        Economy afterward.

Does not modify train_rl.py, reward.py, or rl_packer_adapter.py -- only
imports from them, and temporarily monkeypatches
train_rl.PRIORITY_CONSOLIDATION_MIN_K (and, for the extra-slack candidate,
train_rl._consolidate_priority_by_capacity itself) for the duration of each
internal call, always restored via try/finally.
"""
from __future__ import annotations

from . import train_rl as _tr
from .reward import compute_packing_cost


def _consolidate_priority_with_extra_uld(packages_df, ulds_df):
    """Same greedy FFD-by-volume consolidation as
    _consolidate_priority_by_capacity, but tries n_ulds+1 (one more ULD
    than the minimal count that fits) whenever a larger fleet allows it --
    giving the packer more real margin at the cost of one extra
    Priority-holding ULD's worth of nominal spread."""
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
    prio_ids   = packages_df[packages_df['Type'].str.upper() == 'PRIORITY']['Package_ID'].tolist()
    if not prio_ids:
        return {}, {}, {}
    prio_sorted = sorted(
        prio_ids,
        key=lambda p: -(pkg_lookup[p]['Length'] * pkg_lookup[p]['Width'] * pkg_lookup[p]['Height']),
    )
    ulds_by_vol_desc = sorted(
        uld_lookup,
        key=lambda u: -(uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height']),
    )

    def _try_pack(candidate_ulds):
        weight_used = {u: 0.0 for u in candidate_ulds}
        volume_used = {u: 0.0 for u in candidate_ulds}
        assignment  = {}
        for pid in prio_sorted:
            p = pkg_lookup[pid]
            pw = p['Weight']
            pv = p['Length'] * p['Width'] * p['Height']
            for u in candidate_ulds:
                cap_w = uld_lookup[u]['Weight_Limit']
                cap_v = uld_lookup[u]['Length'] * uld_lookup[u]['Width'] * uld_lookup[u]['Height']
                if weight_used[u] + pw <= cap_w + 1e-6 and volume_used[u] + pv <= cap_v + 1e-6:
                    weight_used[u] += pw
                    volume_used[u] += pv
                    assignment[pid] = u
                    break
            else:
                return None
        return assignment, weight_used, volume_used

    minimal_n = None
    for n_ulds in range(1, len(ulds_by_vol_desc) + 1):
        if _try_pack(ulds_by_vol_desc[:n_ulds]) is not None:
            minimal_n = n_ulds
            break
    if minimal_n is None:
        return {}, {}, {}

    extra_n = min(minimal_n + 1, len(ulds_by_vol_desc))
    result = _try_pack(ulds_by_vol_desc[:extra_n])
    return result if result is not None else _try_pack(ulds_by_vol_desc[:minimal_n])


def rl_assign_argmax_adaptive(model, packages_df, ulds_df, device, k_value, packer,
                               max_pkgs=None, max_ulds=None, extra_candidates=False,
                               econ_sort_keys=('value_density', 'value_density_pow1.5')):
    """
    Returns (assignment, placements, cost, total_unfit, chosen) where
    `chosen` names whichever (Priority strategy, Economy sort key) candidate
    produced the lowest actual packed cost for this specific instance at
    this specific K, and `total_unfit` is that candidate's own
    packer.pack() unfit count (packages the packer itself couldn't
    physically fit -- distinct from packages the assignment stage sent
    straight to 'NONE').

    extra_candidates : if True, also evaluates 'heuristic_plus_one'
        alongside 'heuristic' and 'model'. Defaults to False -- tested on
        8 sampled instances x 5 K values, it won 0/40 times (never actually
        cheaper than the two base candidates on that sample), so it costs
        an extra forward pass + packer.pack() call for no measured benefit.
        Left available (not deleted) since it's a strict min-of-N and could
        still help on a different data distribution -- just not confirmed
        to yet, so not worth its cost by default.

    econ_sort_keys : tuple of candidate values passed to rl_assign_argmax_
        safe's econ_sort_key (the final Economy greedy first-fit ordering),
        each evaluated for real and compared by ACTUAL packed cost, same
        min-of-N pattern as the Priority heuristic/model choice above --
        deliberately NOT a single hardcoded winner. 'value_density_pow1.5'
        (delay_cost/volume^1.5) was found to beat the original
        'value_density' (delay_cost/volume, exponent 1.0) on one specific
        real 400-package instance (prompted by a competing team's result:
        20 more Economy packages placed at virtually identical per-ULD
        fill %, implying a smarter selection mix, not denser real packing;
        swept exponents 1.0-2.0 via scripts/test_econ_ascending_volume.py,
        1.5 a validated, twice-reproduced peak: 30,822->30,475, 225->233
        packages placed). That sweep was tuned on a SINGLE instance,
        though -- hardcoding 1.5 as a global default would repeat exactly
        the mistake this codebase already made once with the fixed
        PRIORITY_CONSOLIDATION_MIN_K=500 threshold (right in aggregate,
        wrong per-instance). Keeping both as live candidates here means
        1.5 is used ONLY on instances where it's actually verified cheaper,
        and instances where the plain 'value_density' still wins fall back
        to it automatically -- a strict min-of-2, so this can only match
        or beat either single fixed choice, never do worse, and needs no
        re-tuning to generalize to a different data distribution.
    """
    orig_min_k = _tr.PRIORITY_CONSOLIDATION_MIN_K
    orig_consolidate_fn = _tr._consolidate_priority_by_capacity
    candidates = {}

    def _run(label, forced_min_k, consolidate_fn=None, econ_sort_key='value_density'):
        _tr.PRIORITY_CONSOLIDATION_MIN_K = forced_min_k
        if consolidate_fn is not None:
            _tr._consolidate_priority_by_capacity = consolidate_fn
        try:
            assignment = _tr.rl_assign_argmax_safe(
                model, packages_df, ulds_df, device, k_value,
                econ_sort_key=econ_sort_key,
                max_pkgs=max_pkgs, max_ulds=max_ulds,
            )
        finally:
            _tr.PRIORITY_CONSOLIDATION_MIN_K = orig_min_k
            _tr._consolidate_priority_by_capacity = orig_consolidate_fn
        placements, total_unfit = packer.pack(assignment, packages_df, ulds_df)
        cost, delay, spread_cost, n_prio, up, ue = compute_packing_cost(placements, packages_df, k_value)
        candidates[label] = dict(assignment=assignment, placements=placements, cost=cost,
                                  total_unfit=total_unfit, n_prio=n_prio, up=up, ue=ue)

    priority_candidates = [('heuristic', -1, None), ('model', 10**9, None)]
    if extra_candidates:
        priority_candidates.append(
            ('heuristic_plus_one', -1, _consolidate_priority_with_extra_uld))

    for prio_label, forced_min_k, consolidate_fn in priority_candidates:
        for econ_key in econ_sort_keys:
            _run(f'{prio_label}+{econ_key}', forced_min_k, consolidate_fn, econ_key)

    chosen = min(candidates, key=lambda label: candidates[label]['cost'])
    c = candidates[chosen]
    return c['assignment'], c['placements'], c['cost'], c['total_unfit'], chosen
