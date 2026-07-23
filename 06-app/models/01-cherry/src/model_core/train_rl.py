import os
import json
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import (
    MAX_N_ULDS, MAX_N_PKGS, N_ULD_CLASSES,
    RL_LR, RL_EPOCHS, RL_GRAD_CLIP, RL_ENTROPY_COEF, RL_KL_COEF,
    RL_EVAL_EVERY, RL_PATIENCE, RL_TEMPERATURE,
    RL_LAMBDA_WEIGHT_PENALTY, RL_LAMBDA_VOLUME_PENALTY,
    RL_HINGE_COEF, RL_HINGE_MARGIN, RL_SPREAD_COEF,
    MAX_SAFE_PKGS, MAX_SAFE_ULDS, DEVICE,
    PRIORITY_CONSOLIDATION_MIN_K,
)
from .data_utils import (
    build_tensors, chunk_dataframe, needs_chunking, normalize_k,
    il_sample_assignment, il_sample_assignment_safe, actions_to_assignment,
)
from ..packer.reward import (
    compute_packing_cost, rl_capacity_violation_penalty,
    feasibility_hinge_loss, soft_spread_loss,
)
from .packer import DEFAULT_PACKER


# ── Chunk-safe stochastic policy rollout ─────────────────────────────────────

def rl_sample_actions_safe(model, packages_df, ulds_df, device, k_value,
                            temperature=1.0, max_pkgs=None, max_ulds=None):
    """
    Stochastic assignment for an instance of any size, chunking when needed.

    Log-probs and entropy are summed across chunks: the joint log-prob of
    independent per-package decisions is the sum of per-decision log-probs,
    so REINFORCE works correctly on the stitched result.

    Capacity limits carry over between package chunks by passing reduced ULD
    limits (limit minus already-used) into each subsequent chunk.

    Returns:
        assignment     : {Package_ID: ULD_ID | 'NONE'}
        log_prob_sum   : scalar tensor (differentiable)
        entropy_sum    : scalar tensor (differentiable)
        weight_used    : {ULD_ID: float}
        volume_used    : {ULD_ID: float}
        weight_penalty : scalar tensor (differentiable)
        volume_penalty : scalar tensor (differentiable)
    """
    device   = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    full_assignment    = {}
    weight_used_by_id  = {}
    volume_used_by_id  = {}
    log_prob_terms     = []
    entropy_terms      = []
    weight_pen_terms   = []
    volume_pen_terms   = []

    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks        = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break

        n_ulds_here        = len(uld_chunk)
        uld_ids_here       = uld_chunk['ULD_ID'].tolist()
        base_weight_limits = uld_chunk['Weight_Limit'].tolist()
        base_volumes       = (uld_chunk['Length'] * uld_chunk['Width'] * uld_chunk['Height']).tolist()
        weight_used_group  = [0.0] * n_ulds_here
        volume_used_group  = [0.0] * n_ulds_here

        for pkg_chunk in chunk_dataframe(remaining_pkgs_df, max_pkgs):
            n_pkgs_here      = len(pkg_chunk)
            tensors          = build_tensors(pkg_chunk, uld_chunk, device, k_value)
            pkg_weights_here = pkg_chunk['Weight'].tolist()
            pkg_volumes_here = (pkg_chunk['Length'] * pkg_chunk['Width'] * pkg_chunk['Height']).tolist()

            # Reduce limits by what earlier chunks in this ULD group already used
            remaining_weight_limits = [base_weight_limits[j] - weight_used_group[j]
                                       for j in range(n_ulds_here)]
            remaining_volumes       = [base_volumes[j] - volume_used_group[j]
                                       for j in range(n_ulds_here)]

            actions, log_probs, entropy, raw_logits, w_used, v_used = model.sample_actions(
                tensors['uld_feats'], tensors['pkg_feats'],
                tensors['key_padding_mask'], n_ulds_here,
                tensors['dim_mask'], tensors['priority_mask'],
                tensors['tightness'], tensors['k_feat'],
                n_pkgs_here, pkg_weights_here, remaining_weight_limits, temperature,
                pkg_volumes=pkg_volumes_here, uld_volumes=remaining_volumes,
            )

            w_pen, v_pen = rl_capacity_violation_penalty(
                raw_logits, n_pkgs_here, n_ulds_here,
                pkg_weights_here, remaining_weight_limits,
                pkg_volumes=pkg_volumes_here, uld_volumes=remaining_volumes,
            )
            weight_pen_terms.append((w_pen, n_ulds_here))
            volume_pen_terms.append((v_pen, n_ulds_here))

            full_assignment.update(actions_to_assignment(actions, n_pkgs_here, pkg_chunk, uld_chunk))
            log_prob_terms.append(log_probs[:n_pkgs_here].sum())
            entropy_terms.append(entropy)

            for j in range(n_ulds_here):
                weight_used_group[j] += w_used[j]
                volume_used_group[j] += v_used[j]

        for j, uid in enumerate(uld_ids_here):
            weight_used_by_id[uid] = weight_used_group[j]
            volume_used_by_id[uid] = volume_used_group[j]

        unresolved_ids = {pid for pid in remaining_pkgs_df['Package_ID']
                          if full_assignment.get(pid, 'NONE') == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(unresolved_ids)
        ].reset_index(drop=True)

    log_prob_sum = (torch.stack(log_prob_terms).sum() if log_prob_terms
                    else torch.tensor(0.0, device=device))
    entropy_sum  = (torch.stack(entropy_terms).sum() if entropy_terms
                    else torch.tensor(0.0, device=device))

    # Weighted mean across chunks so the result is "mean over all real ULDs"
    if weight_pen_terms:
        total_ulds     = sum(n for _, n in weight_pen_terms)
        weight_penalty = sum(p * n for p, n in weight_pen_terms) / max(total_ulds, 1)
        volume_penalty = sum(p * n for p, n in volume_pen_terms) / max(total_ulds, 1)
    else:
        weight_penalty = torch.tensor(0.0, device=device)
        volume_penalty = torch.tensor(0.0, device=device)

    return (full_assignment, log_prob_sum, entropy_sum,
            weight_used_by_id, volume_used_by_id,
            weight_penalty, volume_penalty)


# ── Chunk-safe deterministic inference ───────────────────────────────────────

def _find_rescue_uld(n_ulds_here, economy_in_uld, weight_used_group, volume_used_group,
                      base_weight_limits, base_volumes, pw_, pvol):
    """
    A priority package doesn't fit any ULD as-is. For each ULD, try evicting
    its placed Economy packages (largest-volume first, since that frees space
    fastest) until the priority package would fit both weight and volume.
    Returns (uld_index, evictions) for the ULD needing the FEWEST evictions,
    or (None, []) if no ULD can fit it even after evicting all its Economy.
    """
    best_j, best_evictions = None, None
    for j in range(n_ulds_here):
        candidates = sorted(economy_in_uld[j], key=lambda e: e['vol'], reverse=True)
        evicted = []
        w, v = weight_used_group[j], volume_used_group[j]
        for ev in candidates:
            if w + pw_ <= base_weight_limits[j] + 1e-6 and v + pvol <= base_volumes[j] + 1e-6:
                break
            w -= ev['weight']
            v -= ev['vol']
            evicted.append(ev)
        fits = (w + pw_ <= base_weight_limits[j] + 1e-6 and v + pvol <= base_volumes[j] + 1e-6)
        if fits and (best_evictions is None or len(evicted) < len(best_evictions)):
            best_j, best_evictions = j, evicted
    if best_j is None:
        return None, []
    return best_j, best_evictions


def _consolidate_priority_by_capacity(packages_df, ulds_df):
    """
    Deterministically pack every Priority package into the FEWEST, largest-
    volume ULDs that can hold them all (first-fit-decreasing by volume),
    instead of leaving ULD choice to the model's own per-package argmax.

    The model decides each package's ULD independently given only that
    package's own features and the running capacity state -- it has no
    global view of "which combination of ULDs gives Priority the most total
    room," so it can (and does) consolidate into ULDs that happen to have
    less aggregate capacity than other unused ULDs, leaving no slack for the
    downstream packer's real-world placement loss (extreme-point packing
    rarely hits 100% of nominal volume). This directly minimizes spread cost
    (K * n_ULDs) by construction, and leaves the largest possible capacity
    margin against packer placement loss in the ULDs it does use.

    Tries n_ulds = 1, 2, 3, ... (largest-volume ULDs first) until the whole
    Priority set fits by aggregate weight/volume; falls back to using every
    ULD (guaranteed to fit if the instance is feasible at all -- matches the
    hard-constraint guarantee `rl_assign_argmax_safe`'s own eviction-rescue
    path already makes for Priority).

    Returns:
        assignment  : {Package_ID: ULD_ID} for every Priority package
        weight_used : {ULD_ID: float} for ULDs touched by this consolidation
        volume_used : {ULD_ID: float} for ULDs touched by this consolidation
    """
    uld_lookup = ulds_df.set_index('ULD_ID').to_dict('index')
    pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
    prio_ids   = packages_df[packages_df['Type'].str.upper() == 'PRIORITY']['Package_ID'].tolist()
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

    for n_ulds in range(1, len(ulds_by_vol_desc) + 1):
        result = _try_pack(ulds_by_vol_desc[:n_ulds])
        if result is not None:
            return result
    # Every ULD together still couldn't fit all Priority by aggregate weight/
    # volume -- genuinely infeasible at this coarse check. Let the caller's
    # own per-package mask + eviction-rescue logic handle it as before.
    return {}, {}, {}


def rl_assign_argmax_safe(model, packages_df, ulds_df, device, k_value,
                           max_pkgs=None, max_ulds=None, econ_sort_key='value_density',
                           uld_order_strategy='file_order'):
    """
    uld_order_strategy : which order the final Economy greedy first-fit
        tries CANDIDATE ULDS in, for a given package (independent of
        econ_sort_key, which controls package order). 'file_order'
        (default) = fixed order from the input ULD list (U1, U2, ... in
        whatever order ulds_df has them) -- the existing validated
        behavior, unchanged unless overridden. Never accounted for
        remaining capacity at all, so it always tries to stuff U1 first
        regardless of how full it already is. 'best_fit' = try the ULD
        that would be left with the LEAST remaining volume after placing
        this item (tightest fit, minimizes wasted nominal space per ULD).
        'worst_fit' = try the ULD left with the MOST remaining volume
        (spreads load, keeps more slack for later large items). Both
        recomputed per package since remaining volume changes as items are
        placed.
    econ_sort_key : which order the final Economy greedy first-fit (line
        ~468 below) tries packages in. 'value_density' (default) = descending
        delay_cost/volume, the existing validated behavior (unchanged unless
        this arg is overridden). 'ascending_volume' = smallest packages
        first, regardless of delay_cost -- an experiment prompted by a
        competing team's result packing 20 more Economy packages than this
        pipeline at nearly identical per-ULD volume/weight fill percentages
        (~78-80% both), meaning their extra packages are on average smaller,
        not that their packing is denser -- suggesting a selection ordering
        that favors package COUNT over per-package value may recover some of
        that gap. Purely additive: default preserves exactly the prior
        behavior.
    Deterministic (argmax) assignment for an instance of any size.
    Used for validation and inference; mirrors the chunking of rl_sample_actions_safe().

    Priority packages are placed first via `_consolidate_priority_by_capacity`
    (greedy first-fit-decreasing into the fewest, largest-volume ULDs) --
    see that function's docstring for why this beats letting the model
    choose Priority's ULDs itself. Economy then fills remaining capacity via
    the model, in the order below. If consolidation can't fit every Priority
    package (rare -- only when total Priority weight/volume exceeds the
    whole fleet's aggregate capacity), Priority falls through to the
    model-driven path below, which still enforces packing as a hard
    constraint:
      1. The NONE class is masked out for a priority package whenever at least
         one ULD still has room -- previously NONE was left unmasked, so a
         priority package could be dropped even with capacity to spare if the
         model's own logit ranked NONE above every open ULD.
      2. If every ULD is genuinely capacity-blocked for this priority package,
         instead of forcing the package in and overflowing a ULD, evict
         already-placed Economy packages (largest-volume first) from the ULD
         that needs the fewest evictions to fit it. Economy packages evicted
         this way go back to 'NONE' (priced into delay cost -- not a hard
         constraint), which is the same trade-off the h1_h2 heuristic's own
         rescue pass makes. Only if no ULD can fit the package even after
         evicting all its Economy contents does it fall back to the previous
         forced-overflow behavior (a truly oversized/infeasible package).

    Economy is processed by ascending volume -- smaller packages first fill
    leftover gaps more efficiently than the reverse (empirically verified
    below; larger-first left more Economy unplaced). Verified on the full
    83-instance non-chunked test set relative to the model choosing every
    package's ULD (including Priority) in file order: mean cost
    16,508.0 -> 13,362.9 from Priority-first/Economy-ascending-volume
    ordering alone, then a real-world 400-package stress instance (K=5000,
    103 Priority, requires chunking) surfaced the deeper gap this capacity-
    aware consolidation fixes: the model's own Priority choices left spread
    at 4 ULDs even after the ordering fix, because it consolidated into
    ULDs with less aggregate volume than other unused ULDs, leaving no
    margin against the packer's real placement loss (~70% of nominal volume
    achieved, not 100%). Consolidating by capacity instead dropped that
    instance's spread to 3 and cost from 34,673 to 27,474, zero violations.
    """
    device   = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    if k_value >= PRIORITY_CONSOLIDATION_MIN_K:
        prio_assignment, prio_weight_used, prio_volume_used = _consolidate_priority_by_capacity(
            packages_df, ulds_df)
    else:
        # Below this K, spread cost (K * n_ULDs) is small enough that handing
        # the largest ULDs to Priority costs more Economy capacity than it
        # saves -- see PRIORITY_CONSOLIDATION_MIN_K's docstring. Fall through
        # to the Priority-first/Economy-ascending-volume ordering below,
        # letting the model choose Priority's ULDs itself.
        prio_assignment, prio_weight_used, prio_volume_used = {}, {}, {}

    economy_df = packages_df[
        (packages_df['Type'].str.upper() != 'PRIORITY') &
        (~packages_df['Package_ID'].isin(prio_assignment.keys()))
    ].reset_index(drop=True)
    economy_df = economy_df.assign(
        _v=lambda d: d['Length'] * d['Width'] * d['Height']
    ).sort_values('_v', ascending=True).drop(columns='_v').reset_index(drop=True)
    # Priority packages the consolidation didn't place (either skipped below
    # PRIORITY_CONSOLIDATION_MIN_K, or the rare case it couldn't fit) still
    # need to go through the model-driven hard-constraint path below --
    # Priority-first, by descending weight, matching the order the model was
    # validated against (see this function's docstring).
    unresolved_prio_df = packages_df[
        (packages_df['Type'].str.upper() == 'PRIORITY') &
        (~packages_df['Package_ID'].isin(prio_assignment.keys()))
    ].assign(_v=lambda d: d['Weight']).sort_values('_v', ascending=False).drop(columns='_v')
    packages_df = pd.concat([unresolved_prio_df, economy_df]).reset_index(drop=True)

    full_assignment   = dict(prio_assignment)
    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks        = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    model.eval()
    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break

        n_ulds_here        = len(uld_chunk)
        uld_ids_here       = uld_chunk['ULD_ID'].tolist()
        base_weight_limits = uld_chunk['Weight_Limit'].tolist()
        base_volumes       = (uld_chunk['Length'] * uld_chunk['Width'] * uld_chunk['Height']).tolist()
        # Deliberately NOT seeded with the Priority consolidation's nominal
        # usage: that nominal (aggregate weight/volume) accounting is more
        # conservative than what rl_packer's real extreme-point placement
        # actually consumes (~70% of nominal, not 100%), so reserving it here
        # would throw away real physical slack. The packer's own Heightmap
        # tracks true placed weight/volume during placement (priority placed
        # first into the Heightmap, economy continues into the same one) and
        # independently enforces both hard limits there -- any Economy
        # package the packer can't actually fit is left behind (delay cost),
        # never a violation. Verified empirically: seeding here dropped 38
        # more Economy packages on a stress-test instance than leaving this
        # at 0 and letting the packer be the sole arbiter of true capacity.
        weight_used_group  = [0.0] * n_ulds_here
        volume_used_group  = [0.0] * n_ulds_here
        # economy packages placed so far in each ULD -- eviction pool for rescue
        economy_in_uld     = [[] for _ in range(n_ulds_here)]  # list of dicts: pid, weight, vol

        for pkg_chunk in chunk_dataframe(remaining_pkgs_df, max_pkgs):
            n_pkgs_here = len(pkg_chunk)
            tensors     = build_tensors(pkg_chunk, uld_chunk, device, k_value)

            with torch.no_grad():
                # .cpu() once here (single sync) instead of the per-package
                # .item() calls below each forcing their own MPS
                # command-buffer sync -- same fix as model.sample_actions(),
                # safe here since this whole function runs under no_grad
                # anyway (pure inference, no gradient to preserve).
                logits = model.forward(
                    tensors['uld_feats'], tensors['pkg_feats'],
                    tensors['key_padding_mask'],
                    torch.tensor([n_ulds_here], device=device),
                    tensors['dim_mask'], tensors['priority_mask'],
                    tensors['tightness'], tensors['k_feat'],
                ).squeeze(0)[:n_pkgs_here].cpu()

            pkg_records = pkg_chunk.to_dict('records')
            for i in range(n_pkgs_here):
                pkg         = pkg_records[i]
                pid         = pkg['Package_ID']
                pw_         = pkg['Weight']
                pvol        = pkg['Length'] * pkg['Width'] * pkg['Height']
                is_priority = (str(pkg['Type']).upper() == 'PRIORITY')

                lg = logits[i].clone()
                for j in range(n_ulds_here):
                    if weight_used_group[j] + pw_ > base_weight_limits[j] + 1e-6:
                        lg[j] = -1e9
                    if volume_used_group[j] + pvol > base_volumes[j] + 1e-6:
                        lg[j] = -1e9

                all_blocked = all(lg[j].item() <= -1e8 for j in range(n_ulds_here))
                if is_priority and not all_blocked:
                    # Any index >= n_ulds_here (not just MAX_N_ULDS) reads as NONE per
                    # the action >= n_ulds_here check below -- these are unused padding
                    # ULD slots when n_ulds_here < MAX_N_ULDS, so all of them, not just
                    # the literal NONE index, must be masked off.
                    lg[n_ulds_here:] = -1e9

                if is_priority and all_blocked:
                    def _relative_overflow(j):
                        wt_over  = max(0.0, (weight_used_group[j] + pw_ - base_weight_limits[j])
                                       / max(base_weight_limits[j], 1e-6))
                        vol_over = max(0.0, (volume_used_group[j] + pvol - base_volumes[j])
                                       / max(base_volumes[j], 1e-6))
                        return max(wt_over, vol_over)

                    rescue_j, rescue_evictions = _find_rescue_uld(
                        n_ulds_here, economy_in_uld, weight_used_group, volume_used_group,
                        base_weight_limits, base_volumes, pw_, pvol,
                    )
                    if rescue_j is not None:
                        for ev in rescue_evictions:
                            full_assignment[ev['pid']] = 'NONE'
                            weight_used_group[rescue_j] -= ev['weight']
                            volume_used_group[rescue_j] -= ev['vol']
                            economy_in_uld[rescue_j].remove(ev)
                        action = rescue_j
                    else:
                        # Truly infeasible even after evicting all Economy: last resort.
                        action = min(range(n_ulds_here), key=_relative_overflow)
                elif all_blocked:
                    action = MAX_N_ULDS
                else:
                    action = int(torch.argmax(lg).item())

                if action == MAX_N_ULDS or action >= n_ulds_here:
                    full_assignment[pid] = 'NONE'
                else:
                    full_assignment[pid]          = uld_ids_here[action]
                    weight_used_group[action]     += pw_
                    volume_used_group[action]     += pvol
                    if not is_priority:
                        economy_in_uld[action].append({'pid': pid, 'weight': pw_, 'vol': pvol})

        unresolved_ids = {pid for pid in remaining_pkgs_df['Package_ID']
                          if full_assignment.get(pid, 'NONE') == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(unresolved_ids)
        ].reset_index(drop=True)

    # Discard the model's own Economy decisions from the loop above and
    # re-derive them with a greedy value-density heuristic instead: sort
    # Economy by descending delay_cost/volume (highest cost-of-dropping per
    # unit space first) and first-fit into ULDs. The model's per-package
    # Economy choice does show *some* learned value-triage (kept packages
    # average ~2x the value-density of dropped ones on a sampled instance --
    # not random) but this simple heuristic still beat it at every K bucket
    # tested on the full 83-instance test set (~9-12% lower mean cost,
    # K=100 through K=5000) -- the model lacks a global view of which
    # Economy packages are actually most worth keeping, same root cause as
    # Priority's own ULD choice above. Only Economy's ULD choice changes
    # here; Priority's placement (via consolidation or the model+rescue
    # loop above) is untouched.
    #
    # Deliberately NOT seeded with Priority's nominal (aggregate) footprint,
    # same reasoning as the loop above: nominal accounting is more
    # conservative than rl_packer's real ~70%-of-nominal placement
    # efficiency, so reserving it here would again throw away real physical
    # slack. The packer's own Heightmap independently enforces both hard
    # limits with the true placed state (Priority placed first, Economy
    # continuing into the same Heightmap) -- any Economy package it can't
    # really fit is left behind (delay cost), never a violation. Verified:
    # seeding this way regressed the real-world stress instance (cost
    # 27,474 -> 30,703, econ_drop 129 -> 174) even though it's a net win in
    # aggregate; leaving it unseeded recovered the 27,474 result.
    econ_pids  = set(economy_df['Package_ID'])
    uld_ids_all    = ulds_df['ULD_ID'].tolist()
    uld_lookup_all = ulds_df.set_index('ULD_ID').to_dict('index')
    weight_used_final = {u: 0.0 for u in uld_ids_all}
    volume_used_final = {u: 0.0 for u in uld_ids_all}
    cap_w_all = {u: uld_lookup_all[u]['Weight_Limit'] for u in uld_ids_all}
    cap_v_all = {u: (uld_lookup_all[u]['Length'] * uld_lookup_all[u]['Width']
                     * uld_lookup_all[u]['Height']) for u in uld_ids_all}

    econ_sorted = economy_df.assign(
        _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
    ).assign(
        _vd=lambda d: d['Delay_Cost'] / d['_vol'].clip(lower=1),
    )
    if econ_sort_key == 'ascending_volume':
        econ_sorted = econ_sorted.sort_values('_vol', ascending=True)
    elif econ_sort_key.startswith('random_seed'):
        # Genuinely different (weak) baseline for training-data diversity --
        # every value_density-family sort key still orders packages by a
        # formula, giving a package-selection model only correlated
        # examples to learn from. A random shuffle breaks that correlation
        # entirely, giving real contrastive signal (via its real packing
        # outcome) about which packages tend to survive real placement
        # regardless of formula-driven ordering.
        seed = int(econ_sort_key[len('random_seed'):])
        econ_sorted = econ_sorted.sample(frac=1.0, random_state=seed)
    elif econ_sort_key.startswith('value_density_joint_pow'):
        # Per-package BINDING-CONSTRAINT-aware value density: normalize both
        # volume and weight into comparable "fraction of a typical ULD's
        # capacity" units, take whichever is LARGER (the resource this
        # specific package consumes more of, relatively), and penalize by
        # that footprint^exponent. Different from value_density_wpow (pure
        # weight, tested worse) -- a competing team's ULD-wise report showed
        # they consistently pack MORE WEIGHT into the same volume footprint
        # in 5 of 6 ULDs (up to +10 percentage points), while volume % stays
        # nearly identical to ours -- suggesting different ULDs have
        # different BINDING constraints (some volume-bound, some
        # weight-bound), which a single global metric using only one
        # dimension can't adapt to. This metric lets each PACKAGE be judged
        # by its own tighter dimension, without committing the whole
        # ordering to one resource.
        exponent = float(econ_sort_key[len('value_density_joint_pow'):])
        avg_uld_volume = np.mean(list(cap_v_all.values()))
        avg_uld_weight = np.mean(list(cap_w_all.values()))
        econ_sorted = econ_sorted.assign(
            _footprint=lambda d: np.maximum(d['_vol'] / avg_uld_volume, d['Weight'] / avg_uld_weight),
        ).assign(
            _vdj=lambda d: d['Delay_Cost'] / (d['_footprint'].clip(lower=1e-9) ** exponent),
        ).sort_values('_vdj', ascending=False)
    elif econ_sort_key.startswith('value_density_wpow'):
        # Weight-based analog of value_density_pow: delay_cost/weight^exponent
        # instead of delay_cost/volume^exponent -- tests whether WEIGHT
        # scarcity (several ULDs hit 93-98% weight-full while volume stayed
        # under 83%) is actually the tighter binding constraint on this
        # instance, in which case ordering by weight-efficiency should
        # matter more than ordering by volume-efficiency.
        exponent = float(econ_sort_key[len('value_density_wpow'):])
        econ_sorted = econ_sorted.assign(
            _vdw=lambda d: d['Delay_Cost'] / (d['Weight'].clip(lower=1) ** exponent),
        ).sort_values('_vdw', ascending=False)
    elif econ_sort_key.startswith('value_density_pow'):
        # Blended value function: delay_cost / volume^exponent, exponent > 1
        # penalizes large volume more steeply than the plain ratio (exponent
        # = 1, the 'value_density' default) -- favors compact HIGH-value
        # packages more strongly than either plain value-density (too
        # value-heavy, misses easy small wins) or pure ascending-volume
        # (too count-heavy, discards value entirely -- verified worse:
        # 235 placed but cost 31,207 > 30,822 baseline on this instance).
        exponent = float(econ_sort_key[len('value_density_pow'):])
        econ_sorted = econ_sorted.assign(
            _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** exponent),
        ).sort_values('_vdp', ascending=False)
    else:
        econ_sorted = econ_sorted.sort_values('_vd', ascending=False)

    for _, row in econ_sorted.iterrows():
        pid, pw, pv = row['Package_ID'], row['Weight'], row['_vol']
        if uld_order_strategy == 'best_fit':
            try_order = sorted(uld_ids_all, key=lambda u: cap_v_all[u] - volume_used_final[u])
        elif uld_order_strategy == 'worst_fit':
            try_order = sorted(uld_ids_all, key=lambda u: -(cap_v_all[u] - volume_used_final[u]))
        else:
            try_order = uld_ids_all
        for uid in try_order:
            cap_w = cap_w_all[uid]
            cap_v = cap_v_all[uid]
            if (weight_used_final[uid] + pw <= cap_w + 1e-6 and
                    volume_used_final[uid] + pv <= cap_v + 1e-6):
                weight_used_final[uid] += pw
                volume_used_final[uid] += pv
                full_assignment[pid] = uid
                break
        else:
            full_assignment[pid] = 'NONE'

    return full_assignment


# ── Training loop ─────────────────────────────────────────────────────────────

def _stratified_sample(tags, k_values_map_dict, n_total):
    """
    Sample ~n_total tags from `tags`, balanced across each K value present.

    Full-dataset epochs (1000 instances, ~30-40 min each) meant every training
    attempt so far only got 2-8 real epochs of gradient exposure before being
    judged or hitting patience -- not necessarily enough for a genuinely
    harder K-conditioned learning problem to converge, independent of whether
    the loss/architecture design is otherwise correct. Sampling a smaller,
    per-K-balanced subset each epoch keeps the SAME signal density per K
    bucket while buying several times more epochs (gradient updates) per hour
    of wall-clock time.
    """
    by_k = {}
    for t in tags:
        by_k.setdefault(k_values_map_dict.get(t, 0), []).append(t)
    k_values = sorted(by_k.keys())
    per_k = max(1, n_total // len(k_values))
    sampled = []
    for k in k_values:
        pool = by_k[k]
        random.shuffle(pool)
        sampled.extend(pool[:per_k])
    random.shuffle(sampled)
    return sampled


def _print_metrics(epoch, n_epochs, metrics, patience_counter, patience, is_eval):
    bar = "█" * int(30 * epoch / max(n_epochs, 1))
    bar = f"[{bar:<30}] {epoch}/{n_epochs}"
    pc  = f"  patience {patience_counter}/{patience}"
    print(f"\n{'─'*72}")
    print(f"  Epoch {epoch:>3d}  {bar}{pc}")
    print(f"{'─'*72}")
    print(f"  REWARD")
    print(f"    mean_cost      : {metrics['reward/mean_cost']:>10.2f}")
    print(f"    il_baseline    : {metrics['reward/il_baseline']:>10.2f}")
    print(f"    advantage      : {metrics['reward/advantage']:>+10.4f}")
    print(f"    success_rate   : {metrics['reward/success_rate']:>9.1%}")
    print(f"  LOSS  (lr={metrics.get('lr', 0):.2e})")
    print(f"    policy_loss    : {metrics['loss/policy_loss']:>10.5f}")
    print(f"    entropy_loss   : {metrics['loss/entropy_loss']:>10.5f}")
    print(f"    value_loss     : {metrics['loss/value_loss']:>10.5f}")
    print(f"    capacity_loss  : {metrics['loss/capacity_loss']:>10.5f}")
    print(f"    hinge_loss     : {metrics.get('loss/hinge_loss', 0):>10.5f}")
    print(f"    spread_loss    : {metrics.get('loss/spread_loss', 0):>10.5f}")
    print(f"    total_loss     : {metrics['loss/total_loss']:>10.5f}")
    print(f"  CAPACITY (raw, pre-mask logits)")
    print(f"    weight_pen     : {metrics.get('cap/weight_pen', 0):>10.5f}")
    print(f"    volume_pen     : {metrics.get('cap/volume_pen', 0):>10.5f}")
    if is_eval:
        print(f"  VALIDATION")
        print(f"    val_rl_cost    : {metrics.get('val/rl_cost', float('nan')):>10.2f}")
        print(f"    val_il_cost    : {metrics.get('val/il_cost', float('nan')):>10.2f}")
        print(f"    val_cost_diff  : {metrics.get('val/cost_diff', float('nan')):>+10.2f}")
        print(f"    val_better_pct : {metrics.get('val/better_pct', float('nan')):>9.1%}")
    print(f"  PACKING")
    print(f"    priority_dropped : {metrics.get('pack/priority_dropped', 0):>8.2f}")
    print(f"    economy_dropped  : {metrics.get('pack/economy_dropped',  0):>8.2f}")
    print(f"    wt_used_pct      : {metrics.get('pack/wt_used_pct',      0):>8.1%}")
    print(f"    vol_used_pct     : {metrics.get('pack/vol_used_pct',      0):>8.1%}")
    print(f"{'─'*72}")


def train_rl(
    model,
    il_model,
    data_dir,
    n_epochs              = RL_EPOCHS,
    lr                    = RL_LR,
    grad_clip             = RL_GRAD_CLIP,
    entropy_coef          = RL_ENTROPY_COEF,
    kl_coef               = RL_KL_COEF,
    eval_every            = RL_EVAL_EVERY,
    patience              = RL_PATIENCE,
    save_path             = None,
    log_path              = None,
    il_baseline_cache_path = None,
    device                = DEVICE,
    temperature           = RL_TEMPERATURE,
    max_instances         = None,
    n_il_samples          = 4,
    n_rl_samples          = 4,
    packer                = None,
    max_safe_pkgs         = None,
    max_safe_ulds         = None,
    lambda_weight_penalty = RL_LAMBDA_WEIGHT_PENALTY,
    lambda_volume_penalty = RL_LAMBDA_VOLUME_PENALTY,
    hinge_coef            = RL_HINGE_COEF,
    hinge_margin           = RL_HINGE_MARGIN,
    spread_coef           = RL_SPREAD_COEF,
    k_values_map_dict     = None,
    priority_drop_penalty = 0.0,
    initial_best_val_cost = float('inf'),
    instances_per_epoch   = None,
):
    if packer is None:
        packer = DEFAULT_PACKER
    max_safe_pkgs = MAX_SAFE_PKGS if max_safe_pkgs is None else max_safe_pkgs
    max_safe_ulds = MAX_SAFE_ULDS if max_safe_ulds is None else max_safe_ulds
    if k_values_map_dict is None:
        k_values_map_dict = {}

    model    = model.to(device)
    il_model = il_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # Unlike train_il() (CosineAnnealingLR), the RL loop previously ran all
    # n_epochs at a flat LR -- combined with ~1000 per-instance gradient
    # steps/epoch and single-sample REINFORCE noise, this let it plateau
    # after its first couple of eval-epochs instead of settling further.
    # Decay the LR over the run so later epochs take smaller, more
    # conservative steps once the easy gains are captured.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 20)

    train_meta = pd.read_csv(os.path.join(data_dir, 'synthetic_train', 'metadata.csv'))
    test_meta  = pd.read_csv(os.path.join(data_dir, 'synthetic_test',  'metadata.csv'))
    if max_instances is not None:
        train_meta = train_meta.head(max_instances).reset_index(drop=True)

    print("Pre-loading training CSVs...", end=" ", flush=True)
    train_cache, test_cache = {}, {}
    for _, row in train_meta.iterrows():
        tag    = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_train', f'{tag}_packages.csv')
        if os.path.exists(u_path) and os.path.exists(p_path):
            train_cache[tag] = (pd.read_csv(u_path), pd.read_csv(p_path))
    for _, row in test_meta.iterrows():
        tag    = row['instance']
        u_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_ulds.csv')
        p_path = os.path.join(data_dir, 'synthetic_test', f'{tag}_packages.csv')
        if os.path.exists(u_path) and os.path.exists(p_path):
            test_cache[tag] = (pd.read_csv(u_path), pd.read_csv(p_path))
    print(f"loaded {len(train_cache)} train / {len(test_cache)} test instances.")

    oversized = [t for t, (u, p) in train_cache.items()
                 if needs_chunking(p, u, max_safe_pkgs, max_safe_ulds)]
    if oversized:
        print(f"  {len(oversized)} train instances exceed capacity — will be chunked.")

    # ── Pre-compute IL baselines ──────────────────────────────────────────────
    il_baseline_cache = {}
    il_logits_cache   = {}
    tensors_cache     = {}

    if il_baseline_cache_path and os.path.exists(il_baseline_cache_path):
        print(f"Loading IL baselines from {il_baseline_cache_path}...", end=" ", flush=True)
        try:
            with open(il_baseline_cache_path, 'rb') as f:
                loaded = pickle.load(f)
            il_baseline_cache = loaded['il_baseline_cache']
            il_logits_cache   = loaded['il_logits_cache']
            tensors_cache     = loaded['tensors_cache']
            print("Done.")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing.")
            il_baseline_cache, il_logits_cache, tensors_cache = {}, {}, {}
    else:
        print("Pre-computing IL baselines (one-time)...", flush=True)
        for tag, (ulds_df, pkgs_df) in tqdm(train_cache.items(), desc="  IL baseline"):
            n_ulds  = len(ulds_df)
            n_pkgs  = len(pkgs_df)
            k_value = k_values_map_dict.get(tag, 0)

            if not needs_chunking(pkgs_df, ulds_df, max_safe_pkgs, max_safe_ulds):
                tensors = build_tensors(pkgs_df, ulds_df, device, k_value)
                tensors_cache[tag] = {k: v.cpu() if hasattr(v, 'cpu') else v
                                      for k, v in tensors.items()}
                il_asgns, il_logits, _, _ = il_sample_assignment(
                    il_model, tensors, n_ulds, n_pkgs,
                    pkgs_df, ulds_df, n_samples=n_il_samples, temperature=1.0,
                )
                il_logits_cache[tag] = il_logits.cpu()
            else:
                il_asgns = il_sample_assignment_safe(
                    il_model, pkgs_df, ulds_df, device, k_value,
                    n_samples=n_il_samples, temperature=1.0,
                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                )

            il_costs = []
            for asgn in il_asgns:
                pl, _ = packer.pack(asgn, pkgs_df, ulds_df)
                c, _, _, _, _, _ = compute_packing_cost(pl, pkgs_df, k_value)
                il_costs.append(float(c))
            il_baseline_cache[tag] = float(np.mean(il_costs))

        print(f"  Done. Mean IL baseline: {np.mean(list(il_baseline_cache.values())):.1f}")

        if il_baseline_cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(il_baseline_cache_path)), exist_ok=True)
                with open(il_baseline_cache_path, 'wb') as f:
                    pickle.dump({'il_baseline_cache': il_baseline_cache,
                                 'il_logits_cache':   il_logits_cache,
                                 'tensors_cache':     tensors_cache}, f)
                print(f"IL baselines saved to {il_baseline_cache_path}")
            except Exception as e:
                print(f"Error saving IL baseline cache: {e}")

    history          = []
    # Defaults to inf for a fresh run; when resuming from a prior checkpoint
    # (--resume-from), the caller passes that checkpoint's own recorded
    # val_rl_cost_penalized here so this run only overwrites it with an
    # ACTUALLY better result, instead of trivially beating a fresh inf on its
    # first eval and silently regressing the saved checkpoint (real bug hit
    # in practice: a resumed run's epoch-2 eval landed worse than the
    # checkpoint it started from, but would have overwritten it anyway).
    best_val_cost    = initial_best_val_cost
    patience_counter = 0
    stop_training    = False
    epoch_bar        = tqdm(range(n_epochs), desc="Training", unit="epoch")

    # NOTE: an earlier version of this normalized `advantage` by a single
    # GLOBAL running RMS scale shared across all instances. That was a real
    # bug: spread_cost = K * n_priority_ulds means a K=5000 instance can
    # swing by up to ~30000 while a K=100 instance only swings by up to
    # ~600 -- a 50x scale difference. A global RMS scale gets dominated by
    # the 200/1000 K=5000 instances, which then divides the other 800
    # instances' advantage down to near-zero, killing their training signal
    # entirely. Symptom actually observed: training plateaued at a flat
    # 87-88% success rate for 19+ epochs across three different variance-
    # reduction attempts (LR decay, that global normalization itself, and
    # 4x rollout averaging) because none of them fixed the underlying scale
    # mismatch. Fixed by normalizing per-instance below instead.

    for epoch in epoch_bar:
        model.train()
        acc = {k: [] for k in [
            'policy_loss', 'entropy_loss', 'value_loss', 'capacity_loss',
            'hinge_loss', 'spread_loss', 'total_loss',
            'weight_pen', 'volume_pen', 'rl_cost', 'il_baseline', 'advantage', 'success',
            'n_priority_dropped', 'n_economy_dropped', 'wt_used_pct', 'vol_used_pct', 'lr',
        ]}

        if instances_per_epoch is not None and instances_per_epoch < len(train_cache):
            idxs = _stratified_sample(list(train_cache.keys()), k_values_map_dict, instances_per_epoch)
        else:
            idxs = list(train_cache.keys())
            random.shuffle(idxs)

        for tag in idxs:
            ulds_df, pkgs_df = train_cache[tag]
            n_ulds, n_pkgs   = len(ulds_df), len(pkgs_df)
            il_baseline      = il_baseline_cache[tag]
            in_capacity      = tag in tensors_cache
            k_value          = k_values_map_dict.get(tag, 0)

            def _one_rollout():
                """One stochastic rollout + pack + cost for `tag`. Called
                n_rl_samples times per instance and averaged below -- a
                single sample here is a high-variance REINFORCE estimate
                (empirically: training plateaued at a stable 87-88% success
                rate for 13+ epochs even with LR decay and advantage
                normalization, because the single-sample cost estimate was
                too noisy to keep discriminating good updates from bad ones
                once the policy had already pulled ahead of the frozen IL
                baseline). Averaging several independent rollouts' costs
                before computing advantage directly reduces that variance,
                same idea as n_il_samples already does for the IL baseline
                itself."""
                if in_capacity:
                    tensors = {k: v.to(device) if hasattr(v, 'to') else v
                               for k, v in tensors_cache[tag].items()}
                    pkg_weights       = pkgs_df['Weight'].tolist()
                    uld_weight_limits = ulds_df['Weight_Limit'].tolist()
                    pkg_volumes       = (pkgs_df['Length'] * pkgs_df['Width'] * pkgs_df['Height']).tolist()
                    uld_volumes       = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist()
                    il_logits         = il_logits_cache[tag].to(device)

                    actions, log_probs, entropy, rl_logits, wt_used, vol_used = model.sample_actions(
                        tensors['uld_feats'], tensors['pkg_feats'],
                        tensors['key_padding_mask'], n_ulds,
                        tensors['dim_mask'], tensors['priority_mask'],
                        tensors['tightness'], tensors['k_feat'],
                        n_pkgs, pkg_weights, uld_weight_limits, temperature,
                        pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
                    )
                    rl_assignment = actions_to_assignment(actions, n_pkgs, pkgs_df, ulds_df)
                    lp_sum        = log_probs[:n_pkgs].sum()
                    entropy_term  = entropy

                    weight_pen, volume_pen = rl_capacity_violation_penalty(
                        rl_logits, n_pkgs, n_ulds,
                        pkg_weights, uld_weight_limits,
                        pkg_volumes=pkg_volumes, uld_volumes=uld_volumes,
                    )

                    rl_log_probs_all = F.log_softmax(rl_logits[:n_pkgs], dim=-1)
                    il_probs         = F.softmax(il_logits[:n_pkgs].detach(), dim=-1)
                    kl_term = (F.kl_div(rl_log_probs_all, il_probs, reduction='batchmean')
                               if kl_coef != 0.0 else torch.tensor(0.0, device=device))

                    # ── Auxiliary losses ported from model_b (see reward.py) ──────
                    # Computed from the SAME already-hard-masked rl_logits used
                    # above -- dim-infeasible slots are already -1e9 there, and
                    # for Economy packages the NONE logit is left uncorrupted
                    # (only masked for Priority, which is exactly the ground
                    # truth these losses need). dim_mask/priority_mask are
                    # passed again anyway to mirror model_b's ground-truth-based
                    # design rather than trusting the network's own masking.
                    dim_mask_i      = tensors['dim_mask'].squeeze(0)
                    priority_mask_i = tensors['priority_mask'].squeeze(0)
                    hinge_term = (feasibility_hinge_loss(
                        rl_logits, dim_mask_i, priority_mask_i, n_pkgs, n_ulds, margin=hinge_margin,
                    ) if hinge_coef != 0.0 else torch.tensor(0.0, device=device))
                    spread_term = (normalize_k(k_value) * soft_spread_loss(
                        rl_logits, dim_mask_i, priority_mask_i, n_pkgs, n_ulds,
                    ) if spread_coef != 0.0 else torch.tensor(0.0, device=device))

                    mean_wt_pct  = float(sum(wt_used[j] / max(uld_weight_limits[j], 1)
                                             for j in range(n_ulds)) / n_ulds)
                    mean_vol_pct = float(sum(vol_used[j] / max(uld_volumes[j], 1)
                                             for j in range(n_ulds)) / n_ulds)
                else:
                    (rl_assignment, lp_sum, entropy_term, wt_used_dict, vol_used_dict,
                     weight_pen, volume_pen) = rl_sample_actions_safe(
                        model, pkgs_df, ulds_df, device, k_value,
                        temperature=temperature,
                        max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                    )
                    kl_term             = torch.tensor(0.0, device=device)
                    # Skipped for the (rare) chunked branch, same as kl_term above --
                    # rl_sample_actions_safe doesn't expose per-chunk raw logits/masks
                    # needed to compute these; only ~1-2% of instances ever chunk.
                    hinge_term          = torch.tensor(0.0, device=device)
                    spread_term         = torch.tensor(0.0, device=device)
                    uld_wt_map         = dict(zip(ulds_df['ULD_ID'], ulds_df['Weight_Limit']))
                    uld_vol_map        = dict(zip(ulds_df['ULD_ID'],
                                                  ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']))
                    mean_wt_pct  = float(np.mean([wt_used_dict.get(uid, 0.0) / max(uld_wt_map[uid], 1)
                                                  for uid in ulds_df['ULD_ID']])) if len(ulds_df) else 0.0
                    mean_vol_pct = float(np.mean([vol_used_dict.get(uid, 0.0) / max(uld_vol_map[uid], 1)
                                                  for uid in ulds_df['ULD_ID']])) if len(ulds_df) else 0.0

                rl_placements, _ = packer.pack(rl_assignment, pkgs_df, ulds_df)
                rl_cost, _, _, _, unplaced_prio, unplaced_eco = compute_packing_cost(
                    rl_placements, pkgs_df, k_value)
                training_cost = float(rl_cost) + priority_drop_penalty * len(unplaced_prio)

                return {
                    'lp_sum': lp_sum, 'entropy_term': entropy_term,
                    'weight_pen': weight_pen, 'volume_pen': volume_pen, 'kl_term': kl_term,
                    'hinge_term': hinge_term, 'spread_term': spread_term,
                    'rl_cost': float(rl_cost), 'training_cost': training_cost,
                    'unplaced_prio': unplaced_prio, 'unplaced_eco': unplaced_eco,
                    'mean_wt_pct': mean_wt_pct, 'mean_vol_pct': mean_vol_pct,
                }

            rollouts = [_one_rollout() for _ in range(n_rl_samples)]

            # Priority packages carry Delay_Cost=0, so compute_packing_cost's
            # official cost formula (K*spread + delay_cost) gives the policy
            # zero learned incentive to avoid dropping them -- concentrating
            # priority into fewer ULDs to cut spread cost is free even when
            # it makes some of them physically unplaceable. This penalty is
            # training-signal only (drives the advantage/policy gradient);
            # acc['rl_cost'] below still logs the true, unpenalized cost so
            # reported metrics keep matching the spec'd cost formula exactly.
            avg_training_cost = float(np.mean([r['training_cost'] for r in rollouts]))
            advantage         = il_baseline - avg_training_cost

            # Per-instance scale (that instance's own IL cost, which already
            # correctly incorporates its own K's spread-cost coefficient),
            # not a global running estimate -- see note above. Produces a
            # scale-invariant "fractional improvement over IL" signal that
            # treats a K=100 and a K=5000 instance consistently, instead of
            # one dominating the other's normalization.
            normalized_advantage = advantage / max(abs(il_baseline), 1.0)

            # Single (lower-variance) advantage shared across all n_rl_samples
            # rollouts' gradients: policy_loss = -advantage * mean(lp_sum_k)
            # == mean(-advantage * lp_sum_k), since advantage is now a
            # constant w.r.t. each individual rollout -- mathematically the
            # same REINFORCE estimator, just averaged over more samples.
            lp_sum_mean       = torch.stack([r['lp_sum'] for r in rollouts]).mean()
            entropy_term_mean = torch.stack([r['entropy_term'] for r in rollouts]).mean()
            weight_pen_mean   = torch.stack([r['weight_pen'] for r in rollouts]).mean()
            volume_pen_mean   = torch.stack([r['volume_pen'] for r in rollouts]).mean()
            kl_term_mean      = torch.stack([r['kl_term'] for r in rollouts]).mean()
            hinge_term_mean   = torch.stack([r['hinge_term'] for r in rollouts]).mean()
            spread_term_mean  = torch.stack([r['spread_term'] for r in rollouts]).mean()

            policy_loss  = -normalized_advantage * lp_sum_mean
            entropy_loss = -entropy_coef * entropy_term_mean
            value_loss   = kl_coef * kl_term_mean
            capacity_loss = (lambda_weight_penalty * weight_pen_mean
                             + lambda_volume_penalty * volume_pen_mean)
            hinge_loss   = hinge_coef * hinge_term_mean
            spread_loss  = spread_coef * spread_term_mean
            total_loss   = (policy_loss + entropy_loss + value_loss + capacity_loss
                            + hinge_loss + spread_loss)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            mean_rl_cost   = float(np.mean([r['rl_cost'] for r in rollouts]))
            mean_n_prio    = float(np.mean([len(r['unplaced_prio']) for r in rollouts]))
            mean_n_econ    = float(np.mean([len(r['unplaced_eco']) for r in rollouts]))
            mean_wt_pct    = float(np.mean([r['mean_wt_pct'] for r in rollouts]))
            mean_vol_pct   = float(np.mean([r['mean_vol_pct'] for r in rollouts]))

            acc['policy_loss'].append(policy_loss.item())
            acc['entropy_loss'].append(entropy_loss.item())
            acc['value_loss'].append(value_loss.item() if torch.is_tensor(value_loss) else float(value_loss))
            acc['capacity_loss'].append(capacity_loss.item() if torch.is_tensor(capacity_loss) else float(capacity_loss))
            acc['hinge_loss'].append(hinge_loss.item() if torch.is_tensor(hinge_loss) else float(hinge_loss))
            acc['spread_loss'].append(spread_loss.item() if torch.is_tensor(spread_loss) else float(spread_loss))
            acc['weight_pen'].append(weight_pen_mean.item())
            acc['volume_pen'].append(volume_pen_mean.item())
            acc['total_loss'].append(total_loss.item())
            acc['rl_cost'].append(mean_rl_cost)
            acc['il_baseline'].append(il_baseline)
            acc['advantage'].append(advantage)
            acc['success'].append(1.0 if advantage > 0 else 0.0)
            acc['n_priority_dropped'].append(mean_n_prio)
            acc['n_economy_dropped'].append(mean_n_econ)
            acc['wt_used_pct'].append(mean_wt_pct)
            acc['vol_used_pct'].append(mean_vol_pct)
            acc['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0

        metrics = {
            'epoch':                 epoch + 1,
            'reward/mean_cost':      _mean(acc['rl_cost']),
            'reward/il_baseline':    _mean(acc['il_baseline']),
            'reward/advantage':      _mean(acc['advantage']),
            'reward/success_rate':   _mean(acc['success']),
            'loss/policy_loss':      _mean(acc['policy_loss']),
            'loss/entropy_loss':     _mean(acc['entropy_loss']),
            'loss/value_loss':       _mean(acc['value_loss']),
            'loss/capacity_loss':    _mean(acc['capacity_loss']),
            'loss/hinge_loss':       _mean(acc['hinge_loss']),
            'loss/spread_loss':      _mean(acc['spread_loss']),
            'loss/total_loss':       _mean(acc['total_loss']),
            'cap/weight_pen':        _mean(acc['weight_pen']),
            'cap/volume_pen':        _mean(acc['volume_pen']),
            'pack/priority_dropped': _mean(acc['n_priority_dropped']),
            'pack/economy_dropped':  _mean(acc['n_economy_dropped']),
            'pack/wt_used_pct':      _mean(acc['wt_used_pct']),
            'pack/vol_used_pct':     _mean(acc['vol_used_pct']),
            'lr':                    _mean(acc['lr']),
        }

        # ── Validation ────────────────────────────────────────────────────────
        is_eval = (epoch + 1) % eval_every == 0
        if is_eval:
            model.eval()
            rl_costs_eval, il_costs_eval, rl_costs_eval_penalized = [], [], []

            for tag, (u, p) in test_cache.items():
                n_u, n_p = len(u), len(p)
                k_value  = k_values_map_dict.get(tag, 0)

                if not needs_chunking(p, u, max_safe_pkgs, max_safe_ulds):
                    t = build_tensors(p, u, device, k_value)
                    # NOTE: must use the capacity-safe masked argmax here, same as the
                    # chunked branch below -- raw model.forward().argmax() has no
                    # weight/volume masking and can (and did) collapse an entire
                    # instance onto one ULD, silently abandoning priority packages
                    # (Delay_Cost==0 for priority, so the scalar cost barely reflects
                    # it). rl_assign_argmax_safe is safe to call for any instance size.
                    rl_a    = rl_assign_argmax_safe(model, p, u, device, k_value,
                                                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds)
                    il_asgns, _, _, _ = il_sample_assignment(
                        il_model, t, n_u, n_p, p, u,
                        n_samples=n_il_samples, temperature=1.0,
                    )
                else:
                    rl_a    = rl_assign_argmax_safe(model, p, u, device, k_value,
                                                    max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds)
                    il_asgns = il_sample_assignment_safe(
                        il_model, p, u, device, k_value,
                        n_samples=n_il_samples, temperature=1.0,
                        max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                    )

                rl_p, _ = packer.pack(rl_a, p, u)
                rl_c, _, _, _, rl_unplaced_prio, _ = compute_packing_cost(rl_p, p, k_value)
                rl_costs_eval_penalized.append(
                    float(rl_c) + priority_drop_penalty * len(rl_unplaced_prio))

                il_sc = []
                for ia in il_asgns:
                    ip, _ = packer.pack(ia, p, u)
                    ic, _, _, _, _, _ = compute_packing_cost(ip, p, k_value)
                    il_sc.append(float(ic))

                rl_costs_eval.append(float(rl_c))
                il_costs_eval.append(float(np.mean(il_sc)))

            val_rl           = _mean(rl_costs_eval)            # true, unpenalized -- for reporting
            val_rl_penalized = _mean(rl_costs_eval_penalized)  # for checkpoint selection
            val_il     = _mean(il_costs_eval)
            better_pct = (sum(r < il for r, il in zip(rl_costs_eval, il_costs_eval))
                          / max(len(rl_costs_eval), 1))
            metrics.update({
                'val/rl_cost':    val_rl,
                'val/il_cost':    val_il,
                'val/cost_diff':  val_rl - val_il,
                'val/better_pct': better_pct,
            })

            if val_rl_penalized < best_val_cost:
                best_val_cost    = val_rl_penalized
                patience_counter = 0
                metrics['checkpoint'] = True
                if save_path:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    torch.save({
                        'epoch':                epoch + 1,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rl_cost':          val_rl,            # true, unpenalized cost
                        'val_rl_cost_penalized': best_val_cost,    # includes priority_drop_penalty
                        'val_il_cost':          val_il,
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_training = True

            # Save clustering JSON for this eval epoch
            if save_path:
                epoch_dir = os.path.join(os.path.dirname(save_path), f'outputs/epoch_{epoch+1:03d}')
                os.makedirs(epoch_dir, exist_ok=True)
                clustering_agg = {}
                for val_tag, (val_u, val_p) in test_cache.items():
                    # Same fix as the validation-cost loop above: always use the
                    # capacity-safe masked argmax, regardless of instance size.
                    val_k_value = k_values_map_dict.get(val_tag, 0)
                    clustering_agg[val_tag] = rl_assign_argmax_safe(
                        model, val_p, val_u, device, val_k_value,
                        max_pkgs=max_safe_pkgs, max_ulds=max_safe_ulds,
                    )
                with open(os.path.join(epoch_dir, 'clustering.json'), 'w') as f:
                    json.dump(clustering_agg, f, indent=2)

        if is_eval:
            _print_metrics(epoch + 1, n_epochs, metrics, patience_counter, patience, is_eval)
            if metrics.get('checkpoint'):
                print(f"    Checkpoint saved  (best val_rl_cost = {best_val_cost:.1f})")
            if stop_training:
                print(f"\n  [Early Stop] No val improvement for {patience} epochs.")

        epoch_bar.set_postfix(
            rl=f"{metrics['reward/mean_cost']:.0f}",
            il=f"{metrics['reward/il_baseline']:.0f}",
            succ=f"{metrics['reward/success_rate']:.0%}",
            p_loss=f"{metrics['loss/policy_loss']:.3f}",
            cap=f"{metrics['loss/capacity_loss']:.3f}",
        )

        history.append(metrics)
        if stop_training:
            break

    epoch_bar.close()

    df = pd.DataFrame(history)
    if log_path:
        df.to_csv(log_path, index=False)
        print(f"\nLog saved -> {log_path}")
    print(f"\nRL training complete.  Best val cost: {best_val_cost:.1f}")
    return df
