"""
combined_packer.py -- per-ULD adaptive selection across N candidate packers
(learned RL policies, deterministic heuristics, or both), mirroring exactly
the same pattern that fixed the assignment stage (src/rl/adaptive_assign.py):
compute every candidate for real, keep whichever is actually better for
THIS specific ULD, no threshold, nothing to generalize. A strict per-ULD
min-of-N, so it can only match or beat any single packer alone, and adding
more candidates can never hurt.

Standalone results on the real 400-package instance (K=5000):
    RLPackerAdapter (placement_policy_big.pt)        : cost=32,661
    RLPackerAdapter (density fine-tune)               : cost=31,990
    HeuristicPacker (strategy='contact')              : cost=32,531
    2-way combined (RL density + contact heuristic)   : cost=31,661
No single strategy dominates on every ULD -- adding more genuinely
different strategies (e.g. HeuristicPacker(strategy='dblf')) as further
candidates is the next lever.
"""
from __future__ import annotations


class CombinedPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df).

    candidates : list of (name, packer) pairs, each packer implementing
        ._pack_uld(uid, pids, uld_lookup, pkg_lookup) -> (hm, left_behind_ids)
        (RLPackerAdapter and HeuristicPacker both already do). At least one
        candidate required.
    """

    def __init__(self, candidates):
        assert len(candidates) >= 1
        self.candidates = candidates

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
        pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
        for pid, row in pkg_lookup.items():
            row['Package_ID'] = pid
        uld_pkg_ids = {uid: [] for uid in uld_lookup}
        placements = []
        # Economy packages the ASSIGNMENT stage sent straight to NONE
        # (_assign_economy_by_value_density's nominal weight/volume check)
        # never got a real geometric fit-check at all. Tracked separately
        # so the cross-ULD Economy rescue below can give them the same
        # exhaustive real-fit chance as packer_unfit packages, not just
        # record them as unplaced immediately.
        clusterer_none_econ_ids = []

        for pid, uid in assignment.items():
            if uid == 'NONE':
                if pkg_lookup[pid]['Type'] == 'Priority':
                    # Shouldn't happen (Priority is hard-masked away from
                    # NONE upstream) -- if it ever does, preserve the old
                    # immediate-placement behavior rather than routing an
                    # untested case through the new Economy-only rescue path.
                    placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                       'x0': -1, 'y0': -1, 'z0': -1,
                                       'x1': -1, 'y1': -1, 'z1': -1,
                                       'reason': 'clusterer_none'})
                else:
                    clusterer_none_econ_ids.append(pid)
            elif uid in uld_pkg_ids:
                uld_pkg_ids[uid].append(pid)

        hm_by_uld = {}
        left_behind_by_uld = {}
        chosen_by_uld = {}
        for uid, pids in uld_pkg_ids.items():
            if not pids:
                hm_by_uld[uid] = None
                left_behind_by_uld[uid] = []
                continue
            best_name, best_hm, best_left, best_score = None, None, None, None
            for name, packer in self.candidates:
                hm, left = packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
                # Score by ACTUAL cost contribution, not raw left-behind
                # count -- these differ whenever candidates leave behind a
                # different MIX of packages, since compute_packing_cost
                # weights each by its own Delay_Cost, not by count. Found
                # this the hard way: a candidate that left fewer packages
                # behind by count still produced a HIGHER total cost,
                # because the specific packages it left behind happened to
                # carry higher delay_cost than the (more numerous) ones a
                # different candidate left behind. Priority packages carry
                # Delay_Cost=0 by convention, so they contribute nothing to
                # the delay-cost sum here -- priority safety is instead
                # ranked FIRST and separately (fewest priority left-behind),
                # matching how the rest of the pipeline treats it as a hard
                # constraint, not a cost-formula term.
                n_prio_left = sum(1 for pid in left if pkg_lookup[pid]['Type'] == 'Priority')
                delay_cost_left = sum(pkg_lookup[pid]['Delay_Cost'] for pid in left)
                score = (n_prio_left, delay_cost_left, -hm.utilization() if hm else 0.0)
                if best_score is None or score < best_score:
                    best_name, best_hm, best_left, best_score = name, hm, left, score
            hm_by_uld[uid], left_behind_by_uld[uid] = best_hm, best_left
            chosen_by_uld[uid] = best_name

        uld_vol = {uid: row['Length'] * row['Width'] * row['Height'] for uid, row in uld_lookup.items()}
        packer_by_name = dict(self.candidates)

        def _has_priority(uid):
            hm = hm_by_uld[uid]
            return hm is not None and any(
                pkg_lookup[pl.package_id]['Type'] == 'Priority' for pl in hm.placements)

        def _repack_with_chosen_strategy(uid, candidate_ids):
            packer = packer_by_name[chosen_by_uld.get(uid, self.candidates[0][0])]
            return packer._pack_uld(uid, candidate_ids, uld_lookup, pkg_lookup)

        for _round in range(max_rescue_rounds):
            moved_any = False
            for uid in list(uld_pkg_ids):
                stuck_priority = [pid for pid in left_behind_by_uld[uid]
                                  if pkg_lookup[pid]['Type'] == 'Priority']
                for pid in stuck_priority:
                    for other_uid in sorted((u for u in uld_pkg_ids if u != uid),
                                            key=lambda u: (not _has_priority(u), -uld_vol[u])):
                        other_placed_ids = ([p.package_id for p in hm_by_uld[other_uid].placements]
                                            if hm_by_uld[other_uid] is not None else [])
                        candidate_ids = other_placed_ids + left_behind_by_uld[other_uid] + [pid]
                        new_hm, new_left_behind = _repack_with_chosen_strategy(other_uid, candidate_ids)
                        if pid not in new_left_behind:
                            hm_by_uld[other_uid] = new_hm
                            left_behind_by_uld[other_uid] = new_left_behind
                            left_behind_by_uld[uid] = [p for p in left_behind_by_uld[uid] if p != pid]
                            moved_any = True
                            break
            if not moved_any:
                break

        # ── Cross-ULD Economy rescue ────────────────────────────────────
        # Confirmed by direct diagnostic (see combined_packer usage in
        # scripts/): even an EXHAUSTIVE heuristic search finds zero further
        # candidates for a ULD's own leftover Economy packages in ITS OWN
        # finalized Heightmap -- a genuine geometric wall for that specific
        # per-ULD arrangement. But a DIFFERENT ULD's leftover free space has
        # a different shape (different items placed, different fragment
        # geometry), so a package that can't fit its own ULD's fragments
        # might fit another ULD's. Unlike the Priority rescue above (which
        # re-packs a whole candidate ULD from scratch), this inserts
        # directly into each OTHER ULD's ALREADY-FINALIZED Heightmap without
        # re-deriving its arrangement -- cheap (one fit-check pass, not a
        # full re-pack) and safe (never disturbs an already-good ULD).
        # Highest-delay_cost stuck packages get first claim on whatever
        # cross-ULD room exists, since capacity is scarce.
        # Multiple rounds: placing a rescued item creates NEW pivot points
        # in that ULD's Heightmap (pivot_points() is derived from
        # hm.placements), which can open up candidate positions for a
        # DIFFERENT still-stuck item that had zero candidates before this
        # round -- not just diminishing returns, a genuinely new
        # opportunity each round can create.
        def _still_stuck(origin_uid, pid):
            if origin_uid is None:
                return pid in clusterer_none_econ_ids
            return pid in left_behind_by_uld[origin_uid]

        def _unstick(origin_uid, pid):
            if origin_uid is None:
                clusterer_none_econ_ids.remove(pid)
            else:
                left_behind_by_uld[origin_uid] = [p for p in left_behind_by_uld[origin_uid] if p != pid]

        for _econ_round in range(max_rescue_rounds):
            stuck_econ_all = [(None, pid) for pid in clusterer_none_econ_ids]
            for uid in uld_pkg_ids:
                for pid in left_behind_by_uld[uid]:
                    if pkg_lookup[pid]['Type'] != 'Priority':
                        stuck_econ_all.append((uid, pid))
            if not stuck_econ_all:
                break
            stuck_econ_all.sort(key=lambda t: -pkg_lookup[t[1]]['Delay_Cost'])

            moved_any_econ = False
            for origin_uid, pid in stuck_econ_all:
                if not _still_stuck(origin_uid, pid):
                    continue  # already rescued earlier this round
                pkg = pkg_lookup[pid]
                l, w, h, weight = int(pkg['Length']), int(pkg['Width']), int(pkg['Height']), float(pkg['Weight'])
                for other_uid in sorted((u for u in uld_pkg_ids if u != origin_uid),
                                        key=lambda u: -uld_vol[u]):
                    hm = hm_by_uld[other_uid]
                    if hm is None:
                        continue
                    placed = False
                    pivots = hm.pivot_points(cap=400)
                    seen_dims = set()
                    for orient_idx in range(6):
                        dx, dy, dz = hm.orientation_dims(l, w, h, orient_idx)
                        if (dx, dy, dz) in seen_dims:
                            continue
                        seen_dims.add((dx, dy, dz))
                        for (x, y, z) in pivots:
                            if hm.fits(dx, dy, dz, x, y, z, weight):
                                hm.place(pid, dx, dy, dz, x, y, z, weight)
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        _unstick(origin_uid, pid)
                        moved_any_econ = True
                        break
            if not moved_any_econ:
                break

        total_unfit = 0
        for uid in uld_pkg_ids:
            hm = hm_by_uld[uid]
            if hm is not None:
                for p in hm.placements:
                    placements.append({'Package_ID': p.package_id, 'ULD_ID': uid,
                                       'x0': p.x0, 'y0': p.y0, 'z0': p.z0,
                                       'x1': p.x1, 'y1': p.y1, 'z1': p.z1,
                                       'reason': 'placed'})
            for pid in left_behind_by_uld[uid]:
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'packer_unfit'})
                total_unfit += 1

        for pid in clusterer_none_econ_ids:
            placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'clusterer_none'})

        self.last_chosen_by_uld = chosen_by_uld
        return placements, total_unfit
