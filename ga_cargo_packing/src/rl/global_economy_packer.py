"""
global_economy_packer.py -- replaces the two-stage "nominally assign
Economy per-ULD by value-density, THEN try to physically pack it" process
with ONE integrated, real-fit-driven greedy selection across the WHOLE
fleet at once.

Why: exhaustively confirmed (combined_packer.py, cross-ULD rescue covering
both packer_unfit AND clusterer_none packages) that the packing LAYER is at
its ceiling for whatever Economy packages the ASSIGNMENT stage already
selected -- multiple placement algorithms (RL, two heuristic strategies)
plus cross-ULD rescue all independently hit zero further real candidates.
The remaining gap isn't a placement-quality problem anymore, it's a
selection problem: train_rl.py's _assign_economy_by_value_density decides
which Economy packages compete for space using NOMINAL (aggregate) weight/
volume accounting, per ULD independently, greedily in isolation from real
geometry -- so it can "accept" a package that fits nominally but not
really, wasting that package's slot in the value-density order that a
DIFFERENT, still-available package could have real-fit into instead.

Fix: skip the nominal accounting stage entirely for Economy. Pack Priority
first (per ULD, same as before -- trusted, already near-optimal). Then
collect ALL Economy packages across the WHOLE instance (not pre-assigned to
any particular ULD), sort by descending delay_cost/volume (same value-
density principle as before, just applied globally instead of per-ULD),
and greedily place each one into whichever ULD's CURRENT real Heightmap
actually accepts it (first ULD, by remaining nominal volume, that a real
fit-check confirms) -- exactly mirroring combined_packer.py's cross-ULD
rescue logic, just run as the PRIMARY selection process instead of a
patch-up pass afterward.

Does not modify train_rl.py, reward.py, adaptive_assign.py, or
combined_packer.py -- a new, standalone Packer-interface-compatible class.
Reuses CombinedPacker's own Priority-strategy comparison for the Priority
phase (best of RL/heuristic candidates, same as combined_packer.py).
"""
from __future__ import annotations

import pandas as pd

from .heuristic_packer import _face_contact_area


class GlobalEconomyPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df).

    priority_candidates : list of (name, packer) pairs used ONLY for the
        Priority phase (same per-ULD best-of-N comparison as
        CombinedPacker), e.g. [('rl', RLPackerAdapter(...)),
        ('contact', HeuristicPacker(strategy='contact')), ...].
    fit_check_packer : any packer exposing .geometry-style Heightmap
        methods (HeuristicPacker instance) -- used for the global Economy
        fit-check/placement machinery (pivot_points/orientation_dims/fits/
        place). Any HeuristicPacker instance works regardless of its own
        `strategy`, since only its Heightmap access is used here, not its
        _greedy_pack_into selection criterion.
    """

    def __init__(self, priority_candidates, fit_check_packer):
        assert len(priority_candidates) >= 1
        self.priority_candidates = priority_candidates
        self.fit_check_packer = fit_check_packer

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
        pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
        for pid, row in pkg_lookup.items():
            row['Package_ID'] = pid

        # ── Priority phase: same per-ULD best-of-N as CombinedPacker ──────
        uld_priority_ids = {uid: [] for uid in uld_lookup}
        for pid, uid in assignment.items():
            if uid != 'NONE' and uid in uld_lookup and pkg_lookup[pid]['Type'] == 'Priority':
                uld_priority_ids[uid].append(pid)

        hm_by_uld = {}
        prio_left_behind_by_uld = {}
        for uid, pids in uld_priority_ids.items():
            if not pids:
                hm_by_uld[uid] = None
                prio_left_behind_by_uld[uid] = []
                continue
            best_hm, best_left, best_score = None, None, None
            for name, packer in self.priority_candidates:
                priority_df = pd.DataFrame([pkg_lookup[pid] for pid in pids])
                hm, left = packer._pack_uld(uid, pids, uld_lookup,
                                             {pid: pkg_lookup[pid] for pid in pids})
                score = (len(left), -hm.utilization() if hm else 0.0)
                if best_score is None or score < best_score:
                    best_hm, best_left, best_score = hm, left, score
            hm_by_uld[uid], prio_left_behind_by_uld[uid] = best_hm, best_left

        # Cross-ULD Priority rescue -- same logic as combined_packer.py,
        # simplified (single candidate strategy per ULD already fixed above;
        # re-pack with whichever packer produced that ULD's chosen hm is
        # unnecessary here since we insert directly like the Economy phase does).
        uld_vol = {uid: row['Length'] * row['Width'] * row['Height'] for uid, row in uld_lookup.items()}
        for _round in range(max_rescue_rounds):
            moved_any = False
            for uid in list(uld_priority_ids):
                stuck = list(prio_left_behind_by_uld[uid])
                for pid in stuck:
                    pkg = pkg_lookup[pid]
                    l, w, h, weight = int(pkg['Length']), int(pkg['Width']), int(pkg['Height']), float(pkg['Weight'])
                    for other_uid in sorted((u for u in uld_priority_ids if u != uid),
                                            key=lambda u: -uld_vol[u]):
                        hm = hm_by_uld[other_uid]
                        if hm is None:
                            continue
                        if self._try_place(hm, pid, l, w, h, weight):
                            prio_left_behind_by_uld[uid].remove(pid)
                            moved_any = True
                            break
            if not moved_any:
                break

        # ── Economy phase: GLOBAL greedy, real-fit-driven ──────────────────
        priority_ids_all = set(pid for pids in uld_priority_ids.values() for pid in pids)
        economy_ids_all = [pid for pid in packages_df['Package_ID'] if pid not in priority_ids_all]
        economy_sorted = sorted(
            economy_ids_all,
            key=lambda pid: -(pkg_lookup[pid]['Delay_Cost']
                               / max(pkg_lookup[pid]['Length'] * pkg_lookup[pid]['Width'] * pkg_lookup[pid]['Height'], 1)),
        )

        econ_left_behind = []
        for pid in economy_sorted:
            pkg = pkg_lookup[pid]
            l, w, h, weight = int(pkg['Length']), int(pkg['Width']), int(pkg['Height']), float(pkg['Weight'])
            for uid in uld_lookup:
                if hm_by_uld.get(uid) is None:
                    hm_by_uld[uid] = self.fit_check_packer._geometry.Heightmap(
                        length=int(uld_lookup[uid]['Length']), width=int(uld_lookup[uid]['Width']),
                        height=int(uld_lookup[uid]['Height']), weight_limit=float(uld_lookup[uid]['Weight_Limit']),
                    )
            # Best-fit across the WHOLE fleet by contact area, not just the
            # first ULD (by size) that happens to fit -- a package that fits
            # several ULDs should go wherever it hugs tightest, consistent
            # with the same fragmentation-avoidance principle used
            # throughout (heuristic_packer.py, train_placement_density_
            # finetune.py), now applied to WHICH ULD too, not just where
            # within one ULD.
            best_uid, best_cand = None, None  # best_cand = (contact, dx,dy,dz,x,y,z)
            for uid in uld_lookup:
                hm = hm_by_uld[uid]
                cand = self._find_best_candidate(hm, l, w, h, weight)
                if cand is not None and (best_cand is None or cand[0] > best_cand[0]):
                    best_uid, best_cand = uid, cand
            if best_cand is not None:
                contact, dx, dy, dz, x, y, z = best_cand
                hm_by_uld[best_uid].place(pid, dx, dy, dz, x, y, z, weight)
            else:
                econ_left_behind.append(pid)

        placements = []
        for uid, hm in hm_by_uld.items():
            if hm is not None:
                for p in hm.placements:
                    placements.append({'Package_ID': p.package_id, 'ULD_ID': uid,
                                       'x0': p.x0, 'y0': p.y0, 'z0': p.z0,
                                       'x1': p.x1, 'y1': p.y1, 'z1': p.z1,
                                       'reason': 'placed'})
        total_unfit = 0
        for uid, pids in prio_left_behind_by_uld.items():
            for pid in pids:
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'packer_unfit'})
                total_unfit += 1
        for pid in econ_left_behind:
            placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                               'x0': -1, 'y0': -1, 'z0': -1,
                               'x1': -1, 'y1': -1, 'z1': -1,
                               'reason': 'packer_unfit'})
            total_unfit += 1

        return placements, total_unfit

    @staticmethod
    def _try_place(hm, pid, l, w, h, weight):
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
                    return True
        return False

    @staticmethod
    def _find_best_candidate(hm, l, w, h, weight):
        """Non-mutating: returns (contact_area, dx,dy,dz,x,y,z) for the
        highest-contact-area valid placement of this item in hm, or None if
        it doesn't fit anywhere in hm at all."""
        best = None
        pivots = hm.pivot_points(cap=400)
        seen_dims = set()
        for orient_idx in range(6):
            dx, dy, dz = hm.orientation_dims(l, w, h, orient_idx)
            if (dx, dy, dz) in seen_dims:
                continue
            seen_dims.add((dx, dy, dz))
            for (x, y, z) in pivots:
                if hm.fits(dx, dy, dz, x, y, z, weight):
                    contact = _face_contact_area(hm, x, y, z, dx, dy, dz)
                    if best is None or contact > best[0]:
                        best = (contact, dx, dy, dz, x, y, z)
        return best
