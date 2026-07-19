"""
mixed_priority_packer.py -- Packer-interface-compatible wrapper implementing
Option 1 from the user's proposal: pack Priority and Economy TOGETHER in
one unified greedy pass per ULD (not Priority-strictly-first), falling
back to the guaranteed-Priority-first HeuristicPacker._pack_uld only if
the mixed pass would drop a Priority package.

Rationale: Priority carries Delay_Cost=0, so the existing strict
Priority-first convention gives it first claim on the best-scoring
position at EVERY step purely because of its type label, never because
it's the geometrically better choice at that step. A joint greedy pass
lets whichever package (either type) best fits the current best slot go
first -- potentially packing tighter overall -- but isn't guaranteed to
leave room for every Priority package. Verified after the fact per ULD:
if the mixed pass placed all Priority packages, keep it (it can only be
as good or better in that case, since it's a strictly more free
selection); otherwise fall back to the safe, Priority-first result.
"""
from __future__ import annotations


class MixedPriorityPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df).

    Wraps a HeuristicPacker instance, using its _pack_uld_mixed +
    _pack_uld methods per ULD.
    """

    def __init__(self, heuristic_packer):
        self._packer = heuristic_packer
        self._geometry = heuristic_packer._geometry

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        mixed_result = self._packer._pack_uld_mixed(uid, pids, uld_lookup, pkg_lookup)
        if mixed_result is not None:
            return mixed_result
        return self._packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        """Same overall structure as HeuristicPacker.pack() (chained
        per-ULD pack + cross-ULD eviction-rescue for stuck Priority), using
        this class's own _pack_uld (mixed-with-fallback) per ULD."""
        uld_lookup = {row['ULD_ID']: row for _, row in ulds_df.iterrows()}
        pkg_lookup = packages_df.set_index('Package_ID').to_dict('index')
        for pid, row in pkg_lookup.items():
            row['Package_ID'] = pid
        uld_pkg_ids = {uid: [] for uid in uld_lookup}
        placements = []

        for pid, uid in assignment.items():
            if uid == 'NONE':
                placements.append({'Package_ID': pid, 'ULD_ID': 'NONE',
                                   'x0': -1, 'y0': -1, 'z0': -1,
                                   'x1': -1, 'y1': -1, 'z1': -1,
                                   'reason': 'clusterer_none'})
            elif uid in uld_pkg_ids:
                uld_pkg_ids[uid].append(pid)

        hm_by_uld = {}
        left_behind_by_uld = {}
        for uid, pids in uld_pkg_ids.items():
            if not pids:
                hm_by_uld[uid] = None
                left_behind_by_uld[uid] = []
                continue
            hm_by_uld[uid], left_behind_by_uld[uid] = self._pack_uld(uid, pids, uld_lookup, pkg_lookup)

        uld_vol = {uid: row['Length'] * row['Width'] * row['Height'] for uid, row in uld_lookup.items()}

        def _has_priority(uid):
            hm = hm_by_uld[uid]
            return hm is not None and any(
                pkg_lookup[pl.package_id]['Type'] == 'Priority' for pl in hm.placements)

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
                        new_hm, new_left_behind = self._pack_uld(other_uid, candidate_ids, uld_lookup, pkg_lookup)
                        if pid not in new_left_behind:
                            hm_by_uld[other_uid] = new_hm
                            left_behind_by_uld[other_uid] = new_left_behind
                            left_behind_by_uld[uid] = [p for p in left_behind_by_uld[uid] if p != pid]
                            moved_any = True
                            break
            if not moved_any:
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

        return placements, total_unfit
