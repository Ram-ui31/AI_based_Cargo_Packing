"""
heuristic_packer.py -- a deterministic, non-learned 3D placement packer:
at every step, exhaustively evaluates every valid (item, pivot-point,
orientation) candidate via rl_packer's own geometry.py (Heightmap.fits(),
pivot_points()), and greedily places whichever candidate maximizes contact
area with the ULD's walls/floor/ceiling and already-placed boxes.

Why: train_placement_density_finetune.py showed a contact-area REWARD BONUS
(a soft nudge during RL training) gave a real, verified 2.1% improvement on
the real 400-package instance (32,661 -> 31,990) by reducing fragmentation.
A direct, exhaustive greedy maximization of that same objective -- not
filtered through a learned, imperfect policy -- should do at least as well,
and is a genuinely different algorithm from the RL policy (deterministic
search vs. learned sequential decisions), making it a good candidate for
adaptive selection alongside RLPackerAdapter: mirrors exactly the
heuristic-vs-model pattern that fixed the assignment stage
(src/rl/adaptive_assign.py) -- compute both, keep whichever is actually
cheaper per ULD, no training or external dependency needed for this
candidate at all.

Same chained Priority-then-Economy structure as RLPackerAdapter._pack_uld
(Priority gets first claim on every position, Economy continues into the
SAME Heightmap) and the same cross-ULD eviction-rescue loop, so this is a
drop-in Packer-interface-compatible alternative, safe to swap in anywhere
RLPackerAdapter is used.
"""
from __future__ import annotations

import os
import sys

import pandas as pd


def _import_geometry(rl_packer_src):
    if rl_packer_src not in sys.path:
        sys.path.insert(0, rl_packer_src)
    import importlib
    saved = sys.modules.pop('geometry', None)
    try:
        geometry = importlib.import_module('geometry')
    finally:
        if saved is not None:
            sys.modules['geometry'] = saved
        else:
            sys.modules.pop('geometry', None)
    return geometry


def _face_contact_area(hm, x, y, z, dx, dy, dz):
    """Same fragmentation-avoidance proxy as
    rl_packer/training/train_placement_density_finetune.py's reward shaping
    -- total surface area touching a wall/floor/ceiling or an existing
    box's adjacent face."""
    x1, y1, z1 = x + dx, y + dy, z + dz
    area = 0.0
    if x == 0: area += dy * dz
    if y == 0: area += dx * dz
    if z == 0: area += dx * dy
    if x1 == hm.length: area += dy * dz
    if y1 == hm.width:  area += dx * dz
    if z1 == hm.height: area += dx * dy
    for p in hm.placements:
        if x == p.x1 or x1 == p.x0:
            oy = max(0, min(y1, p.y1) - max(y, p.y0))
            oz = max(0, min(z1, p.z1) - max(z, p.z0))
            area += oy * oz
        if y == p.y1 or y1 == p.y0:
            ox = max(0, min(x1, p.x1) - max(x, p.x0))
            oz = max(0, min(z1, p.z1) - max(z, p.z0))
            area += ox * oz
        if z == p.z1 or z1 == p.z0:
            ox = max(0, min(x1, p.x1) - max(x, p.x0))
            oy = max(0, min(y1, p.y1) - max(y, p.y0))
            area += ox * oy
    return area


class HeuristicPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df).

    strategy : 'contact' (default) -- greedily maximize contact area (hug
        walls/floor/existing boxes tightly, consolidates leftover space).
        'dblf' -- Deepest-Bottom-Left-Fill: greedily MINIMIZE (z, y, x)
        lexicographically (prefer the lowest, most back-left-corner valid
        position), ties broken by max contact area. A genuinely different
        placement philosophy (systematic corner-filling vs tightness-
        maximizing) -- can produce a different final arrangement, and
        therefore rescue a different subset of packages, than 'contact'
        does, which is why both are offered as separate candidates for
        combined_packer.py's per-ULD adaptive selection rather than picking
        just one.
        'min_envelope' -- greedily MINIMIZE the growth of the smallest
        axis-aligned bounding envelope containing all placed items so far
        (tie-broken by max contact area). Both 'contact' and 'dblf' score
        candidates by LOCAL properties (touching faces, or corner
        position) -- neither directly optimizes overall compactness.
        'min_envelope' is a genuinely different, GLOBAL criterion: it
        directly penalizes any placement that pushes the overall occupied
        volume's bounding box outward, which locally-tight-fitting
        placements can still do (e.g. a small item that hugs a wall but
        extends the envelope in the one dimension nothing else reaches
        yet). A real candidate for combined_packer.py's per-ULD adaptive
        selection precisely because it can rescue a different subset of
        packages than either of the other two.
    """

    def __init__(self, rl_packer_src=None, max_pivots=400, strategy='contact',
                 epsilon=0.0, rng=None, origin_source='pivot'):
        """
        origin_source : 'pivot' (default, unchanged behavior) or 'ems'.
            'pivot' uses Heightmap.pivot_points() -- corner-adjacent points
            of already-placed boxes only, structurally blind to gaps that
            don't touch any existing box's corner. 'ems' uses
            Heightmap.ems_candidates() -- Empty-Maximal-Space decomposition,
            a complete representation of remaining placement opportunities.
            Only the candidate-ORIGIN source changes; scoring (contact/
            dblf/min_envelope) is untouched, isolating this as the only
            variable for the pivot-vs-EMS comparison in
            scripts/test_ems_packer.py.
        epsilon, rng : epsilon-greedy exploration for the per-step greedy
            choice below (default epsilon=0.0 -- fully deterministic,
            unchanged from prior behavior). With probability epsilon,
            SECOND-best candidate is placed instead of the single best one.
            _greedy_pack_into is otherwise a pure hill-climb -- always the
            single best (item, orientation, pivot) move, every step, no
            exploration at all -- so it can only ever produce ONE
            arrangement per ULD; if an early greedy choice blocks a better
            combination a few moves later, nothing recovers from it.
            Intended for MultiRestartPacker (below), which runs N
            independent epsilon-greedy trials per ULD and keeps whichever
            actually real-packs cheapest -- a genuinely different lever
            from tuning the scoring key itself (contact/dblf/min_envelope),
            all of which remain single deterministic passes.
        """
        if rl_packer_src is None:
            rl_packer_src = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                'rl_packer', 'src'
            )
        rl_packer_src = os.path.abspath(rl_packer_src)
        self._geometry = _import_geometry(rl_packer_src)
        self.max_pivots = max_pivots
        assert strategy in ('contact', 'dblf', 'min_envelope')
        self.strategy = strategy
        self.epsilon = epsilon
        self.rng = rng
        assert origin_source in ('pivot', 'ems')
        self.origin_source = origin_source

    def _greedy_pack_into(self, hm, pkgs_df):
        """Greedily place packages from pkgs_df into hm (already possibly
        partially filled), choosing the (item, pivot, orientation) that
        scores best under self.strategy among all currently valid
        candidates (or, with probability self.epsilon, the SECOND-best,
        for epsilon-greedy multi-restart exploration). Returns left_behind
        Package_IDs."""
        pool = pkgs_df.reset_index(drop=True).copy()
        while len(pool) > 0:
            best = None       # (sort_key, pool_idx, dx,dy,dz,x,y,z, weight, pid)
            second_best = None
            if self.origin_source == 'pivot':
                pivots = hm.pivot_points(cap=self.max_pivots)
            else:
                min_dims = (int(pool['Length'].min()), int(pool['Width'].min()), int(pool['Height'].min()))
                pivots = hm.ems_candidates(cap=self.max_pivots, min_dims=min_dims)
            if self.strategy == 'min_envelope':
                cur_x1 = max((p.x1 for p in hm.placements), default=0)
                cur_y1 = max((p.y1 for p in hm.placements), default=0)
                cur_z1 = max((p.z1 for p in hm.placements), default=0)
                cur_envelope = cur_x1 * cur_y1 * cur_z1
            for pool_idx, row in pool.iterrows():
                l, w, h, weight = int(row['Length']), int(row['Width']), int(row['Height']), float(row['Weight'])
                seen_dims = set()
                for orient_idx in range(6):
                    dx, dy, dz = hm.orientation_dims(l, w, h, orient_idx)
                    if (dx, dy, dz) in seen_dims:
                        continue
                    seen_dims.add((dx, dy, dz))
                    for (x, y, z) in pivots:
                        if hm.fits(dx, dy, dz, x, y, z, weight):
                            volume = dx * dy * dz
                            if self.strategy == 'contact':
                                contact = _face_contact_area(hm, x, y, z, dx, dy, dz)
                                key = (contact, volume)  # maximize
                            elif self.strategy == 'min_envelope':
                                new_envelope = (max(cur_x1, x + dx) * max(cur_y1, y + dy)
                                                 * max(cur_z1, z + dz))
                                contact = _face_contact_area(hm, x, y, z, dx, dy, dz)
                                key = (-(new_envelope - cur_envelope), contact)  # maximize (minimize growth)
                            else:  # dblf
                                contact = _face_contact_area(hm, x, y, z, dx, dy, dz)
                                key = (-z, -y, -x, contact)  # maximize (i.e. minimize z,y,x)
                            if best is None or key > best[0]:
                                second_best = best
                                best = (key, volume, pool_idx, dx, dy, dz, x, y, z, weight, row['Package_ID'])
                            elif second_best is None or key > second_best[0]:
                                second_best = (key, volume, pool_idx, dx, dy, dz, x, y, z, weight, row['Package_ID'])
            if best is None:
                break  # no valid placement for anything left in the pool
            chosen = best
            if (self.epsilon > 0 and second_best is not None
                    and self.rng.random() < self.epsilon):
                chosen = second_best
            _, _, pool_idx, dx, dy, dz, x, y, z, weight, pid = chosen
            hm.place(pid, dx, dy, dz, x, y, z, weight)
            pool = pool.drop(pool_idx)
        return list(pool['Package_ID'])

    def _pack_uld_mixed(self, uid, pids, uld_lookup, pkg_lookup):
        """Alternative to _pack_uld: packs Priority and Economy TOGETHER in
        ONE combined greedy pass (not Priority-strictly-first) -- since
        Priority carries Delay_Cost=0, the strict priority-first convention
        gives it first claim on the best-scoring position at every single
        step purely by TYPE, never because it's geometrically the better
        choice. A joint greedy pass lets whichever package (either type)
        best fits the current best-scoring slot go first, which could
        pack tighter overall -- but risks leaving some Priority package
        unplaced if Economy happens to claim space Priority needed.
        Falls back to the guaranteed-Priority _pack_uld if that happens
        (verified after the fact, never risking the hard Priority
        constraint) -- a strict min-of-2 in spirit: try the potentially
        better mixed pack, keep it only if it's ALSO safe."""
        combined_ids = list(pids)
        combined_df = pd.DataFrame([pkg_lookup[pid] for pid in combined_ids])
        uld_row = uld_lookup[uid]
        hm = self._geometry.Heightmap(
            length=int(uld_row['Length']), width=int(uld_row['Width']),
            height=int(uld_row['Height']), weight_limit=float(uld_row['Weight_Limit']),
        )
        left_behind_ids = self._greedy_pack_into(hm, combined_df) if len(combined_df) > 0 else []
        priority_ids = [pid for pid in pids if pkg_lookup[pid]['Type'] == 'Priority']
        if any(pid in left_behind_ids for pid in priority_ids):
            return None  # unsafe -- caller falls back to _pack_uld
        return hm, left_behind_ids

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        priority_ids = [pid for pid in pids if pkg_lookup[pid]['Type'] == 'Priority']
        economy_ids = [pid for pid in pids if pkg_lookup[pid]['Type'] != 'Priority']
        priority_df = pd.DataFrame([pkg_lookup[pid] for pid in priority_ids])
        economy_df = pd.DataFrame([pkg_lookup[pid] for pid in economy_ids])

        uld_row = uld_lookup[uid]
        hm = self._geometry.Heightmap(
            length=int(uld_row['Length']), width=int(uld_row['Width']),
            height=int(uld_row['Height']), weight_limit=float(uld_row['Weight_Limit']),
        )
        left_behind_ids = []
        if len(priority_df) > 0:
            left_behind_ids += self._greedy_pack_into(hm, priority_df)
        if len(economy_df) > 0:
            left_behind_ids += self._greedy_pack_into(hm, economy_df)
        return hm, left_behind_ids

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        """Same overall structure (chained per-ULD pack + cross-ULD
        eviction-rescue for stuck Priority) as RLPackerAdapter.pack()."""
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
