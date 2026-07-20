"""
multi_restart_packer.py -- Packer-interface-compatible wrapper that runs N
independent epsilon-greedy trials of a HeuristicPacker strategy per ULD and
keeps whichever trial is actually cheapest, instead of a single
deterministic pass.

Why: HeuristicPacker._greedy_pack_into is a pure hill-climb -- at every
step it deterministically places the single best (item, orientation,
pivot) by the scoring key, with ZERO exploration. Every strategy tried so
far this session (contact, dblf, min_envelope) is still just ONE
arrangement per ULD; if an early greedy choice blocks a better combination
a few moves later, nothing recovers from it, no matter how good the
scoring key is. This is a genuinely different lever: same validated
scoring keys, but with epsilon-greedy randomization (HeuristicPacker's
epsilon/rng params) run N times per ULD, keeping the real cheapest result
-- a real multi-start local search, not a new heuristic.

Cheap in practice: a single-ULD pack with ~50 items at max_pivots=200 is on
the order of ~1-2s (see scripts/test_multi_restart_packer.py's timing
calibration), so N=15-20 restarts x 6 ULDs is a few minutes, not hours.
"""
from __future__ import annotations

import numpy as np

from .heuristic_packer import HeuristicPacker


class MultiRestartPacker:
    """Packer-interface-compatible: pack(assignment, packages_df, ulds_df).

    base_strategy : which HeuristicPacker scoring key to randomize
        ('contact', 'dblf', or 'min_envelope').
    n_restarts    : number of independent epsilon-greedy trials per ULD.
    epsilon       : per-step probability of taking the second-best
        candidate instead of the best one (see HeuristicPacker.__init__).
    max_pivots    : pivot cap per trial (kept moderate since this runs
        n_restarts times per ULD -- total cost scales linearly with it).
    """

    def __init__(self, base_strategy='contact', n_restarts=15, epsilon=0.15,
                 max_pivots=200, rl_packer_src=None, seed=0):
        self.base_strategy = base_strategy
        self.n_restarts = n_restarts
        self.epsilon = epsilon
        self.max_pivots = max_pivots
        self.rl_packer_src = rl_packer_src
        self._rng = np.random.default_rng(seed)
        # One reusable HeuristicPacker instance per trial (rng swapped out
        # between trials so each restart is independently randomized).
        self._packer = HeuristicPacker(rl_packer_src=rl_packer_src, max_pivots=max_pivots,
                                        strategy=base_strategy, epsilon=epsilon, rng=self._rng)
        self._geometry = self._packer._geometry

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        best_hm, best_left, best_score = None, None, None
        for trial in range(self.n_restarts):
            hm, left = self._packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
            delay_left = sum(pkg_lookup[pid]['Delay_Cost'] for pid in left)
            score = (delay_left, -hm.utilization() if hm else 0.0)
            if best_score is None or score < best_score:
                best_hm, best_left, best_score = hm, left, score
        return best_hm, best_left

    def pack(self, assignment, packages_df, ulds_df, max_rescue_rounds=8):
        """Same overall structure as HeuristicPacker.pack() (chained
        per-ULD pack + cross-ULD eviction-rescue for stuck Priority), but
        each per-ULD pack is the best-of-N_restarts trial above."""
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
