"""
centrifuged_packer.py -- Packer-interface-compatible wrapper implementing
_pack_uld (the per-container method CombinedPacker actually calls for its
min-of-N selection, NOT .pack()): pack normally with a given
HeuristicPacker strategy, "centrifuge" (Heightmap.compact()) to
consolidate fragmented free space, then retry fitting whichever packages
didn't make it the first time.

Why this needs to compete AS A CANDIDATE, not run as post-processing
after CombinedPacker has already chosen: post-processing centrifuge_refine()
AFTER the ensemble's best-of-N-per-container selection recovers ~0 points
on this instance -- the ensemble already tends toward whichever candidate
is least fragmented, so there's little left to consolidate by the time
centrifuging runs on its output. But offered as its own candidate, a
centrifuged strategy can still win the per-container comparison even when
the ensemble's other candidates are already strong -- proven on a single
strategy alone (dblf+ems: 30,819 -> 29,441, 1,378 points before any
ensembling at all).
"""
from __future__ import annotations

import pandas as pd

from .heuristic_packer import HeuristicPacker


class CentrifugedPacker:
    """Packer-interface-compatible: _pack_uld(uid, pids, uld_lookup, pkg_lookup).

    base_packer may be any object implementing _pack_uld with that same
    signature (a HeuristicPacker, an RLPackerAdapter, etc) -- centrifuging
    is a post-hoc compaction+refill step, agnostic to what produced the
    initial placement. The refill step itself always uses an EMS-origin
    HeuristicPacker regardless of what packed the initial placements,
    since refill's job is candidate generation into freed space, not
    reproducing the base packer's own placement policy.
    """

    def __init__(self, base_packer=None, strategy='dblf', n_cycles=2, rl_packer_src=None):
        self.base_packer = base_packer if base_packer is not None else HeuristicPacker(
            rl_packer_src=rl_packer_src, strategy=strategy, origin_source='ems')
        self.filler = HeuristicPacker(rl_packer_src=rl_packer_src, strategy=strategy, origin_source='ems')
        self.n_cycles = n_cycles

    def _pack_uld(self, uid, pids, uld_lookup, pkg_lookup):
        hm, left = self.base_packer._pack_uld(uid, pids, uld_lookup, pkg_lookup)
        if hm is None or not left:
            return hm, left

        compacted = hm
        still_left = left
        for _ in range(self.n_cycles):
            weight_by_pid = {p.package_id: pkg_lookup[p.package_id]['Weight'] for p in compacted.placements}
            compacted = compacted.compact(weight_by_pid)
            left_df = pd.DataFrame([pkg_lookup[pid] for pid in still_left])
            left_df = left_df.assign(
                _vol=lambda d: d['Length'] * d['Width'] * d['Height'],
            ).assign(
                _vdp=lambda d: d['Delay_Cost'] / (d['_vol'].clip(lower=1) ** 1.5),
            ).sort_values('_vdp', ascending=False)
            new_left = self.filler._greedy_pack_into(compacted, left_df)
            if len(new_left) == len(still_left):
                break  # no change this cycle, converged
            still_left = new_left

        return compacted, still_left
