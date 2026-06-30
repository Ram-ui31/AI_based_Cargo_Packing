"""
greedy_pipeline.py — orchestrator.

KEY FIXES
---------
FIX 1 (CRITICAL — infeasible split caused priority+economy mixed packing):
  When binary_search_split returned feasible=False, the pipeline set
      pass1_packages = priority + ALL economy
  This meant economy packages filled space that priority needed, causing
  catastrophic priority under-placement (e.g. 1/61, 1/125, 1/128).

  FIX: pass1 ALWAYS packs priority packages only, regardless of split.feasible.
  Economy packages only enter after all priority rescue attempts are done.

FIX 2 (priority sort order — in greedy_pack.py):
  Sorting by (-max_dim, -volume, -weight) and trying multiple orderings
  prevents large-dimension packages from being blocked by earlier placements.
  See greedy_pack.py for full details.

FIX 3 (_rescue_priority no-economy skip):
  Previous code skipped ULDs with no economy to evict in Attempt 3.
  Now we attempt placement in priority-only tracker unconditionally.

FIX 4 (_nuclear_eviction infinite bump loop):
  Fixed via round limit and tracking displaced packages.

FIX 5 (merge correctness):
  _merge recomputes final_unplaceable from placed_boxes.

CHANGE (cost reduction):
  - pack_threshold default raised 0.60 -> 0.92: forces binary_search_split to
    find a larger Set 1, packing more economy value before the H2 pass.
    The pipeline's graceful infeasible fallback means this is safe — if the
    threshold can't be met, it falls back to packing all economy anyway.
  - sort_by_h2 call now explicitly passes boosted cost weights (w_cost=4.0)
    consistent with h2_heuristic.py defaults, ensuring expensive leftover
    economy packages are always placed first in pass2.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set
import copy

from geometry import Package, ULD, PlacedBox
from extreme_points import ExtremePointTracker
from uld_partition import partition_ulds
from h1_heuristic import sort_by_h1
from binary_search_split import binary_search_split
from h2_heuristic import sort_by_h2
from greedy_pack import (
    greedy_pack, PackResult,
    _rebuild_tracker,
    _best_placement_no_support_on_tracker,
    _best_placement_no_support,
)
from selector import rank_placements

_EPS = 1e-9


class GreedyPipeline:
    def __init__(
        self,
        ulds:                     List[ULD],
        packages:                 List[Package],
        k_penalty:                float,
        candidates_per_uld:       int   = 5,
        fill_target:              float = 1.2,
        pack_threshold:           float = 0.92,   # CHANGED: was 0.60
        binary_search_candidates: int   = 5,
        **kwargs,
    ):
        self.ulds                     = ulds
        self.packages                 = packages
        self.k_penalty                = k_penalty
        self.candidates_per_uld       = candidates_per_uld
        self.fill_target              = fill_target
        self.pack_threshold           = pack_threshold
        self.binary_search_candidates = binary_search_candidates

        self.priority_packages = [p for p in packages if p.is_priority]
        self.economy_packages  = [p for p in packages if not p.is_priority]
        self.priority_ids      = {p.id for p in self.priority_packages}
        self._pkg_by_id        = {p.id: p for p in packages}

    # ── main pipeline ─────────────────────────────────────────────────────

    def solve(self) -> PackResult:
        partition  = partition_ulds(
            ulds=self.ulds,
            packages=self.packages,
            fill_target=self.fill_target,
        )
        economy_h1 = sort_by_h1(self.economy_packages)
        split = binary_search_split(
            priority_packages=self.priority_packages,
            economy_sorted=economy_h1,
            priority_ulds=partition.priority_ulds,
            other_ulds=partition.other_ulds,
            pack_threshold=self.pack_threshold,
            candidates_per_uld=self.binary_search_candidates,
        )

        # FIX 1: ALWAYS pack priority-only in pass1, regardless of split.feasible.
        # Previously, infeasible split caused priority+ALL_economy to be packed
        # together, allowing economy to steal space from priority packages.
        # Economy only enters after all priority rescue attempts complete.
        pass1 = greedy_pack(
            packages=self.priority_packages,
            ulds=self.ulds,
            candidates_per_uld=self.candidates_per_uld,
        )

        # Rescue pass
        still_unplaceable = list(pass1.unplaceable)
        if still_unplaceable:
            still_unplaceable, pass1 = self._rescue_priority(still_unplaceable, pass1)

        # Nuclear eviction — absolute last resort
        if still_unplaceable:
            still_unplaceable, pass1 = self._nuclear_eviction(still_unplaceable, pass1)

        # Economy: determine which economy packages to include from split
        placed_ids = {b.package_id for b in pass1.placed_boxes}
        if split.feasible:
            # Use the binary-search-determined subset of economy packages
            economy_to_pack = [p for p in split.set1 if p.id not in placed_ids]
        else:
            # Infeasible split: try to pack all economy (priority already placed above)
            economy_to_pack = [p for p in self.economy_packages if p.id not in placed_ids]

        # CHANGE: explicitly pass boosted cost weights to H2 so expensive
        # leftover economy packages are always prioritised in pass2.
        h2_sorted = (
            sort_by_h2(
                economy_to_pack,
                trackers=pass1.trackers,
                w_fit=2.0,   # CHANGED: was 3.0 (default)
                w_cost=4.0,  # CHANGED: was 2.0 (default)
                w_small=0.5, # CHANGED: was 1.0 (default)
            )
            if economy_to_pack else []
        )

        if h2_sorted:
            seeded = {uid: copy.deepcopy(tr) for uid, tr in pass1.trackers.items()}
            pass2  = greedy_pack(
                packages=h2_sorted,
                ulds=self.ulds,
                candidates_per_uld=self.candidates_per_uld,
                trackers=seeded,
            )
        else:
            pass2 = None

        return self._merge(pass1, pass2)

    # ── rescue pass ───────────────────────────────────────────────────────

    def _rescue_priority(
        self,
        unplaceable_ids: List[str],
        pass1: PackResult,
    ):
        """
        Three-attempt rescue for each unplaced priority package:
        1. Normal placement with no consolidation penalty.
        2. No-support placement across all ULDs.
        3. Evict economy from most spacious compatible ULD, try again.
           Also attempts ULDs with zero economy to evict.
        """
        trackers = pass1.trackers
        placed_boxes = list(pass1.placed_boxes)
        still_failed: List[str] = []

        for pid in unplaceable_ids:
            if pid not in self._pkg_by_id:
                still_failed.append(pid)
                continue
            pkg = self._pkg_by_id[pid]

            # Attempt 1: normal, no consolidation penalty
            best_box = None
            best_score = float("inf")
            for uld in self.ulds:
                cands = rank_placements(
                    pkg, trackers[uld.id],
                    priority_uld_ids=set(),
                    top_k=self.candidates_per_uld,
                )
                if cands and cands[0].score < best_score:
                    best_score = cands[0].score
                    best_box   = cands[0].to_placed_box(pkg.id)

            if best_box:
                trackers[best_box.uld_id].commit(best_box)
                trackers[best_box.uld_id].add_weight(pkg.weight)
                placed_boxes.append(best_box)
                continue

            # Attempt 2: no-support across all ULDs
            best_box = _best_placement_no_support(pkg, self.ulds, trackers)
            if best_box:
                trackers[best_box.uld_id].commit(best_box)
                trackers[best_box.uld_id].add_weight(pkg.weight)
                placed_boxes.append(best_box)
                continue

            # Attempt 3: evict economy, rebuild, retry
            rescued = False
            pkg_dims = sorted([pkg.length, pkg.width, pkg.height])
            for uld in sorted(self.ulds, key=lambda u: -u.volume):
                uld_dims = sorted([uld.length, uld.width, uld.height])
                if (pkg_dims[0] > uld_dims[0] + _EPS or
                        pkg_dims[1] > uld_dims[1] + _EPS or
                        pkg_dims[2] > uld_dims[2] + _EPS):
                    continue
                if pkg.weight > uld.weight_limit + _EPS:
                    continue

                economy_in_uld = [
                    b for b in placed_boxes
                    if b.uld_id == uld.id and b.package_id not in self.priority_ids
                ]
                priority_in_uld = [
                    b for b in placed_boxes
                    if b.uld_id == uld.id and b.package_id in self.priority_ids
                ]

                new_tr = _rebuild_tracker(uld, priority_in_uld, self._pkg_by_id)

                cands = rank_placements(
                    pkg, new_tr,
                    priority_uld_ids=set(),
                    top_k=self.candidates_per_uld,
                )
                if not cands:
                    rescue_box = _best_placement_no_support_on_tracker(pkg, uld, new_tr)
                else:
                    rescue_box = cands[0].to_placed_box(pkg.id)

                if rescue_box is None:
                    if not economy_in_uld:
                        continue
                    empty_tr = ExtremePointTracker(uld)
                    rescue_box = _best_placement_no_support_on_tracker(pkg, uld, empty_tr)
                    if rescue_box is None:
                        continue
                    evicted_ids = {b.package_id for b in placed_boxes if b.uld_id == uld.id}
                    placed_boxes = [b for b in placed_boxes if b.package_id not in evicted_ids]
                    empty_tr.commit(rescue_box)
                    empty_tr.add_weight(pkg.weight)
                    trackers[uld.id] = empty_tr
                    placed_boxes.append(rescue_box)
                    rescued = True
                    break

                evicted_ids = {b.package_id for b in economy_in_uld}
                placed_boxes = [b for b in placed_boxes if b.package_id not in evicted_ids]
                new_tr.commit(rescue_box)
                new_tr.add_weight(pkg.weight)
                trackers[uld.id] = new_tr
                placed_boxes.append(rescue_box)
                rescued = True
                break

            if not rescued:
                still_failed.append(pid)

        placed_ids = {b.package_id for b in placed_boxes}
        all_economy_left = [p.id for p in self.economy_packages if p.id not in placed_ids]

        updated = PackResult(
            trackers=trackers,
            placed_boxes=placed_boxes,
            left_behind=all_economy_left,
            unplaceable=still_failed,
            priority_ids=self.priority_ids,
        )
        return still_failed, updated

    # ── nuclear eviction ──────────────────────────────────────────────────

    def _nuclear_eviction(
        self,
        unplaceable_ids: List[str],
        pass1: PackResult,
    ):
        """
        Last resort: clear entire ULDs to force-place priority packages.
        Processes the full list of unplaced priority packages sorted largest-first.
        Re-queues displaced priority packages for subsequent rounds.
        """
        trackers = pass1.trackers
        placed_boxes = list(pass1.placed_boxes)

        need_placement: Set[str] = set(unplaceable_ids)

        max_rounds = len(self.priority_packages) + 1
        for _ in range(max_rounds):
            if not need_placement:
                break

            to_place = sorted(
                [self._pkg_by_id[pid] for pid in need_placement if pid in self._pkg_by_id],
                key=lambda p: (-p.volume, -p.weight),
            )
            need_placement.clear()
            still_failed: List[str] = []

            for pkg in to_place:
                pkg_dims = sorted([pkg.length, pkg.width, pkg.height])
                placed = False

                for uld in sorted(self.ulds, key=lambda u: -u.volume):
                    uld_dims = sorted([uld.length, uld.width, uld.height])
                    if (pkg_dims[0] > uld_dims[0] + _EPS or
                            pkg_dims[1] > uld_dims[1] + _EPS or
                            pkg_dims[2] > uld_dims[2] + _EPS):
                        continue
                    if pkg.weight > uld.weight_limit + _EPS:
                        continue

                    # Try in current state first (no eviction)
                    cands = rank_placements(
                        pkg, trackers[uld.id],
                        priority_uld_ids=set(),
                        top_k=self.candidates_per_uld,
                    )
                    if not cands:
                        cands_box = _best_placement_no_support_on_tracker(
                            pkg, uld, trackers[uld.id]
                        )
                    else:
                        cands_box = cands[0].to_placed_box(pkg.id)

                    if cands_box:
                        trackers[cands_box.uld_id].commit(cands_box)
                        trackers[cands_box.uld_id].add_weight(pkg.weight)
                        placed_boxes.append(cands_box)
                        placed = True
                        break

                    # Must evict — clear this ULD completely
                    displaced_in_uld = [b for b in placed_boxes if b.uld_id == uld.id]
                    displaced_ids = {b.package_id for b in displaced_in_uld}

                    new_tr = ExtremePointTracker(uld)
                    cands = rank_placements(
                        pkg, new_tr,
                        priority_uld_ids=set(),
                        top_k=self.candidates_per_uld,
                    )
                    if not cands:
                        rescue_box = _best_placement_no_support_on_tracker(pkg, uld, new_tr)
                    else:
                        rescue_box = cands[0].to_placed_box(pkg.id)

                    if rescue_box is None:
                        continue

                    placed_boxes = [b for b in placed_boxes if b.uld_id != uld.id]
                    new_tr.commit(rescue_box)
                    new_tr.add_weight(pkg.weight)
                    trackers[uld.id] = new_tr
                    placed_boxes.append(rescue_box)

                    for did in displaced_ids:
                        if did in self.priority_ids and did != pkg.id:
                            need_placement.add(did)
                    placed = True
                    break

                if not placed:
                    still_failed.append(pkg.id)

            if still_failed:
                break

        placed_ids = {b.package_id for b in placed_boxes}
        final_economy_left = [p.id for p in self.economy_packages if p.id not in placed_ids]
        final_unplaceable  = [p.id for p in self.priority_packages if p.id not in placed_ids]

        updated = PackResult(
            trackers=trackers,
            placed_boxes=placed_boxes,
            left_behind=final_economy_left,
            unplaceable=final_unplaceable,
            priority_ids=self.priority_ids,
        )
        return final_unplaceable, updated

    # ── merge ─────────────────────────────────────────────────────────────

    def _merge(
        self,
        pass1: PackResult,
        pass2: Optional[PackResult],
    ) -> PackResult:
        pass2_boxes = pass2.placed_boxes if pass2 else []

        all_placed_ids = (
            {b.package_id for b in pass1.placed_boxes}
            | {b.package_id for b in pass2_boxes}
        )

        final_left_behind = [
            p.id for p in self.economy_packages if p.id not in all_placed_ids
        ]
        final_unplaceable = [
            p.id for p in self.priority_packages if p.id not in all_placed_ids
        ]

        return PackResult(
            trackers=pass2.trackers if pass2 else pass1.trackers,
            placed_boxes=pass1.placed_boxes + pass2_boxes,
            left_behind=final_left_behind,
            unplaceable=final_unplaceable,
            priority_ids=self.priority_ids,
        )
