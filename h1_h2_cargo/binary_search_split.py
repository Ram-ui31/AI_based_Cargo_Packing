"""
binary_search_split.py

FIX: _trial_pack now tries a no-support fallback for each priority package
that fails standard placement. This prevents the trial from declaring
feasible=False just because the support heuristic blocked a valid position
in the trial's simplified packer, which would cause the pipeline to use a
degraded fallback path and ultimately leave priority packages unplaced.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from geometry import Package, ULD, PlacedBox
from extreme_points import ExtremePointTracker
from selector import rank_placements

_EPS = 1e-9


@dataclass
class SplitResult:
    set1: List[Package]
    set2: List[Package]
    best_n: int
    feasible: bool


def binary_search_split(
    priority_packages:  List[Package],
    economy_sorted:     List[Package],
    priority_ulds:      List[ULD],
    other_ulds:         List[ULD],
    pack_threshold:     float = 0.60,
    candidates_per_uld: int   = 5,
) -> SplitResult:
    priority_packages = sorted(priority_packages, key=lambda p: (-p.volume, -p.weight))
    all_ulds = priority_ulds + other_ulds
    n = len(economy_sorted)

    if n == 0:
        ok, _ = _trial_pack(priority_packages, [], all_ulds, candidates_per_uld)
        return SplitResult(set1=[], set2=[], best_n=0, feasible=ok)

    full_ok, _ = _trial_pack(
        priority_packages, economy_sorted, all_ulds,
        candidates_per_uld, pack_threshold,
    )
    if full_ok:
        return SplitResult(set1=list(economy_sorted), set2=[], best_n=n, feasible=True)

    pri_ok, _ = _trial_pack(priority_packages, [], all_ulds, candidates_per_uld)
    if not pri_ok:
        return SplitResult(set1=[], set2=list(economy_sorted), best_n=0, feasible=False)

    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _ = _trial_pack(
            priority_packages, economy_sorted[:mid], all_ulds,
            candidates_per_uld, pack_threshold,
        )
        if ok:
            lo = mid
        else:
            hi = mid

    return SplitResult(
        set1=list(economy_sorted[:lo]),
        set2=list(economy_sorted[lo:]),
        best_n=lo,
        feasible=True,
    )


def _trial_pack(
    priority_packages:  List[Package],
    economy_subset:     List[Package],
    ulds:               List[ULD],
    candidates_per_uld: int,
    pack_threshold:     float = 0.0,
) -> Tuple[bool, float]:
    trackers: Dict[str, ExtremePointTracker] = {
        u.id: ExtremePointTracker(u) for u in ulds
    }
    priority_ids: Set[str] = {p.id for p in priority_packages}
    placed_boxes: List[PlacedBox] = []

    for pkg in priority_packages:
        priority_uld_ids = {
            b.uld_id for b in placed_boxes if b.package_id in priority_ids
        }
        box = _best_placement(pkg, ulds, trackers, priority_uld_ids, candidates_per_uld)
        if box is None:
            # FIX: try no-support fallback before declaring infeasible
            box = _best_placement_no_support(pkg, ulds, trackers)
        if box is None:
            return False, 0.0
        trackers[box.uld_id].commit(box)
        trackers[box.uld_id].add_weight(pkg.weight)
        placed_boxes.append(box)

    total_cost  = sum(p.delay_cost for p in economy_subset)
    packed_cost = 0.0
    for pkg in economy_subset:
        priority_uld_ids = {
            b.uld_id for b in placed_boxes if b.package_id in priority_ids
        }
        box = _best_placement(pkg, ulds, trackers, priority_uld_ids, candidates_per_uld)
        if box is not None:
            trackers[box.uld_id].commit(box)
            trackers[box.uld_id].add_weight(pkg.weight)
            placed_boxes.append(box)
            packed_cost += pkg.delay_cost

    fraction = packed_cost / total_cost if total_cost > 1e-9 else 1.0
    return fraction >= pack_threshold, fraction


def _best_placement(
    pkg:                Package,
    ulds:               List[ULD],
    trackers:           Dict[str, ExtremePointTracker],
    priority_uld_ids:   Set[str],
    candidates_per_uld: int,
) -> Optional[PlacedBox]:
    best_score = float("inf")
    best_box   = None
    for uld in ulds:
        candidates = rank_placements(
            pkg, trackers[uld.id],
            priority_uld_ids=priority_uld_ids,
            top_k=candidates_per_uld,
        )
        if candidates and candidates[0].score < best_score:
            best_score = candidates[0].score
            best_box   = candidates[0].to_placed_box(pkg.id)
    return best_box


def _best_placement_no_support(
    pkg: Package,
    ulds: List[ULD],
    trackers: Dict[str, ExtremePointTracker],
) -> Optional[PlacedBox]:
    """Genuine no-support fallback: overlap check only, no support filter."""
    best_score = float("inf")
    best_box   = None

    for uld in ulds:
        tracker = trackers[uld.id]
        if pkg.weight > tracker.remaining_weight_capacity() + _EPS:
            continue
        ul, uw, uh = uld.length, uld.width, uld.height
        placed = tracker.placed

        for extents in pkg.orientations():
            ex, ey, ez = extents
            if ex > ul + _EPS or ey > uw + _EPS or ez > uh + _EPS:
                continue
            for point in tracker.points:
                x0, y0, z0 = point
                x1 = x0 + ex; y1 = y0 + ey; z1 = z0 + ez
                if x0 < -_EPS or y0 < -_EPS or z0 < -_EPS:
                    continue
                if x1 > ul + _EPS or y1 > uw + _EPS or z1 > uh + _EPS:
                    continue
                overlaps = False
                for b in placed:
                    if (x0 >= b.x0 + b.extent_x - _EPS or b.x0 >= x1 - _EPS or
                            y0 >= b.y0 + b.extent_y - _EPS or b.y0 >= y1 - _EPS or
                            z0 >= b.z0 + b.extent_z - _EPS or b.z0 >= z1 - _EPS):
                        continue
                    overlaps = True
                    break
                if overlaps:
                    continue
                if z0 < best_score:
                    best_score = z0
                    best_box = PlacedBox(
                        package_id=pkg.id, uld_id=uld.id,
                        x0=x0, y0=y0, z0=z0,
                        extent_x=ex, extent_y=ey, extent_z=ez,
                    )
    return best_box
