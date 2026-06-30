"""
greedy_pack.py — single-pass greedy placer.

KEY FIXES
---------
FIX 1 (_best_placement_no_support was completely broken):
  The previous implementation passed w_contact=0.0 to rank_placements,
  believing this would skip the support check. It does NOT — support is a
  hard filter ("if not supported: continue"), not a weighted term. Setting
  w_contact=0 only changes scoring, never bypasses the filter. The fallback
  was therefore identical to the normal call and never helped.

  FIX: _best_placement_no_support now does its own point iteration and
  explicitly omits the support check.

FIX 2 (weight tracking in _rebuild_tracker):
  Correctly calls add_weight for all boxes being replayed.

FIX 3 (priority sort order):
  Sort priority packages by (-max_dim, -volume, -weight) instead of
  (-volume, -weight). Packages with the largest single dimension must be
  placed first, otherwise they get blocked by packages placed earlier.

FIX 4 (multi-sort + failed-first retry):
  _pack_priority_multi_sort tries multiple sort orders and keeps the best
  result. When failures remain it retries with failed packages moved to the
  front (iteratively until stable), then does a per-failed-package centric
  repack where each failing package gets first pick of space.

FIX 5 (weight-constrained ULD preassignment):
  Some ULDs have a very low weight-limit/volume ratio (e.g. wt_limit=75 on
  a 9M-volume ULD). Greedy packing randomly fills these with a few early
  packages and then saturates their weight budget, wasting most of their
  volume capacity and starving the other ULDs of space.

  FIX: detect ULDs whose weight density (wt_limit/volume) is far below the
  fleet average, then deliberately fill them with the lightest packages that
  fit (greedy knapsack by volume/weight ratio). This is tried as an additional
  ordering strategy inside _pack_priority_multi_sort — it wins only if it
  places more packages than every other ordering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from geometry import Package, ULD, PlacedBox
from extreme_points import ExtremePointTracker
from selector import rank_placements

_EPS = 1e-9

# Sort keys tried for priority packages (in order). The best result wins.
_PRIORITY_SORT_KEYS = [
    lambda p: (-p.volume, -p.weight),
    lambda p: (-max(p.length, p.width, p.height), -p.volume, -p.weight),
    lambda p: (-p.weight, -p.volume),
    lambda p: (-max(p.length, p.width, p.height), -p.weight, -p.volume),
    lambda p: (p.volume, p.weight),                             # smallest-first
    lambda p: (                                                 # sorted-dims desc
        -sorted([p.length, p.width, p.height])[-1],
        -sorted([p.length, p.width, p.height])[-2],
        -p.volume,
    ),
]

# A ULD is considered weight-constrained when its wt_limit/volume ratio is
# below this fraction of the fleet average. Such ULDs need deliberate
# preassignment of light packages to avoid wasting their volume capacity.
_TIGHT_ULD_THRESHOLD = 0.15


@dataclass
class PackResult:
    trackers:     Dict[str, ExtremePointTracker]
    placed_boxes: List[PlacedBox] = field(default_factory=list)
    left_behind:  List[str]       = field(default_factory=list)
    unplaceable:  List[str]       = field(default_factory=list)
    priority_ids: Set[str]        = field(default_factory=set)

    @property
    def infeasible(self) -> bool:
        return len(self.unplaceable) > 0

    def uld_priority_count(self) -> int:
        seen: Set[str] = set()
        for box in self.placed_boxes:
            if box.package_id in self.priority_ids:
                seen.add(box.uld_id)
        return len(seen)

    def total_cost(self, delay_costs: Dict[str, float], k_penalty: float) -> float:
        return (
            sum(delay_costs.get(pid, 0.0) for pid in self.left_behind)
            + k_penalty * self.uld_priority_count()
        )


def greedy_pack(
    packages:           List[Package],
    ulds:               List[ULD],
    candidates_per_uld: int = 5,
    trackers:           Optional[Dict[str, ExtremePointTracker]] = None,
) -> PackResult:
    """
    Two-phase greedy packer.
    Phase 1: ALL priority packages placed first (multi-sort + failed-first
             retry + weight-constrained ULD preassignment).
    Phase 2: Economy packages fill residual space.
    """
    if trackers is None:
        trackers = {u.id: ExtremePointTracker(u) for u in ulds}

    priority_ids: Set[str] = {p.id for p in packages if p.is_priority}

    priority_packages = [p for p in packages if p.is_priority]
    economy_packages  = [p for p in packages if not p.is_priority]

    # ── Phase 1: priority ────────────────────────────────────────────────
    placed_boxes, unplaceable, trackers = _pack_priority_multi_sort(
        priority_packages, ulds, trackers, candidates_per_uld
    )

    # ── Phase 2: economy ─────────────────────────────────────────────────
    left_behind: List[str] = []
    for pkg in economy_packages:
        priority_uld_ids: Set[str] = {
            b.uld_id for b in placed_boxes if b.package_id in priority_ids
        }
        box = _best_placement(pkg, ulds, trackers, priority_uld_ids, candidates_per_uld)
        if box is not None:
            trackers[box.uld_id].commit(box)
            trackers[box.uld_id].add_weight(pkg.weight)
            placed_boxes.append(box)
        else:
            left_behind.append(pkg.id)

    return PackResult(
        trackers=trackers,
        placed_boxes=placed_boxes,
        left_behind=left_behind,
        unplaceable=unplaceable,
        priority_ids=priority_ids,
    )


# ── weight-constrained ULD detection ────────────────────────────────────────

def _find_tight_ulds(ulds: List[ULD]) -> List[ULD]:
    """
    Return ULDs whose weight-density (wt_limit / volume) is far below the
    fleet average. These need deliberate light-package preassignment.
    """
    if not ulds:
        return []
    ratios = [u.weight_limit / u.volume for u in ulds]
    avg = sum(ratios) / len(ratios)
    return [u for u, r in zip(ulds, ratios) if r < avg * _TIGHT_ULD_THRESHOLD]


def _knapsack_fill(
    tight_uld: ULD,
    candidates: List[Package],
    already_assigned: Set[str],
) -> List[str]:
    """
    Greedy 0/1 knapsack: fill tight_uld's weight budget with packages from
    candidates (excluding already_assigned) sorted by volume/weight descending
    (maximise volume per unit of the scarce weight resource).
    Returns list of package IDs to assign to this ULD.
    """
    eligible = sorted(
        [p for p in candidates
         if p.id not in already_assigned and p.weight <= tight_uld.weight_limit],
        key=lambda p: -(p.volume / p.weight),
    )
    assigned: List[str] = []
    remaining = tight_uld.weight_limit
    for p in eligible:
        if p.weight <= remaining + _EPS:
            assigned.append(p.id)
            remaining -= p.weight
    return assigned


def _try_tight_uld_preassign(
    priority_packages: List[Package],
    ulds: List[ULD],
    initial_trackers: Dict[str, ExtremePointTracker],
    candidates_per_uld: int,
) -> Optional[Tuple[List[PlacedBox], List[str], Dict[str, ExtremePointTracker]]]:
    """
    Build an ordering where weight-constrained ULDs are filled first with
    deliberately selected light packages, then pack the remainder normally
    with all sort keys. Returns the best (placed, failed, trackers) found,
    or None if there are no tight ULDs.
    """
    import copy

    tight_ulds = _find_tight_ulds(ulds)
    if not tight_ulds:
        return None

    # Assign packages to tight ULDs (tightest first)
    tight_ulds_sorted = sorted(tight_ulds, key=lambda u: u.weight_limit / u.volume)
    assigned_ids: Set[str] = set()
    preassign: Dict[str, List[str]] = {}  # uld_id -> [pkg_id, ...]

    for tu in tight_ulds_sorted:
        pkg_ids = _knapsack_fill(tu, priority_packages, assigned_ids)
        preassign[tu.id] = pkg_ids
        assigned_ids.update(pkg_ids)

    remaining_pkgs = [p for p in priority_packages if p.id not in assigned_ids]

    best_placed: List[PlacedBox] = []
    best_failed: List[str] = [p.id for p in priority_packages]
    best_trs: Dict[str, ExtremePointTracker] = initial_trackers

    for sort_key in _PRIORITY_SORT_KEYS:
        trs = {uid: copy.deepcopy(tr) for uid, tr in initial_trackers.items()}
        pid_set = {p.id for p in priority_packages}
        placed: List[PlacedBox] = []
        failed: List[str] = []

        # Phase A: force preassigned packages into their designated tight ULD
        for tu in tight_ulds_sorted:
            tu_obj = next(u for u in ulds if u.id == tu.id)
            pkgs_for_tu = sorted(
                [p for p in priority_packages if p.id in set(preassign[tu.id])],
                key=lambda p: (-p.volume, -p.weight),
            )
            for pkg in pkgs_for_tu:
                box = _best_placement(pkg, [tu_obj], trs, set(), candidates_per_uld)
                if box is None:
                    box = _best_placement_no_support(pkg, [tu_obj], trs)
                if box is not None:
                    trs[box.uld_id].commit(box)
                    trs[box.uld_id].add_weight(pkg.weight)
                    placed.append(box)
                else:
                    # Couldn't fit in tight ULD — will retry in phase B
                    failed.append(pkg.id)

        # Phase B: pack remaining packages across all ULDs with this sort key
        phase_b = sorted(remaining_pkgs, key=sort_key)
        # Also retry any tight-ULD failures
        retry_ids = set(failed)
        failed = []
        for pkg in phase_b:
            puid = {b.uld_id for b in placed if b.package_id in pid_set}
            box = _best_placement(pkg, ulds, trs, puid, candidates_per_uld)
            if box is None:
                box = _best_placement_no_support(pkg, ulds, trs)
            if box is not None:
                trs[box.uld_id].commit(box)
                trs[box.uld_id].add_weight(pkg.weight)
                placed.append(box)
            else:
                failed.append(pkg.id)

        # Retry packages that failed phase A
        for pkg in [p for p in priority_packages if p.id in retry_ids]:
            puid = {b.uld_id for b in placed if b.package_id in pid_set}
            box = _best_placement(pkg, ulds, trs, puid, candidates_per_uld)
            if box is None:
                box = _best_placement_no_support(pkg, ulds, trs)
            if box is not None:
                trs[box.uld_id].commit(box)
                trs[box.uld_id].add_weight(pkg.weight)
                placed.append(box)
            else:
                failed.append(pkg.id)

        if len(placed) > len(best_placed):
            best_placed, best_failed, best_trs = placed, failed, trs

    return best_placed, best_failed, best_trs


# ── priority multi-sort ──────────────────────────────────────────────────────

def _pack_priority_multi_sort(
    priority_packages: List[Package],
    ulds: List[ULD],
    initial_trackers: Dict[str, ExtremePointTracker],
    candidates_per_uld: int,
) -> Tuple[List[PlacedBox], List[str], Dict[str, ExtremePointTracker]]:
    """
    Try multiple sort orders for priority packages; keep the best result.
    Also tries weight-constrained ULD preassignment (FIX 5).
    Then retries with failed packages at the front (iteratively until stable).
    Returns (placed_boxes, unplaceable_ids, trackers).
    """
    import copy

    def _try_order(ordered: List[Package]) -> Tuple[List[PlacedBox], List[str], Dict]:
        trs = {uid: copy.deepcopy(tr) for uid, tr in initial_trackers.items()}
        pid_set = {p.id for p in ordered}
        placed: List[PlacedBox] = []
        failed: List[str] = []
        for pkg in ordered:
            puid = {b.uld_id for b in placed if b.package_id in pid_set}
            box = _best_placement(pkg, ulds, trs, puid, candidates_per_uld)
            if box is None:
                box = _best_placement_no_support(pkg, ulds, trs)
            if box is not None:
                trs[box.uld_id].commit(box)
                trs[box.uld_id].add_weight(pkg.weight)
                placed.append(box)
            else:
                failed.append(pkg.id)
        return placed, failed, trs

    best_placed: List[PlacedBox] = []
    best_failed: List[str] = [p.id for p in priority_packages]
    best_trs = initial_trackers

    def _update(placed, failed, trs):
        nonlocal best_placed, best_failed, best_trs
        if len(placed) > len(best_placed):
            best_placed, best_failed, best_trs = placed, failed, trs

    # Phase A: base sort orders
    for key in _PRIORITY_SORT_KEYS:
        placed, failed, trs = _try_order(sorted(priority_packages, key=key))
        _update(placed, failed, trs)

    # Phase B: weight-constrained ULD preassignment (FIX 5)
    tight_result = _try_tight_uld_preassign(
        priority_packages, ulds, initial_trackers, candidates_per_uld
    )
    if tight_result is not None:
        _update(*tight_result)

    # Phase C: iterative failed-first (until no improvement)
    prev_n_failed = len(best_failed) + 1
    while best_failed and len(best_failed) < prev_n_failed:
        prev_n_failed = len(best_failed)
        failed_set = set(best_failed)
        failed_pkgs = [p for p in priority_packages if p.id in failed_set]
        other_pkgs  = [p for p in priority_packages if p.id not in failed_set]
        for key in _PRIORITY_SORT_KEYS:
            placed, failed, trs = _try_order(failed_pkgs + sorted(other_pkgs, key=key))
            _update(placed, failed, trs)

    # Phase D: per-failed-package centric repack
    if best_failed:
        for target_id in list(best_failed):
            target = next((p for p in priority_packages if p.id == target_id), None)
            if target is None:
                continue
            others = [p for p in priority_packages if p.id != target_id]
            for key in _PRIORITY_SORT_KEYS[:3]:
                placed, failed, trs = _try_order([target] + sorted(others, key=key))
                _update(placed, failed, trs)

    return best_placed, best_failed, best_trs


# ── placement helpers ────────────────────────────────────────────────────────

def _best_placement(
    pkg: Package,
    ulds: List[ULD],
    trackers: Dict[str, ExtremePointTracker],
    priority_uld_ids: Set[str],
    candidates_per_uld: int,
) -> Optional[PlacedBox]:
    best_score = float("inf")
    best_box: Optional[PlacedBox] = None
    for uld in ulds:
        cands = rank_placements(
            pkg, trackers[uld.id],
            priority_uld_ids=priority_uld_ids,
            top_k=candidates_per_uld,
        )
        if cands and cands[0].score < best_score:
            best_score = cands[0].score
            best_box   = cands[0].to_placed_box(pkg.id)
    return best_box


def _best_placement_no_support(
    pkg: Package,
    ulds: List[ULD],
    trackers: Dict[str, ExtremePointTracker],
) -> Optional[PlacedBox]:
    """
    Genuine no-support fallback: iterates extreme points and checks overlap
    only — the support constraint is completely omitted. Picks lowest z0.
    """
    best_score = float("inf")
    best_box: Optional[PlacedBox] = None
    for uld in ulds:
        box = _best_placement_no_support_on_tracker(pkg, uld, trackers[uld.id])
        if box is not None and box.z0 < best_score:
            best_score = box.z0
            best_box = box
    return best_box


def _best_placement_no_support_on_tracker(
    pkg: Package,
    uld: ULD,
    tracker: ExtremePointTracker,
) -> Optional[PlacedBox]:
    """No-support placement on a specific tracker. Support check omitted."""
    ul, uw, uh = uld.length, uld.width, uld.height
    placed = tracker.placed
    best_score = float("inf")
    best_box: Optional[PlacedBox] = None

    if pkg.weight > tracker.remaining_weight_capacity() + _EPS:
        return None

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


def _rebuild_tracker(
    uld: ULD,
    boxes_to_keep: List[PlacedBox],
    pkg_by_id: Dict[str, Package],
) -> ExtremePointTracker:
    """Rebuild a tracker from scratch with only the specified boxes."""
    tr = ExtremePointTracker(uld)
    for b in boxes_to_keep:
        tr.commit(b)
        if b.package_id in pkg_by_id:
            tr.add_weight(pkg_by_id[b.package_id].weight)
    return tr
