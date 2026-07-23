"""
selector.py — candidate generator and scorer.

FIX vs previous version
-----------------------
BUG: consolidation_penalty fired for the FIRST priority package placed into
ANY ULD because priority_uld_ids was empty at that point, making
`uld_id not in priority_uld_ids` always True on the first attempt.
This inflated scores for every first priority placement, harming binary-search
threshold checks without providing any consolidation signal.

FIX: penalty only applies when there is ALREADY at least one priority ULD
AND the current ULD is not one of them (i.e. we'd be spreading to a new ULD).
When priority_uld_ids is empty the penalty is 0 — the first placement can go
anywhere without penalty.

CHANGE (cost reduction):
    Added a fragmentation_penalty for economy packages. When an economy package
    would be the FIRST item placed in a completely empty ULD, it gets a small
    score penalty (0.5) that discourages opening a fresh ULD when other ULDs
    still have usable space. This leaves larger contiguous gaps available for
    higher-value packages placed later in the H2 pass, reducing overall cost.
    The penalty is small enough that it never blocks a package that genuinely
    has no other option.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set

from geometry import Package, PlacedBox, fits_in_bounds
from extreme_points import ExtremePointTracker, Point

_EPS = 1e-9
_SUP_EPS = 1e-6


@dataclass
class Candidate:
    uld_id: str
    point: Point
    orientation: tuple
    score: float

    def to_placed_box(self, package_id: str) -> PlacedBox:
        x0, y0, z0 = self.point
        ex, ey, ez = self.orientation
        return PlacedBox(
            package_id=package_id, uld_id=self.uld_id,
            x0=x0, y0=y0, z0=z0,
            extent_x=ex, extent_y=ey, extent_z=ez,
        )


def rank_placements(
    package: Package,
    tracker: ExtremePointTracker,
    priority_uld_ids: Set[str],
    top_k: Optional[int] = None,
    w_height: float = 1.0,
    w_contact: float = 2.0,
    w_consolidation: float = 3.0,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    placed = tracker.placed
    uld = tracker.uld
    uld_id = uld.id
    ul = uld.length; uw = uld.width; uh = uld.height
    remaining_cap = tracker.remaining_weight_capacity()
    pw = package.weight
    is_priority = package.is_priority

    # FIX: only apply consolidation penalty when priority packages are ALREADY
    # in other ULDs. When priority_uld_ids is empty (nothing placed yet) there
    # is no spreading penalty — the first placement goes wherever fits best.
    if is_priority and len(priority_uld_ids) > 0 and uld_id not in priority_uld_ids:
        consolidation_penalty = w_consolidation
    else:
        consolidation_penalty = 0.0

    # CHANGE: discourage economy packages from opening a completely empty ULD
    # when other ULDs still have boxes (and therefore usable extreme points).
    # A penalty of 0.5 is small — it yields to any ULD with even one box —
    # but never blocks a package that has no other option (it will still place
    # here if every other ULD is also empty or full).
    if not is_priority and len(placed) == 0 and len(tracker.points) == 1:
        fragmentation_penalty = 0.5
    else:
        fragmentation_penalty = 0.0

    if pw > remaining_cap + _EPS:
        return candidates

    points = tracker.points  # snapshot

    for extents in package.orientations():
        ex, ey, ez = extents

        if ex > ul + _EPS or ey > uw + _EPS or ez > uh + _EPS:
            continue

        for point in points:
            x0, y0, z0 = point
            x1 = x0 + ex
            y1 = y0 + ey
            z1 = z0 + ez

            if x0 < -_EPS or y0 < -_EPS or z0 < -_EPS:
                continue
            if x1 > ul + _EPS or y1 > uw + _EPS or z1 > uh + _EPS:
                continue

            # Support check
            if z0 > _SUP_EPS:
                supported = False
                for b in placed:
                    bz1 = b.z0 + b.extent_z
                    diff = bz1 - z0
                    if diff < -_SUP_EPS or diff > _SUP_EPS:
                        continue
                    xo = min(b.x0 + b.extent_x, x1) - max(b.x0, x0)
                    if xo <= _SUP_EPS:
                        continue
                    yo = min(b.y0 + b.extent_y, y1) - max(b.y0, y0)
                    if yo > _SUP_EPS:
                        supported = True
                        break
                if not supported:
                    continue

            # Overlap check
            overlaps = False
            for b in placed:
                if x0 >= b.x0 + b.extent_x - _EPS or b.x0 >= x1 - _EPS:
                    continue
                if y0 >= b.y0 + b.extent_y - _EPS or b.y0 >= y1 - _EPS:
                    continue
                if z0 >= b.z0 + b.extent_z - _EPS or b.z0 >= z1 - _EPS:
                    continue
                overlaps = True
                break
            if overlaps:
                continue

            # Score
            if z0 <= _SUP_EPS:
                contact_area = ex * ey
            else:
                contact_area = 0.0
                for b in placed:
                    bz1 = b.z0 + b.extent_z
                    diff = bz1 - z0
                    if diff < -_SUP_EPS or diff > _SUP_EPS:
                        continue
                    xo = min(b.x0 + b.extent_x, x1) - max(b.x0, x0)
                    if xo <= _SUP_EPS:
                        continue
                    yo = min(b.y0 + b.extent_y, y1) - max(b.y0, y0)
                    if yo > _SUP_EPS:
                        contact_area += xo * yo

            max_contact = ex * ey
            contact_ratio = contact_area / max_contact if max_contact > 0 else 0.0
            score = (w_height * z0
                     + w_contact * (1.0 - contact_ratio)
                     + consolidation_penalty
                     + fragmentation_penalty)   # CHANGE: added fragmentation term

            candidates.append(Candidate(
                uld_id=uld_id, point=point,
                orientation=extents, score=score,
            ))

    candidates.sort(key=lambda c: c.score)
    if top_k is not None:
        candidates = candidates[:top_k]
    return candidates
