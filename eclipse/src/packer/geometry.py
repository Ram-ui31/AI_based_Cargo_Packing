"""True 3D placement for a single ULD, via AABB occupancy + pivot-point
candidate generation (the same method py3dbp itself uses internally).

The problem has NO gravity constraint -- the only real rules are: no overlap,
stay within the ULD envelope, respect the weight limit. An earlier version of
this module enforced a "drop to the top of the footprint" rule as a
tractability simplification, but that rule is *not* required by the problem
and it structurally forbids placements a true 3D packer allows (resting a box
in a partial gap next to a taller neighbor). This version drops that rule:
z is a genuine free choice, validated only against real 3D overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

# All 6 ways to assign a box's (length, width, height) to the ULD's (x, y, z) axes.
ORIENTATIONS: list[tuple[int, int, int]] = list(set(permutations(range(3))))


@dataclass
class Placement:
    package_id: str
    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int


def _intersects(a: Placement, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> bool:
    return not (a.x1 <= x0 or x1 <= a.x0 or a.y1 <= y0 or y1 <= a.y0 or a.z1 <= z0 or z1 <= a.z0)


@dataclass
class EMS:
    """A maximal empty axis-aligned box: no smaller empty box strictly
    contains it, and no larger empty box would still be entirely free.
    Candidate origins derived from these (Heightmap.ems_candidates) are a
    COMPLETE representation of remaining placement opportunities, unlike
    pivot_points' corner-adjacent-points-of-placed-boxes heuristic, which
    can structurally miss valid placements in gaps that don't touch any
    existing box's corner (e.g. the far wall behind a "floating" box that
    only touches the floor)."""
    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int

    @property
    def volume(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)


def _ems_overlaps(e: EMS, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> bool:
    return not (e.x1 <= x0 or x1 <= e.x0 or e.y1 <= y0 or y1 <= e.y0 or e.z1 <= z0 or z1 <= e.z0)


def _split_ems(e: EMS, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> list[EMS]:
    """Clip EMS e against a newly-placed box's AABB. Up to 6 remainder
    boxes (one per face of the placement), degenerate (zero/negative
    volume) ones dropped."""
    out = []
    if e.x0 < x0:
        out.append(EMS(e.x0, e.y0, e.z0, x0, e.y1, e.z1))
    if x1 < e.x1:
        out.append(EMS(x1, e.y0, e.z0, e.x1, e.y1, e.z1))
    if e.y0 < y0:
        out.append(EMS(e.x0, e.y0, e.z0, e.x1, y0, e.z1))
    if y1 < e.y1:
        out.append(EMS(e.x0, y1, e.z0, e.x1, e.y1, e.z1))
    if e.z0 < z0:
        out.append(EMS(e.x0, e.y0, e.z0, e.x1, e.y1, z0))
    if z1 < e.z1:
        out.append(EMS(e.x0, e.y0, z1, e.x1, e.y1, e.z1))
    return out


def _ems_contains(a: EMS, b: EMS) -> bool:
    """Is a fully inside b (a is redundant once b exists)? If an item's
    dims fit at a's origin, they also fit at b's origin -- b's extent
    along every axis is >= a's, and b is itself guaranteed empty. So a
    never enables a placement b's own origin doesn't also enable; it's
    lossless to drop it, not a lossy approximation."""
    return (b.x0 <= a.x0 and a.x1 <= b.x1 and b.y0 <= a.y0 and a.y1 <= b.y1
            and b.z0 <= a.z0 and a.z1 <= b.z1)


@dataclass
class Heightmap:
    """Name kept for compatibility with the rest of the codebase -- this is
    now a true 3D AABB-occupancy tracker, not a 2D height grid."""
    length: int  # x extent
    width: int  # y extent
    height: int  # z extent
    weight_limit: float
    max_ems: int = 300  # analogous to pivot_points' cap=400

    def __post_init__(self):
        self.placements: list[Placement] = []
        self.weight_used = 0.0
        self.volume_used = 0
        self.ems_list: list[EMS] = [EMS(0, 0, 0, self.length, self.width, self.height)]

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height

    def orientation_dims(self, l: int, w: int, h: int, orient_idx: int) -> tuple[int, int, int]:
        dims = (l, w, h)
        perm = ORIENTATIONS[orient_idx]
        return dims[perm[0]], dims[perm[1]], dims[perm[2]]

    def fits(self, dx: int, dy: int, dz: int, x: int, y: int, z: int, weight: float) -> bool:
        if x < 0 or y < 0 or z < 0:
            return False
        if x + dx > self.length or y + dy > self.width or z + dz > self.height:
            return False
        if self.weight_used + weight > self.weight_limit + 1e-9:
            return False
        x1, y1, z1 = x + dx, y + dy, z + dz
        for p in self.placements:
            if _intersects(p, x, y, z, x1, y1, z1):
                return False
        return True

    def place(self, package_id: str, dx: int, dy: int, dz: int, x: int, y: int, z: int, weight: float) -> None:
        if not self.fits(dx, dy, dz, x, y, z, weight):
            raise ValueError(f"invalid placement for {package_id} at ({x},{y},{z}) dims ({dx},{dy},{dz})")
        self.weight_used += weight
        self.volume_used += dx * dy * dz
        self.placements.append(Placement(package_id, x, y, z, x + dx, y + dy, z + dz))
        x1, y1, z1 = x + dx, y + dy, z + dz
        new_ems_list = []
        for e in self.ems_list:
            if _ems_overlaps(e, x, y, z, x1, y1, z1):
                new_ems_list.extend(_split_ems(e, x, y, z, x1, y1, z1))
            else:
                new_ems_list.append(e)
        self.ems_list = self._prune_ems(new_ems_list)

    def _prune_ems(self, ems_list: list[EMS]) -> list[EMS]:
        """(1) Containment elimination -- lossless, always applied (see
        _ems_contains docstring). (2) Hard cap at self.max_ems, keeping
        the largest-by-volume survivors DETERMINISTICALLY -- unlike
        pivot_points(cap)'s arbitrary set-iteration-order truncation."""
        by_vol_desc = sorted(ems_list, key=lambda e: -e.volume)
        survivors: list[EMS] = []
        for e in by_vol_desc:
            if not any(_ems_contains(e, s) for s in survivors):
                survivors.append(e)
        if len(survivors) > self.max_ems:
            survivors = survivors[: self.max_ems]
        return survivors

    def ems_candidates(self, cap: int = 400, min_dims: tuple[int, int, int] | None = None) -> list[tuple[int, int, int]]:
        """Candidate origins: the lower corner of every surviving maximal
        empty box, optionally filtered to boxes that could plausibly fit
        an item of at least min_dims, capped deterministically by volume
        (largest-first) rather than arbitrary set order. Same return
        shape as pivot_points() -- a drop-in alternative origin source."""
        ems = self.ems_list
        if min_dims is not None:
            smallest_edge = min(min_dims)
            ems = [e for e in ems
                   if min(e.x1 - e.x0, e.y1 - e.y0, e.z1 - e.z0) >= smallest_edge]
        ems = sorted(ems, key=lambda e: -e.volume)
        pts: list[tuple[int, int, int]] = []
        seen = set()
        for e in ems:
            origin = (e.x0, e.y0, e.z0)
            if origin not in seen:
                seen.add(origin)
                pts.append(origin)
            if len(pts) >= cap:
                break
        return pts if pts else [(0, 0, 0)]

    def pivot_points(self, cap: int = 400) -> list[tuple[int, int, int]]:
        """Candidate origins: py3dbp's own pivot-point method -- for every
        already-placed box, one pivot along each axis (right of it, above
        it, behind it), plus the origin when the ULD is still empty."""
        if not self.placements:
            return [(0, 0, 0)]
        pts = set()
        for p in self.placements:
            pts.add((p.x1, p.y0, p.z0))
            pts.add((p.x0, p.y1, p.z0))
            pts.add((p.x0, p.y0, p.z1))
        pts_list = list(pts)
        if len(pts_list) > cap:
            pts_list = pts_list[:cap]
        return pts_list

    def utilization(self) -> float:
        return self.volume_used / self.volume if self.volume else 0.0
