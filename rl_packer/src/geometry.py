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
class Heightmap:
    """Name kept for compatibility with the rest of the codebase -- this is
    now a true 3D AABB-occupancy tracker, not a 2D height grid."""
    length: int  # x extent
    width: int  # y extent
    height: int  # z extent
    weight_limit: float

    def __post_init__(self):
        self.placements: list[Placement] = []
        self.weight_used = 0.0
        self.volume_used = 0

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
