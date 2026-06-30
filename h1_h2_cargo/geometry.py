"""
Core geometric primitives for the cargo packing engine.

Coordinate convention (per problem statement):
  - Origin (0,0,0) is the front-left-bottom corner of the ULD.
  - X axis = length, Y axis = width, Z axis = height.
  - A placed box is stored as two diagonal corners:
        (x0, y0, z0) -> (x1, y1, z1)
    where x1 = x0 + extent_x, etc., for whichever orientation was chosen.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import Tuple


# ---------------------------------------------------------------------------
# Static input data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Package:
    """A package to be packed. Immutable input data."""
    id: str
    length: float
    width: float
    height: float
    weight: float
    is_priority: bool
    delay_cost: float = 0.0  # only meaningful for Economy packages

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def orientations(self) -> Tuple[Tuple[float, float, float], ...]:
        """
        All 6 axis-aligned orientations of this package, as (extent_x, extent_y,
        extent_z) tuples. We dedupe permutations in case of cube-like packages
        (e.g. length == width) so we don't waste search effort on identical
        geometric outcomes.
        """
        dims = (self.length, self.width, self.height)
        seen = set()
        result = []
        for p in permutations(dims):
            if p not in seen:
                seen.add(p)
                result.append(p)
        return tuple(result)


@dataclass(frozen=True)
class ULD:
    """A Unit Load Device (container). Immutable input data."""
    id: str
    length: float
    width: float
    height: float
    weight_limit: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


# ---------------------------------------------------------------------------
# Placement result
# ---------------------------------------------------------------------------

@dataclass
class PlacedBox:
    """A package committed to a specific ULD at a specific position/orientation."""
    package_id: str
    uld_id: str
    x0: float
    y0: float
    z0: float
    extent_x: float
    extent_y: float
    extent_z: float

    @property
    def x1(self) -> float:
        return self.x0 + self.extent_x

    @property
    def y1(self) -> float:
        return self.y0 + self.extent_y

    @property
    def z1(self) -> float:
        return self.z0 + self.extent_z

    def as_output_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Matches the required output format: x0,y0,z0,x1,y1,z1."""
        return (self.x0, self.y0, self.z0, self.x1, self.y1, self.z1)


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def boxes_overlap(a: PlacedBox, b: PlacedBox, eps: float = 1e-9) -> bool:
    """
    Axis-aligned bounding box (AABB) overlap test. Two boxes that merely touch
    (share a face with zero-volume intersection) are NOT considered overlapping
    -- that's exactly how flush placements are supposed to work.
    """
    if a.x0 >= b.x1 - eps or b.x0 >= a.x1 - eps:
        return False
    if a.y0 >= b.y1 - eps or b.y0 >= a.y1 - eps:
        return False
    if a.z0 >= b.z1 - eps or b.z0 >= a.z1 - eps:
        return False
    return True


def fits_in_bounds(x0: float, y0: float, z0: float,
                    ex: float, ey: float, ez: float,
                    uld: ULD, eps: float = 1e-9) -> bool:
    """Checks the candidate box lies fully within the ULD's outer dimensions."""
    if x0 < -eps or y0 < -eps or z0 < -eps:
        return False
    if x0 + ex > uld.length + eps:
        return False
    if y0 + ey > uld.width + eps:
        return False
    if z0 + ez > uld.height + eps:
        return False
    return True
