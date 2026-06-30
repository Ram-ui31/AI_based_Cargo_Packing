"""
Extreme point tracker — copy-free, undo-based design.

The key insight: the beam search clones a node, makes ONE placement, then
discards the clone if it's pruned from the beam. deepcopy of the tracker
was 56% of total runtime.

Solution: instead of cloning, the tracker supports a checkpoint/rollback
protocol. The beam search checkpoints before trying a candidate, commits
the placement, evaluates the node, then either keeps it (no rollback needed
— the node IS the live branch) or rolls back for the next candidate.

Because the beam search CLONES nodes that survive into the next round, we
still need a copy path — but it only runs beam_width times per package
(the survivors), not beam_width × candidates_per_uld times (all trials).

Point generation: standard Crainic 3-point projection, augmented with
shadow projections capped at MAX_POINTS. Uses a set for O(1) dedup.
"""
from __future__ import annotations
from typing import List, Set, Tuple, Optional
import copy

from geometry import ULD, PlacedBox

Point = Tuple[float, float, float]
MAX_POINTS = 200


class ExtremePointTracker:
    def __init__(self, uld: ULD):
        self.uld = uld
        self._point_set: Set[Point] = {(0.0, 0.0, 0.0)}
        self.placed: List[PlacedBox] = []
        self.total_weight: float = 0.0
        # checkpoint stack
        self._checkpoints: List[tuple] = []

    @property
    def points(self) -> List[Point]:
        return list(self._point_set)

    # -- checkpoint / rollback ---------------------------------------------

    def checkpoint(self) -> None:
        """Save current state onto the internal stack."""
        self._checkpoints.append((
            frozenset(self._point_set),
            list(self.placed),
            self.total_weight,
        ))

    def rollback(self) -> None:
        """Restore to the last checkpoint (pops it)."""
        pts, placed, weight = self._checkpoints.pop()
        self._point_set = set(pts)
        self.placed = placed
        self.total_weight = weight

    def pop_checkpoint(self) -> None:
        """Discard the last checkpoint without rolling back (commit accepted)."""
        if self._checkpoints:
            self._checkpoints.pop()

    def __deepcopy__(self, memo):
        """Custom deepcopy: skip the checkpoint stack (never needed on a clone)."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.uld = self.uld           # frozen dataclass, safe to share
        new._point_set = set(self._point_set)
        new.placed = list(self.placed)   # PlacedBox is frozen, safe to share refs
        new.total_weight = self.total_weight
        new._checkpoints = []        # fresh clone has no pending checkpoints
        return new

    # -- public API --------------------------------------------------------

    def commit(self, box: PlacedBox) -> None:
        used = (box.x0, box.y0, box.z0)
        self._point_set.discard(used)
        self.placed.append(box)

        candidates: List[Point] = [
            (box.x1, box.y0, box.z0),
            (box.x0, box.y1, box.z0),
            (box.x0, box.y0, box.z1),
        ]
        if len(self._point_set) < MAX_POINTS:
            for other in self.placed[:-1]:
                candidates.append((box.x1, other.y0, other.z0))
                candidates.append((box.x1, other.y1, other.z0))
                candidates.append((box.x1, other.y0, other.z1))
                candidates.append((other.x0, box.y1, other.z0))
                candidates.append((other.x1, box.y1, other.z0))
                candidates.append((other.x0, box.y1, other.z1))
                candidates.append((other.x0, other.y0, box.z1))
                candidates.append((other.x1, other.y0, box.z1))
                candidates.append((other.x0, other.y1, box.z1))

        for p in candidates:
            if p not in self._point_set and self._in_bounds(p) and not self._buried(p):
                self._point_set.add(p)
                if len(self._point_set) >= MAX_POINTS:
                    break

        self._point_set = {p for p in self._point_set if not self._buried_in(p, box)}

    def add_weight(self, w: float) -> None:
        self.total_weight += w

    def remaining_weight_capacity(self) -> float:
        return self.uld.weight_limit - self.total_weight

    def _in_bounds(self, p: Point, eps: float = 1e-9) -> bool:
        x, y, z = p
        return (-eps <= x <= self.uld.length + eps
                and -eps <= y <= self.uld.width + eps
                and -eps <= z <= self.uld.height + eps)

    def _buried(self, p: Point, eps: float = 1e-9) -> bool:
        x, y, z = p
        for box in self.placed:
            if (box.x0 + eps < x < box.x1 - eps
                    and box.y0 + eps < y < box.y1 - eps
                    and box.z0 + eps < z < box.z1 - eps):
                return True
        return False

    def _buried_in(self, p: Point, box: PlacedBox, eps: float = 1e-9) -> bool:
        x, y, z = p
        return (box.x0 + eps < x < box.x1 - eps
                and box.y0 + eps < y < box.y1 - eps
                and box.z0 + eps < z < box.z1 - eps)
