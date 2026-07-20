"""
Simple, classic corner-point 3D placement geometry -- shared by all
heuristics in this folder. Independent of the EMS-based geometry used
elsewhere in this project (cargoism/git/*/src/packer/geometry.py).

Candidate placement points start at the container origin; after each
placement, three new candidate points are generated at the far corners of
the newly placed box (projected along each axis). Candidates are scanned
bottom-first (z, then y, then x), and all 6 axis-permutations of a
package's dimensions are tried at each candidate point.
"""
from itertools import permutations


class BinState:
    """Tracks placed boxes and candidate corner points for one ULD."""

    def __init__(self, length, width, height, weight_limit):
        self.L, self.W, self.H = length, width, height
        self.weight_limit = weight_limit
        self.weight_used = 0.0
        self.boxes = []  # list of (x0,y0,z0,x1,y1,z1)
        self.corners = [(0.0, 0.0, 0.0)]

    def _overlaps(self, x0, y0, z0, x1, y1, z1):
        for bx0, by0, bz0, bx1, by1, bz1 in self.boxes:
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0 and z0 < bz1 and z1 > bz0:
                return True
        return False

    def try_place(self, dims, weight):
        """Try every corner point (bottom-first scan) x every orientation.
        Returns True and mutates state if placed, else False (and leaves
        state unchanged)."""
        if self.weight_used + weight > self.weight_limit + 1e-6:
            return False

        self.corners.sort(key=lambda p: (p[2], p[1], p[0]))
        seen_orients = set(permutations(dims))

        for (x, y, z) in self.corners:
            for (dx, dy, dz) in seen_orients:
                x1, y1, z1 = x + dx, y + dy, z + dz
                if x1 > self.L + 1e-6 or y1 > self.W + 1e-6 or z1 > self.H + 1e-6:
                    continue
                if self._overlaps(x, y, z, x1, y1, z1):
                    continue
                self.boxes.append((x, y, z, x1, y1, z1))
                self.weight_used += weight
                self.corners.remove((x, y, z))
                self.corners.extend([(x1, y, z), (x, y1, z), (x, y, z1)])
                return True
        return False
