import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl_packer", "src"))

import numpy as np
import pytest

from geometry import Heightmap, ORIENTATIONS


def test_orientations_has_6_unique_axis_assignments():
    assert len(ORIENTATIONS) == 6


def test_single_placement_at_origin():
    hm = Heightmap(length=100, width=100, height=100, weight_limit=1000)
    hm.place("P1", dx=10, dy=10, dz=10, x=0, y=0, z=0, weight=5)
    assert hm.placements[0].x1 == 10
    assert hm.utilization() == pytest.approx(1000 / 1_000_000)


def test_overlapping_footprint_and_z_range_rejected():
    hm = Heightmap(length=100, width=100, height=100, weight_limit=1000)
    hm.place("P1", dx=10, dy=10, dz=10, x=0, y=0, z=0, weight=5)
    assert not hm.fits(dx=10, dy=10, dz=10, x=5, y=5, z=5, weight=1)  # overlaps in all 3 axes


def test_no_gravity_box_can_float_over_a_gap():
    """The core capability the old drop-rule model could not represent:
    resting a box at a lower z than the tallest neighbor in its footprint,
    as long as it doesn't actually intersect anything."""
    hm = Heightmap(length=100, width=100, height=100, weight_limit=1000)
    hm.place("Tall", dx=10, dy=10, dz=50, x=0, y=0, z=0, weight=5)
    # a box beside it, NOT stacked on top, at z=0 -- floating is fine, no support required
    assert hm.fits(dx=10, dy=10, dz=10, x=20, y=0, z=0, weight=1)
    # a box that would need to rest at z=50 in a drop model can instead go at z=0
    # right next to the tall box without touching it
    hm.place("Beside", dx=10, dy=10, dz=10, x=20, y=0, z=0, weight=1)
    assert len(hm.placements) == 2


def test_adjacent_box_same_height_does_not_collide():
    hm = Heightmap(length=100, width=100, height=100, weight_limit=1000)
    hm.place("P1", dx=10, dy=10, dz=10, x=0, y=0, z=0, weight=5)
    assert hm.fits(dx=10, dy=10, dz=5, x=10, y=0, z=0, weight=5)


def test_stacking_directly_on_top_works():
    hm = Heightmap(length=100, width=100, height=100, weight_limit=1000)
    hm.place("P1", dx=10, dy=10, dz=10, x=0, y=0, z=0, weight=5)
    assert hm.fits(dx=10, dy=10, dz=5, x=0, y=0, z=10, weight=5)
    hm.place("P2", dx=10, dy=10, dz=5, x=0, y=0, z=10, weight=5)
    assert len(hm.placements) == 2


def test_no_overlap_ever_produced_random_stress():
    rng = np.random.default_rng(0)
    hm = Heightmap(length=50, width=50, height=50, weight_limit=1e9)
    occupied = np.zeros((50, 50, 50), dtype=bool)
    n_placed = 0
    for _ in range(500):
        dx, dy, dz = rng.integers(1, 10, size=3).tolist()
        x = int(rng.integers(0, 50 - dx + 1))
        y = int(rng.integers(0, 50 - dy + 1))
        z = int(rng.integers(0, 50 - dz + 1))
        if not hm.fits(dx, dy, dz, x, y, z, weight=0):
            continue
        assert not occupied[x:x + dx, y:y + dy, z:z + dz].any()
        hm.place("p", dx, dy, dz, x, y, z, weight=0)
        occupied[x:x + dx, y:y + dy, z:z + dz] = True
        n_placed += 1
    assert n_placed > 0


def test_height_limit_enforced():
    hm = Heightmap(length=10, width=10, height=15, weight_limit=1000)
    assert not hm.fits(dx=10, dy=10, dz=10, x=0, y=0, z=10, weight=1)  # 10+10 > 15


def test_weight_limit_enforced():
    hm = Heightmap(length=100, width=100, height=100, weight_limit=10)
    assert not hm.fits(dx=1, dy=1, dz=1, x=0, y=0, z=0, weight=11)
    hm.place("P1", dx=1, dy=1, dz=1, x=0, y=0, z=0, weight=9)
    assert not hm.fits(dx=1, dy=1, dz=1, x=5, y=5, z=5, weight=2)


def test_out_of_bounds_rejected():
    hm = Heightmap(length=10, width=10, height=10, weight_limit=1000)
    assert not hm.fits(dx=5, dy=5, dz=5, x=8, y=0, z=0, weight=0)
    assert not hm.fits(dx=5, dy=5, dz=5, x=-1, y=0, z=0, weight=0)
    assert not hm.fits(dx=5, dy=5, dz=5, x=0, y=0, z=-1, weight=0)


def test_place_raises_on_invalid():
    hm = Heightmap(length=10, width=10, height=10, weight_limit=1000)
    with pytest.raises(ValueError):
        hm.place("P1", dx=20, dy=5, dz=5, x=0, y=0, z=0, weight=0)


def test_pivot_points_start_at_origin_when_empty():
    hm = Heightmap(length=50, width=50, height=100, weight_limit=1000)
    assert hm.pivot_points() == [(0, 0, 0)]


def test_pivot_points_grow_along_all_3_axes_after_placement():
    hm = Heightmap(length=50, width=50, height=100, weight_limit=1000)
    hm.place("P1", dx=10, dy=10, dz=5, x=0, y=0, z=0, weight=1)
    pts = set(hm.pivot_points())
    assert (10, 0, 0) in pts  # right along x
    assert (0, 10, 0) in pts  # along y
    assert (0, 0, 5) in pts  # along z (stacking)


def test_utilization_matches_volume_used():
    hm = Heightmap(length=10, width=10, height=10, weight_limit=1000)
    hm.place("P1", dx=5, dy=5, dz=5, x=0, y=0, z=0, weight=1)
    assert hm.utilization() == pytest.approx(125 / 1000)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
