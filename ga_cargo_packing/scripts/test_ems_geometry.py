"""
test_ems_geometry.py -- minimal, hand-verifiable sanity check that
Heightmap.ems_candidates() recovers placement opportunities
Heightmap.pivot_points() structurally cannot.

Scenario: place one box that touches only the floor (z=0), not the x=0 or
y=0 walls -- a "floating" box. pivot_points() only ever extends outward
from an existing box's OWN corner (right/above/behind of it), so it can
never regenerate a wall-adjacent origin unless some other box happens to
touch that wall too. EMS decomposition tracks the true remaining free
space directly, so it recovers (0,0,0) immediately.

Usage:
    python scripts/test_ems_geometry.py
"""
from __future__ import annotations
import os
import sys

RL_PACKER_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'rl_packer', 'src',
)
sys.path.insert(0, os.path.abspath(RL_PACKER_SRC))
from geometry import Heightmap  # noqa: E402


def main():
    hm = Heightmap(length=10, width=10, height=10, weight_limit=1e9)
    hm.place('P', dx=3, dy=3, dz=4, x=3, y=3, z=0, weight=1.0)

    pivots = set(hm.pivot_points(cap=400))
    ems_origins = set(hm.ems_candidates(cap=400))

    print('pivot_points:', sorted(pivots))
    print('ems_candidates:', sorted(ems_origins))
    print('ems_list (raw):', hm.ems_list)

    assert (0, 0, 0) not in pivots, \
        'expected pivot_points to lose the origin once anything is placed'
    assert (0, 0, 0) in ems_origins, \
        'expected ems_candidates to recover the origin -- the slab x:[0,3) is still empty'
    assert hm.fits(dx=2, dy=2, dz=2, x=0, y=0, z=0, weight=1.0), \
        'expected (0,0,0) to be a genuinely valid placement'

    print('\nAll assertions passed: EMS recovers a valid placement pivot_points misses.')


if __name__ == '__main__':
    main()
