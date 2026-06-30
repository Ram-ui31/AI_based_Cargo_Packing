"""
H2 Heuristic — Leftover economy package scorer for retroactive allocation.

Role in the pipeline (see Figure 3):
    [Pack] --> Leftover Economy Packages --> [H2 Heuristic] --> [Pack] --> Final Solution

Final H2 score (higher = higher priority):
    score = w_fit * gap_fit_score + w_cost * normalised_delay_cost + w_small * normalised_smallness

CHANGE (cost reduction):
    Boosted w_cost 2.0 -> 4.0: at the H2 stage, physical fit is already
    filtered by the packer itself — the remaining decision should be almost
    entirely cost-driven, so expensive leftover packages are placed first.
    Reduced w_fit 3.0 -> 2.0 and w_small 1.0 -> 0.5 to give cost more weight
    without changing the total scale dramatically.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from geometry import Package
from extreme_points import ExtremePointTracker


@dataclass
class H2ScoredPackage:
    package: Package
    score: float


def score_leftover_packages(
    leftover_packages: List[Package],
    trackers: Optional[Dict[str, ExtremePointTracker]] = None,
    w_fit: float = 2.0,    # CHANGED: was 3.0
    w_cost: float = 4.0,   # CHANGED: was 2.0
    w_small: float = 0.5,  # CHANGED: was 1.0
) -> List[H2ScoredPackage]:
    if not leftover_packages:
        return []

    costs     = [p.delay_cost for p in leftover_packages]
    volumes   = [p.volume     for p in leftover_packages]
    smallness = [1.0 / (v + 1e-9) for v in volumes]
    fits      = [_best_gap_fit(p, trackers) for p in leftover_packages] if trackers else [0.0] * len(leftover_packages)

    norm_fit   = _normalise(fits)
    norm_cost  = _normalise(costs)
    norm_small = _normalise(smallness)

    scored = []
    for i, pkg in enumerate(leftover_packages):
        s = (w_fit * norm_fit[i] + w_cost * norm_cost[i] + w_small * norm_small[i])
        scored.append(H2ScoredPackage(package=pkg, score=s))

    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored


def sort_by_h2(
    leftover_packages: List[Package],
    trackers: Optional[Dict[str, ExtremePointTracker]] = None,
    **score_kwargs,
) -> List[Package]:
    scored = score_leftover_packages(leftover_packages, trackers, **score_kwargs)
    return [sp.package for sp in scored]


def _best_gap_fit(pkg: Package, trackers: Dict[str, ExtremePointTracker]) -> float:
    pkg_dims = sorted([pkg.length, pkg.width, pkg.height])
    best = 0.0
    for tracker in trackers.values():
        uld = tracker.uld
        for (px, py, pz) in tracker.points:
            rem = sorted([uld.length - px, uld.width - py, uld.height - pz])
            if rem[0] >= pkg_dims[0] - 1e-9 and rem[1] >= pkg_dims[1] - 1e-9 and rem[2] >= pkg_dims[2] - 1e-9:
                waste = (rem[0] - pkg_dims[0]) + (rem[1] - pkg_dims[1]) + (rem[2] - pkg_dims[2])
                max_dim = max(uld.length, uld.width, uld.height)
                snugness = 1.0 - min(waste / (3 * max_dim + 1e-9), 1.0)
                best = max(best, snugness)
    return best


def _normalise(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / span for v in values]
