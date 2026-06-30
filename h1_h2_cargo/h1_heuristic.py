"""
H1 Heuristic — Economy package scorer for the greedy binary-search split.

Role in the pipeline (see Figure 3):
    Economy Packages --> [H1 Heuristic] --Sort--> [Binary Search]
                                                      |        |
                                                    Set 1    Set 2

Final H1 score (higher = higher priority, placed earlier):
    score = w_density  * normalised_cost_density
          + w_cost     * normalised_delay_cost
          + w_compact  * compactness

CHANGE (cost reduction):
    Boosted w_density 3.0 -> 5.0 and w_cost 1.0 -> 2.0 so that high-value
    economy packages (expensive per unit volume) are sorted to the front of
    Set 1 and packed first. Reduced w_compact 1.0 -> 0.5 because compactness
    is a weak proxy for packability and dilutes the cost signal.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from geometry import Package


@dataclass
class ScoredPackage:
    package: Package
    score: float


def score_economy_packages(
    packages: List[Package],
    w_density: float = 5.0,   # CHANGED: was 3.0
    w_cost: float = 2.0,      # CHANGED: was 1.0
    w_compact: float = 0.5,   # CHANGED: was 1.0
) -> List[ScoredPackage]:
    economy = [p for p in packages if not p.is_priority]
    if not economy:
        return []

    densities = [_cost_density(p) for p in economy]
    costs     = [p.delay_cost     for p in economy]
    compacts  = [_compactness(p)  for p in economy]

    norm_density = _normalise(densities)
    norm_cost    = _normalise(costs)
    norm_compact = _normalise(compacts)

    scored = []
    for i, pkg in enumerate(economy):
        s = (w_density * norm_density[i]
             + w_cost    * norm_cost[i]
             + w_compact * norm_compact[i])
        scored.append(ScoredPackage(package=pkg, score=s))

    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored


def sort_by_h1(packages: List[Package], **score_kwargs) -> List[Package]:
    priority = [p for p in packages if p.is_priority]
    scored   = score_economy_packages(packages, **score_kwargs)
    return priority + [sp.package for sp in scored]


def _cost_density(p: Package) -> float:
    vol = p.volume
    return p.delay_cost / vol if vol > 1e-9 else 0.0


def _compactness(p: Package) -> float:
    dims = sorted([p.length, p.width, p.height])
    if dims[2] < 1e-9:
        return 0.0
    return dims[0] / dims[2]


def _normalise(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]
