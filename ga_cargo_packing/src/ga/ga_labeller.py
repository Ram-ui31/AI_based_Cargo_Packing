"""
ga_labeller.py — GALabeller, a Labeller (matching the good-il-over-greedy(c)
Labeller strategy-pattern interface) that wraps GAPipeline as the IL label
source, mirroring H1H2Labeller's shape exactly:
    good-il-over-greedy(c)/src/labeller.py

Not a subclass of that Labeller ABC (this package doesn't import src.il
to avoid a circular dependency) -- src/il/labeller.py wraps THIS
class in its own Labeller subclass instead. GALabeller only needs to expose
the same `label(packages_df, ulds_df, tag=None, pkg_chunk_idx=None,
uld_chunk_idx=None) -> {Package_ID: ULD_ID | 'NONE'}` method.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_H1H2_SRC = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', 'h1_h2_cargo', 'src'))
if _H1H2_SRC not in sys.path:
    sys.path.insert(0, _H1H2_SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from geometry import Package, ULD    # noqa: E402
from ga_pipeline import GAPipeline   # noqa: E402


class GALabeller:
    """
    cache : optional {(tag, uld_chunk_idx, pkg_chunk_idx): assignment}
        precomputed by scripts/precompute_ga_cache.py. Falls back to a live
        GAPipeline solve on any cache miss or when tag is None.
    """

    def __init__(self, cache=None, pop_size=16, max_generations=20,
                 patience=6, gene_contribution_ratio=0.65, seed=None,
                 time_budget_seconds=90.0):
        self._cache = cache or {}
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.patience = patience
        self.gene_contribution_ratio = gene_contribution_ratio
        self.seed = seed
        self.time_budget_seconds = time_budget_seconds

    def label(self, packages_df, ulds_df, tag=None, pkg_chunk_idx=None, uld_chunk_idx=None):
        cache_key = (tag, uld_chunk_idx, pkg_chunk_idx) if tag is not None else None
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        packages = [
            Package(
                id=row.Package_ID, length=row.Length, width=row.Width,
                height=row.Height, weight=row.Weight,
                is_priority=(row.Type == 'Priority'), delay_cost=row.Delay_Cost,
            )
            for row in packages_df.itertuples()
        ]
        ulds = [
            ULD(
                id=row.ULD_ID, length=row.Length, width=row.Width,
                height=row.Height, weight_limit=row.Weight_Limit,
            )
            for row in ulds_df.itertuples()
        ]

        result = GAPipeline(
            ulds=ulds, packages=packages,
            pop_size=self.pop_size, max_generations=self.max_generations,
            patience=self.patience, gene_contribution_ratio=self.gene_contribution_ratio,
            seed=self.seed, time_budget_seconds=self.time_budget_seconds,
        ).solve()

        assignment = {box.package_id: box.uld_id for box in result.placed_boxes}
        for pid in result.left_behind + result.unplaceable:
            assignment[pid] = 'NONE'
        if cache_key is not None:
            self._cache[cache_key] = assignment
        return assignment
