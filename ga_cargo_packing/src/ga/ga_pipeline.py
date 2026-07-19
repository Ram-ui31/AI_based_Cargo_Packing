"""
ga_pipeline.py — orchestrator: robust priority-only pack (full ULD fleet,
reusing h1_h2_cargo's own rescue/nuclear-eviction) -> uld_partition -> GA
split -> rigorous greedy_pack per bucket, seeded from the priority pack's
leftover space -> merged PackResult.

Imports h1_h2_cargo's own partition_ulds / greedy_pack / GreedyPipeline /
geometry primitives unmodified (sys.path insert, same pattern as
GALabeller/H1H2Labeller). Priority packages are never part of the GA's gene
string and are packed FIRST, against the full ULD fleet, via a throwaway
GreedyPipeline(packages=priority_only) -- this reuses that pipeline's own
hardened rescue_priority/nuclear_eviction fallbacks (with no Economy
present to fail against) rather than reimplementing them, guaranteeing every
Priority package is packed (condition 1) the same way the existing
H1H2-labelled pipeline already does. The GA (ga_solver.run_ga) then only
decides bucket membership for Economy packages (gene in {0,1,2}); the actual
per-package placement inside each bucket is delegated to h1_h2_cargo's own
rigorous multi-sort greedy_pack, seeded from the priority pack's remaining
trackers, run once per bucket for the GA's winning individual -- not during
the generation loop, where ga_solver's cheap trial pack is used for speed.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_H1H2_SRC = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', 'h1_h2_cargo', 'src'))
if _H1H2_SRC not in sys.path:
    sys.path.insert(0, _H1H2_SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from geometry import Package, ULD, PlacedBox            # noqa: E402
from uld_partition import partition_ulds                # noqa: E402
from greedy_pack import greedy_pack, PackResult          # noqa: E402
from greedy_pipeline import GreedyPipeline               # noqa: E402

from ga_solver import run_ga, GAResult                   # noqa: E402


class GAPipeline:
    def __init__(
        self,
        ulds: List[ULD],
        packages: List[Package],
        fill_target: float = 1.2,
        pop_size: int = 16,
        max_generations: int = 20,
        patience: int = 6,
        gene_contribution_ratio: float = 0.65,
        candidates_per_uld: int = 5,
        seed: Optional[int] = None,
        time_budget_seconds: float = 90.0,
    ):
        self.ulds = ulds
        self.packages = packages
        self.fill_target = fill_target
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.patience = patience
        self.gene_contribution_ratio = gene_contribution_ratio
        self.candidates_per_uld = candidates_per_uld
        self.seed = seed
        self.time_budget_seconds = time_budget_seconds

        self.priority_packages = [p for p in packages if p.is_priority]
        self.economy_packages = [p for p in packages if not p.is_priority]
        self.priority_ids = {p.id for p in self.priority_packages}

    def solve(self) -> PackResult:
        # Priority packages are never part of the GA's gene string -- pack
        # them first, against the FULL ULD fleet, via a throwaway
        # GreedyPipeline holding only priority packages. With zero Economy
        # present to compete for space, this reduces to that pipeline's
        # pass1 (multi-sort greedy_pack) + rescue_priority + nuclear_eviction
        # fallbacks -- the same hardened machinery the existing H1H2-labelled
        # pipeline relies on to guarantee every Priority package is packed.
        priority_only = GreedyPipeline(
            ulds=self.ulds, packages=self.priority_packages, k_penalty=0.0,
            candidates_per_uld=self.candidates_per_uld,
        ).solve()
        priority_trackers = priority_only.trackers

        partition = partition_ulds(
            ulds=self.ulds, packages=self.packages, fill_target=self.fill_target,
        )
        priority_uld_ids = {u.id for u in partition.priority_ulds}
        other_uld_ids = {u.id for u in partition.other_ulds}

        ga_result: GAResult = run_ga(
            priority_packages=self.priority_packages,
            economy_packages=self.economy_packages,
            priority_ulds=partition.priority_ulds,
            other_ulds=partition.other_ulds,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            patience=self.patience,
            gene_contribution_ratio=self.gene_contribution_ratio,
            seed=self.seed,
            time_budget_seconds=self.time_budget_seconds,
        )
        self.ga_result = ga_result  # exposed for logging/inspection

        genes = ga_result.best.genes
        econ_by_idx = {i: p for i, p in enumerate(self.economy_packages)}
        bucket1_econ = [econ_by_idx[i] for i, g in enumerate(genes) if g == 2]
        bucket2_econ = [econ_by_idx[i] for i, g in enumerate(genes) if g == 1]

        # Rigorous final pack per bucket (h1_h2_cargo's own multi-sort/retry
        # greedy_pack, unmodified) -- cheap trial-pack was for GA scoring
        # only -- seeded from the leftover space priority_only left behind in
        # each bucket's ULDs, so Economy never displaces an already-placed
        # Priority package. Chained (not both independently seeded from
        # priority_trackers): partition_ulds can make priority_ulds and
        # other_ulds overlap (its own single-ULD edge case), in which case
        # pass2 must see whatever pass1 just placed there too, or the two
        # bucket packs could double-book the same physical space.
        running_trackers = dict(priority_trackers)
        pass1 = self._pack_bucket(bucket1_econ, partition.priority_ulds, priority_uld_ids, running_trackers)
        if pass1:
            running_trackers.update(pass1.trackers)
        pass2 = self._pack_bucket(bucket2_econ, partition.other_ulds, other_uld_ids, running_trackers)

        return self._merge(priority_only, pass1, pass2)

    def _pack_bucket(self, econ_packages, bucket_ulds, bucket_uld_ids, seed_trackers):
        if not econ_packages or not bucket_ulds:
            return None
        seeded = {uid: copy.deepcopy(tr) for uid, tr in seed_trackers.items() if uid in bucket_uld_ids}
        return greedy_pack(
            packages=econ_packages, ulds=bucket_ulds,
            candidates_per_uld=self.candidates_per_uld, trackers=seeded,
        )

    def _merge(self, priority_only: PackResult, pass1: Optional[PackResult],
               pass2: Optional[PackResult]) -> PackResult:
        econ_boxes = (pass1.placed_boxes if pass1 else []) + (pass2.placed_boxes if pass2 else [])
        all_placed_ids = (
            {b.package_id for b in priority_only.placed_boxes} | {b.package_id for b in econ_boxes}
        )
        final_left_behind = [p.id for p in self.economy_packages if p.id not in all_placed_ids]
        final_unplaceable = [p.id for p in self.priority_packages if p.id not in all_placed_ids]

        final_trackers = dict(priority_only.trackers)
        if pass1:
            final_trackers.update(pass1.trackers)
        if pass2:
            final_trackers.update(pass2.trackers)

        return PackResult(
            trackers=final_trackers,
            placed_boxes=priority_only.placed_boxes + econ_boxes,
            left_behind=final_left_behind,
            unplaceable=final_unplaceable,
            priority_ids=self.priority_ids,
        )
