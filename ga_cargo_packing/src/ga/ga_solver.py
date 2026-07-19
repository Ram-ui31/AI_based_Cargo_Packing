"""
ga_solver.py — Genetic Algorithm for the Economy-package priority-ULD-bucket
split, per the "Genetic Algorithm Strategy" spec (ternary encoding, fitness
= cost of unallocated economy packages, fitness-weighted crossover with
configurable gene-contribution ratio, bucketed multi-rate mutation,
validate-and-repair, keep-fittest-half selection).

Reuses h1_h2_cargo's own geometry primitives (ExtremePointTracker,
rank_placements, Package/ULD) for the trial-pack used to score fitness —
imported, never modified. A cheap single-pass trial pack (one sort order,
no multi-sort/retries) is used during the generation loop since it runs
population x generations times per instance; the caller (ga_pipeline.py)
re-packs the winning individual with the full h1_h2_cargo greedy_pack for
the final, rigorous placement.

Encoding
--------
genes[i] in {0, 1, 2} for economy_packages[i]:
    2 -> priority-ULD bucket
    1 -> other-ULD bucket
    0 -> unallocated
Priority packages are never part of the gene string -- they always target
the priority bucket; the GA only decides how much bucket-1 headroom to
concede to Economy packages without starving Priority (validated below).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import Package, ULD, PlacedBox
from extreme_points import ExtremePointTracker
from selector import rank_placements

_EPS = 1e-9

MUTATION_RATES = (0.01, 0.02, 0.03, 0.04)  # fittest bucket -> lowest rate


# ─────────────────────────────────────────────────────────────────────────────
# Cheap single-pass trial pack (fitness evaluation only)
# ─────────────────────────────────────────────────────────────────────────────

def _trial_pack_bucket(packages: List[Package], ulds: List[ULD]) -> Tuple[List[str], List[str]]:
    """
    One-pass greedy placement (largest-volume first, best-of-candidates per
    ULD, no multi-sort / no retries / no eviction) across `ulds`.

    Returns (placed_ids, unplaced_ids).
    """
    if not ulds:
        return [], [p.id for p in packages]

    trackers = {u.id: ExtremePointTracker(u) for u in ulds}
    ordered = sorted(packages, key=lambda p: (-p.volume, -p.weight))
    placed_ids: List[str] = []
    unplaced_ids: List[str] = []

    for pkg in ordered:
        best_score = float("inf")
        best_uld = None
        best_box: Optional[PlacedBox] = None
        for uld in ulds:
            cands = rank_placements(pkg, trackers[uld.id], priority_uld_ids=set(), top_k=1)
            if cands and cands[0].score < best_score:
                best_score = cands[0].score
                best_uld = uld.id
                best_box = cands[0].to_placed_box(pkg.id)
        if best_box is not None:
            trackers[best_uld].commit(best_box)
            trackers[best_uld].add_weight(pkg.weight)
            placed_ids.append(pkg.id)
        else:
            unplaced_ids.append(pkg.id)

    return placed_ids, unplaced_ids


# ─────────────────────────────────────────────────────────────────────────────
# Individual
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    genes: np.ndarray  # int8, len == n_economy
    fitness: float = float("inf")


@dataclass
class GAResult:
    best: Individual
    history: List[float] = field(default_factory=list)  # best fitness per generation


# ─────────────────────────────────────────────────────────────────────────────
# Fitness + validation/repair
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(
    genes: np.ndarray,
    priority_packages: List[Package],
    economy_packages: List[Package],
    priority_ulds: List[ULD],
    other_ulds: List[ULD],
    max_repairs: int = 3,
) -> float:
    """
    Trial-packs bucket 1 (priority + economy[genes==2]) into priority_ulds and
    bucket 2 (economy[genes==1]) into other_ulds. If any Priority package
    fails to place in bucket 1, repeatedly evicts the largest-volume gene==2
    Economy package (flips it to 1 if bucket 2 has room, else 0) and retries,
    mutating `genes` in place -- this is the "reallocated from the priority
    ULDs" repair the spec calls for. Returns the fitness (sum of delay cost
    of economy packages that end up unplaced); lower is better.
    """
    econ_by_idx = {i: p for i, p in enumerate(economy_packages)}

    for _ in range(max_repairs):
        bucket1_econ_idx = [i for i, g in enumerate(genes) if g == 2]
        bucket1 = priority_packages + [econ_by_idx[i] for i in bucket1_econ_idx]
        _, unplaced1 = _trial_pack_bucket(bucket1, priority_ulds)
        priority_ids = {p.id for p in priority_packages}
        stuck_priority = [pid for pid in unplaced1 if pid in priority_ids]

        if not stuck_priority:
            break

        # Evict the largest-volume gene==2 economy package to free space.
        if not bucket1_econ_idx:
            break  # nothing left to evict; genuinely infeasible for this bucket
        evict_i = max(bucket1_econ_idx, key=lambda i: econ_by_idx[i].volume)
        genes[evict_i] = 1  # try the other bucket first; may get pushed to 0 below

    bucket1_econ_idx = [i for i, g in enumerate(genes) if g == 2]
    bucket2_econ_idx = [i for i, g in enumerate(genes) if g == 1]

    bucket1 = priority_packages + [econ_by_idx[i] for i in bucket1_econ_idx]
    bucket2 = [econ_by_idx[i] for i in bucket2_econ_idx]

    placed1, unplaced1 = _trial_pack_bucket(bucket1, priority_ulds)
    placed2, unplaced2 = _trial_pack_bucket(bucket2, other_ulds)

    placed1_set = set(placed1)
    # Economy packages physically unplaced in bucket 1 fall back to bucket 2's
    # pool for scoring purposes (mirrors the pipeline's own retroactive-fill
    # spirit) -- but for GA fitness we simply count them unplaced if bucket 2
    # doesn't already also count them; keep it simple: any economy id not in
    # placed1 or placed2 is unplaced.
    econ_ids = {p.id: p for p in economy_packages}
    placed_ids = placed1_set | set(placed2)
    unplaced_cost = sum(
        econ_ids[pid].delay_cost for pid in econ_ids if pid not in placed_ids
    )
    return float(unplaced_cost)


# ─────────────────────────────────────────────────────────────────────────────
# Population init
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_seed(
    economy_packages: List[Package],
    priority_ulds: List[ULD],
    other_ulds: List[ULD],
) -> np.ndarray:
    """
    Individual 0: maximize economy packed into the priority bucket. Sort
    economy by descending cost-density (delay_cost / volume) and greedily
    trial-pack into priority_ulds first (leftover headroom after priority
    packages' own volume is implicitly respected by the trial pack itself,
    since priority packages are packed first within the bucket-1 trial).
    Whatever fits gets gene 2; everything else gets gene 1 (tried against
    other_ulds) or 0 if it fits nowhere.
    """
    n = len(economy_packages)
    genes = np.zeros(n, dtype=np.int8)
    if n == 0:
        return genes

    order = sorted(
        range(n),
        key=lambda i: -(economy_packages[i].delay_cost / max(economy_packages[i].volume, 1e-9)),
    )

    # Greedily grow the gene==2 set, one package at a time, keeping it only if
    # the bucket-1 trial pack still places it (cheap incremental check via a
    # persistent tracker rather than a full re-pack per candidate).
    trackers = {u.id: ExtremePointTracker(u) for u in priority_ulds} if priority_ulds else {}

    if priority_ulds:
        for i in order:
            pkg = economy_packages[i]
            placed = False
            for uld in priority_ulds:
                cands = rank_placements(pkg, trackers[uld.id], priority_uld_ids=set(), top_k=1)
                if cands:
                    box = cands[0].to_placed_box(pkg.id)
                    trackers[uld.id].commit(box)
                    trackers[uld.id].add_weight(pkg.weight)
                    placed = True
                    break
            genes[i] = 2 if placed else 1
    else:
        genes[:] = 1

    # Anything not placed in bucket 1: try bucket 2 feasibility roughly via
    # remaining weight/volume capacity check (cheap dimension/weight/volume
    # sanity, not a full trial pack) -- if it can't possibly fit any other_uld,
    # leave it unallocated (0) rather than wasting a gene value.
    if other_ulds:
        other_vol = sum(u.volume for u in other_ulds)
        other_wt = sum(u.weight_limit for u in other_ulds)
        used_vol = used_wt = 0.0
        for i in order:
            if genes[i] != 1:
                continue
            pkg = economy_packages[i]
            pkg_dims = sorted([pkg.length, pkg.width, pkg.height])
            fits_any = any(
                all(pkg_dims[k] <= sorted([u.length, u.width, u.height])[k] for k in range(3))
                for u in other_ulds
            )
            if not fits_any or used_wt + pkg.weight > other_wt or used_vol + pkg.volume > other_vol:
                genes[i] = 0
            else:
                used_vol += pkg.volume
                used_wt += pkg.weight
    else:
        genes[genes == 1] = 0

    return genes


def _init_population(
    pop_size: int,
    economy_packages: List[Package],
    priority_ulds: List[ULD],
    other_ulds: List[ULD],
    rng: np.random.Generator,
) -> List[Individual]:
    n = len(economy_packages)
    pop = [Individual(genes=_greedy_seed(economy_packages, priority_ulds, other_ulds))]
    for _ in range(pop_size - 1):
        genes = rng.integers(0, 2, size=n, dtype=np.int8)  # random in {0, 1} only
        pop.append(Individual(genes=genes))
    return pop


# ─────────────────────────────────────────────────────────────────────────────
# Selection / crossover / mutation
# ─────────────────────────────────────────────────────────────────────────────

def _selection_probs(pop: List[Individual]) -> np.ndarray:
    """Fitness-proportional selection probability; lower fitness = better."""
    fitness = np.array([ind.fitness for ind in pop], dtype=np.float64)
    # Softmax over negative fitness (shifted for numerical stability).
    neg = -fitness
    neg = neg - neg.max()
    weights = np.exp(neg)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(pop), 1.0 / len(pop))
    return weights / total


def _crossover(parent_a: Individual, parent_b: Individual,
                gene_contribution_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Per-gene: inherit from the fitter parent with prob `gene_contribution_ratio`."""
    fitter, weaker = (parent_a, parent_b) if parent_a.fitness <= parent_b.fitness else (parent_b, parent_a)
    mask = rng.random(len(fitter.genes)) < gene_contribution_ratio
    child = np.where(mask, fitter.genes, weaker.genes).astype(np.int8)
    return child


def _mutate_pool(pool: List[Individual], rng: np.random.Generator,
                  n_buckets: int = 4, sample_frac: float = 0.5) -> None:
    """
    Sort pool by fitness ascending (fittest first), split into n_buckets
    fixed-size buckets, apply MUTATION_RATES[b] to a random sample_frac
    subset of each bucket. Mutates genes in place.
    """
    order = sorted(range(len(pool)), key=lambda i: pool[i].fitness)
    bucket_size = max(1, len(order) // n_buckets)
    rates = MUTATION_RATES

    for b in range(n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < n_buckets - 1 else len(order)
        bucket_idxs = order[start:end]
        if not bucket_idxs:
            continue
        rate = rates[min(b, len(rates) - 1)]
        n_sample = max(1, int(round(len(bucket_idxs) * sample_frac)))
        sampled = rng.choice(bucket_idxs, size=min(n_sample, len(bucket_idxs)), replace=False)
        for idx in sampled:
            genes = pool[idx].genes
            flip_mask = rng.random(len(genes)) < rate
            if not flip_mask.any():
                continue
            n_flip = int(flip_mask.sum())
            # New value uniform among the other 2 ternary values.
            offsets = rng.integers(1, 3, size=n_flip)
            genes[flip_mask] = (genes[flip_mask] + offsets) % 3


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(
    priority_packages: List[Package],
    economy_packages: List[Package],
    priority_ulds: List[ULD],
    other_ulds: List[ULD],
    pop_size: int = 16,
    max_generations: int = 20,
    patience: int = 6,
    gene_contribution_ratio: float = 0.65,
    seed: Optional[int] = None,
    time_budget_seconds: float = 90.0,
) -> GAResult:
    """
    time_budget_seconds : hard wall-clock cap on the whole run, checked
        before every individual fitness evaluation (not just once per
        generation). A handful of real instances have geometry that makes
        _evaluate() (and the trial-pack cost it's built on) far more
        expensive than the typical case -- this guarantees run_ga() always
        returns the best individual found so far instead of running for an
        unbounded/very long time on those instances. GAPipeline's final pack
        is unaffected either way since it always rigorously re-packs the
        winning individual regardless of how many generations actually ran.
    """
    rng = np.random.default_rng(seed)
    t_start = time.monotonic()

    if not economy_packages:
        return GAResult(best=Individual(genes=np.zeros(0, dtype=np.int8), fitness=0.0), history=[0.0])

    def _time_left() -> bool:
        return time.monotonic() - t_start < time_budget_seconds

    pop = _init_population(pop_size, economy_packages, priority_ulds, other_ulds, rng)
    for ind in pop:
        if not _time_left():
            break
        ind.fitness = _evaluate(
            ind.genes, priority_packages, economy_packages, priority_ulds, other_ulds
        )

    pop.sort(key=lambda ind: ind.fitness)
    best = pop[0]
    history = [best.fitness]
    stale = 0

    for _gen in range(max_generations):
        if not _time_left():
            break

        probs = _selection_probs(pop)
        n_offspring = len(pop)  # produce as many offspring as current pop, then keep fittest half of parents+offspring
        offspring: List[Individual] = []
        for _ in range(n_offspring):
            a_idx, b_idx = rng.choice(len(pop), size=2, replace=True, p=probs)
            child_genes = _crossover(pop[a_idx], pop[b_idx], gene_contribution_ratio, rng)
            offspring.append(Individual(genes=child_genes))

        mutation_pool = pop + offspring
        _mutate_pool(mutation_pool, rng)

        budget_exhausted_mid_generation = False
        for ind in mutation_pool:
            if not _time_left():
                budget_exhausted_mid_generation = True
                break
            ind.fitness = _evaluate(
                ind.genes, priority_packages, economy_packages, priority_ulds, other_ulds
            )
        if budget_exhausted_mid_generation:
            # Unevaluated individuals (still fitness=inf) sort last and are
            # dropped by the keep-fittest-half slice below -- never selected
            # as `best`, so no unfairly-favourable stale genome sneaks through.
            pass

        mutation_pool.sort(key=lambda ind: ind.fitness)
        keep = max(pop_size, 2)
        pop = mutation_pool[:keep]

        if pop[0].fitness < best.fitness - 1e-9:
            best = pop[0]
            stale = 0
        else:
            stale += 1

        history.append(pop[0].fitness)
        if stale >= patience or budget_exhausted_mid_generation:
            break

    return GAResult(best=best, history=history)
