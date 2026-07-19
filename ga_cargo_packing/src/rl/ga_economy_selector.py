"""
ga_economy_selector.py -- a genetic algorithm over the Economy
package-selection problem, operating on the FULL joint (package, ULD)
assignment simultaneously, instead of a single greedy sort order
(value_density_pow1.5) or an exact-but-slow ILP (knapsack_economy_selector.py,
exhaustively confirmed worse than the greedy on this instance across 5
separate calibration attempts).

FIRST ATTEMPT (superseded): a nominal-capacity fitness function (per-ULD
weight/volume budget, volume discounted by ~0.78 to approximate real
packing efficiency) was tried first and FAILED (30,475 -> 31,525, worse).
Root cause, confirmed by direct comparison: the production greedy
(value_density_pow1.5) checks candidates against UNDISCOUNTED nominal
capacity and deliberately relies on the real packer + cross-ULD rescue to
sort out true feasibility afterward -- so seeding the GA with that greedy
solution and then repairing it against a STRICTER discounted budget just
evicted a large chunk of a perfectly good seed before the GA ever got a
chance to improve on it. More fundamentally: ANY optimizer (exact ILP or
population search) that scores candidates by nominal weight/volume --
discounted or not -- is optimizing a proxy that cannot see which SPECIFIC
packages will actually survive real 3D geometry (fragmentation is
package-shape-and-arrangement-dependent, not a fixed fraction). No amount
of searching a nominal formulation harder fixes that information gap.

CURRENT APPROACH: fitness uses REAL geometry, not a nominal proxy, via a
fast (not the full production) single-pass packer: HeuristicPacker
(strategy='contact', max_pivots~20) with NO cross-ULD rescue and no
multi-strategy comparison, calibrated at ~0.15-0.2s per ULD-with-~50-items
call (see scripts/test_ga_economy_selection.py's timing calibration) --
several minutes for a full population x generations search, versus hours
for the same scale using the full CombinedPacker (3-4 strategies + 8
rescue rounds). Priority packages are pre-placed once (outside this
module, by the caller) into a per-ULD base Heightmap; each fitness
evaluation clones that base (cheap -- Heightmap is just a placements list
+ two scalars, see geometry.Heightmap) and packs the gene's assigned
Economy subset into the clone. Real hm.fits()/place() enforce weight AND
volume hard limits exactly, so no discount-factor guessing is needed at
all -- a package either really fits or it doesn't, precisely mirroring the
real system. Fitness = the ACTUAL delay_cost of packages that really got
placed, not a nominal estimate.

Encoding: genes[i] in {0, 1, ..., n_ulds} for each Economy package i --
0 means NONE (not attempted), 1..n_ulds means "attempt to place in that
ULD" (via the fast single-pass packer above; may still fail to fit and
therefore not count toward fitness, exactly like the real pipeline).

Seeding: the initial population includes the current best-known solution
(the greedy value_density_pow1.5 selection) as one individual. Combined
with elitism (the best individual across all generations is never lost),
the GA's own best-found fitness can only match or improve on the seed's
REAL (not nominal) fitness under this module's own accounting. Whether
that translates to a real improvement after the full CombinedPacker's
final validation (multi-strategy + cross-ULD rescue, which this fast
fitness function does NOT include) is exactly what the calling script
checks.
"""
from __future__ import annotations

import numpy as np


def _clone_hm(base_hm, geometry_module):
    new_hm = geometry_module.Heightmap(
        length=base_hm.length, width=base_hm.width,
        height=base_hm.height, weight_limit=base_hm.weight_limit,
    )
    new_hm.placements = list(base_hm.placements)
    new_hm.weight_used = base_hm.weight_used
    new_hm.volume_used = base_hm.volume_used
    return new_hm


def _real_fitness(genes, pkgs, uld_ids, base_hm_by_uld, fast_packer):
    """Real (not nominal) fitness: for each ULD, clone its Priority-only
    base Heightmap and greedily pack the gene-assigned Economy subset into
    it via fast_packer (a HeuristicPacker instance) -- returns total
    delay_cost of packages that ACTUALLY got placed."""
    geometry_module = fast_packer._geometry
    total_captured = 0.0
    for j, uid in enumerate(uld_ids, start=1):
        idx = np.flatnonzero(genes == j)
        if len(idx) == 0:
            continue
        base_hm = base_hm_by_uld[uid]
        if base_hm is None:
            # No Priority in this ULD -- need an empty Heightmap of the
            # right dims; caller guarantees base_hm_by_uld always has a
            # real Heightmap (empty if no Priority), never None, for ULDs
            # that exist. Defensive fallback: skip (shouldn't happen).
            continue
        hm = _clone_hm(base_hm, geometry_module)
        subset_df = pkgs.iloc[idx]
        fast_packer._greedy_pack_into(hm, subset_df)
        subset_pids = set(subset_df['Package_ID'])
        for p in hm.placements:
            if p.package_id in subset_pids:
                total_captured += pkgs.loc[pkgs['Package_ID'] == p.package_id, 'Delay_Cost'].iloc[0]
    return total_captured


def solve_economy_ga_real(economy_df, uld_ids, base_hm_by_uld, fast_packer,
                           population=30, generations=40, patience=15,
                           seed_assignment=None, seed=0, verbose=True):
    """
    economy_df       : DataFrame with Package_ID, Length, Width, Height, Weight, Delay_Cost
    uld_ids          : list of ULD_IDs (defines gene value 1..len(uld_ids))
    base_hm_by_uld   : dict {uld_id: Heightmap} -- Priority-only, already packed,
                       one real (possibly empty) Heightmap per ULD in uld_ids.
    fast_packer      : a HeuristicPacker instance (small max_pivots for speed)
                       used as the fitness function's real-but-fast packer.
    seed_assignment  : optional dict {Package_ID: uld_id}, e.g. the current best
                       greedy selection -- included as one initial-population
                       individual.
    Returns: dict {Package_ID: uld_id} for the best individual found (its
    GENE assignment, not necessarily what fast_packer's own quick pack
    accepted -- the caller re-validates through the real CombinedPacker).
    """
    rng = np.random.default_rng(seed)
    pkgs = economy_df.reset_index(drop=True)
    n_items = len(pkgs)
    n_ulds = len(uld_ids)
    uld_idx = {u: j + 1 for j, u in enumerate(uld_ids)}
    if n_items == 0 or n_ulds == 0:
        return {}

    delay_cost = pkgs['Delay_Cost'].to_numpy(dtype=float)
    volume = (pkgs['Length'] * pkgs['Width'] * pkgs['Height']).to_numpy(dtype=float)
    pid_to_i = {pid: i for i, pid in enumerate(pkgs['Package_ID'])}

    def _fitness(genes):
        return _real_fitness(genes, pkgs, uld_ids, base_hm_by_uld, fast_packer)

    # ── Initial population ──────────────────────────────────────────────
    pop = rng.integers(0, n_ulds + 1, size=(population, n_items), dtype=np.int32)
    if seed_assignment is not None:
        seed_genes = np.zeros(n_items, dtype=np.int32)
        for pid, uid in seed_assignment.items():
            if pid in pid_to_i and uid in uld_idx:
                seed_genes[pid_to_i[pid]] = uld_idx[uid]
        pop[0] = seed_genes
    pop[1] = 0  # safe all-NONE baseline
    order = np.argsort(-(delay_cost / np.clip(volume, 1, None)))
    rr_genes = np.zeros(n_items, dtype=np.int32)
    for rank, i in enumerate(order):
        rr_genes[i] = (rank % n_ulds) + 1
    pop[2] = rr_genes

    fitness = np.array([_fitness(pop[k]) for k in range(population)])
    best_genes = pop[np.argmax(fitness)].copy()
    best_fitness = fitness.max()
    no_improve = 0
    if verbose:
        print(f'[ga-real] gen -init-  best_fitness(real captured delay_cost)={best_fitness:,.0f}  '
              f'(seed alone: {fitness[0]:,.0f})')

    for gen in range(generations):
        n_elite = max(1, population // 10)
        elite_idx = np.argsort(-fitness)[:n_elite]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < population:
            def _tournament():
                cand = rng.integers(0, population, size=3)
                return cand[np.argmax(fitness[cand])]
            p1, p2 = _tournament(), _tournament()
            mask = rng.random(n_items) < 0.5
            child = np.where(mask, pop[p1], pop[p2])
            mut_mask = rng.random(n_items) < 0.03
            if mut_mask.any():
                child[mut_mask] = rng.integers(0, n_ulds + 1, size=mut_mask.sum())
            new_pop.append(child)

        pop = np.stack(new_pop[:population])
        fitness = np.array([_fitness(pop[k]) for k in range(population)])

        gen_best = fitness.max()
        if gen_best > best_fitness + 1e-6:
            best_fitness = gen_best
            best_genes = pop[np.argmax(fitness)].copy()
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(f'[ga-real] gen {gen:4d}  best_fitness={best_fitness:,.0f}  no_improve={no_improve}')

        if no_improve >= patience:
            if verbose:
                print(f'[ga-real] stopped at gen {gen} (patience {patience} exhausted)')
            break

    assignment = {}
    for i in range(n_items):
        if best_genes[i] != 0:
            assignment[pkgs.loc[i, 'Package_ID']] = uld_ids[best_genes[i] - 1]
    return assignment
