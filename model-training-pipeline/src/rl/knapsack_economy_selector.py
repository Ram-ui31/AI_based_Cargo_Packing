"""
knapsack_economy_selector.py -- replaces _assign_economy_by_value_density's
greedy (sort by delay_cost/volume, first-fit) selection with an actual
multi-knapsack ILP solve: which Economy packages should go in which ULD, to
maximize total delay_cost captured, subject to each ULD's real remaining
weight and volume capacity (after Priority placement).

Why: exhaustively confirmed (combined_packer.py, global_economy_packer.py)
that the PACKING layer (which of 3 different real-3D algorithms places a
given set of packages, plus cross-ULD rescue) is at its ceiling. Direct
volume/weight accounting also confirmed this instance is genuinely
overbooked -- roughly half of Economy's total volume physically cannot fit
in the fleet, no matter how good the packer is. So the only remaining lever
is WHICH Economy packages get selected to compete for space at all, not how
well they're arranged once selected -- a value-maximization (knapsack)
problem, not a geometry problem. The current greedy-by-value-density
heuristic is a reasonable approximation but not an actual optimizer.

This solves the NOMINAL (weight + volume, not real 3D shape) relaxation as
a proper 0/1 multi-knapsack ILP via scipy.optimize.milp (HiGHS solver, no
new dependency). The result is a candidate SELECTION, not a placement --
still needs to go through the real packer (combined_packer.py) afterward to
verify/realize it, same as every other selection this session, since
nominal feasibility doesn't guarantee real 3D feasibility (though the two
have been closely correlated in practice here).

Formulation:
    variables   : x[i,j] in {0,1} for each (Economy package i, ULD j)
    objective   : maximize sum(delay_cost[i] * x[i,j] for all i,j)
                  (equivalently: minimize the delay_cost LEFT OUT)
    constraints : sum_j x[i,j] <= 1            (each package placed at most once)
                  sum_i weight[i] * x[i,j] <= W[j]   (per-ULD weight budget)
                  sum_i volume[i] * x[i,j] <= V[j]   (per-ULD volume budget)

Does not modify train_rl.py or any existing file.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds


def solve_economy_knapsack(economy_df, uld_capacities, time_limit=120, volume_utilization_factor=0.72):
    """
    economy_df      : DataFrame with Package_ID, Length, Width, Height, Weight, Delay_Cost
    uld_capacities   : dict {uld_id: (remaining_weight, remaining_volume)} -- REAL
                       remaining capacity after Priority placement, not nominal
                       full ULD capacity.
    volume_utilization_factor : discount applied to the VOLUME budget only
        (not weight -- weight is exactly additive, no packing-loss analog).
        Without this, an earlier version of this function gave the ILP the
        raw nominal remaining volume as its budget and it over-selected:
        verified directly on the real 400-package instance, 199/297 Economy
        packages were selected as nominally fitting, but only 119 (60%) of
        those actually fit once real 3D packing was attempted -- 80 selected
        packages (40%) were pure waste, occupying "nominal slots" that could
        have gone to packages that WOULD have really fit. 0.72 approximates
        the ~65-85% real utilization range this session found repeatedly for
        this packer/instance family -- makes the ILP's budget conservative
        enough that its selections are much more likely to be physically
        realizable, at the cost of being more conservative than the true
        achievable ceiling on instances where real packing does better.
    Returns: dict {Package_ID: uld_id} for selected packages (omitted = not selected).
    """
    pkgs = economy_df.reset_index(drop=True)
    n_items = len(pkgs)
    uld_ids = list(uld_capacities.keys())
    n_ulds = len(uld_ids)
    if n_items == 0 or n_ulds == 0:
        return {}

    weight = pkgs['Weight'].to_numpy(dtype=float)
    volume = (pkgs['Length'] * pkgs['Width'] * pkgs['Height']).to_numpy(dtype=float)
    delay_cost = pkgs['Delay_Cost'].to_numpy(dtype=float)

    n_vars = n_items * n_ulds

    def var_idx(i, j):
        return i * n_ulds + j

    # Objective: minimize -sum(delay_cost[i] * x[i,j]) == maximize captured delay_cost.
    c = np.zeros(n_vars)
    for i in range(n_items):
        for j in range(n_ulds):
            c[var_idx(i, j)] = -delay_cost[i]

    rows_assign, cols_assign, data_assign = [], [], []
    for i in range(n_items):
        for j in range(n_ulds):
            rows_assign.append(i); cols_assign.append(var_idx(i, j)); data_assign.append(1.0)
    A_assign = sp.csr_matrix((data_assign, (rows_assign, cols_assign)), shape=(n_items, n_vars))
    assign_constraint = LinearConstraint(A_assign, -np.inf, np.ones(n_items))

    rows_w, cols_w, data_w = [], [], []
    rows_v, cols_v, data_v = [], [], []
    for j in range(n_ulds):
        for i in range(n_items):
            rows_w.append(j); cols_w.append(var_idx(i, j)); data_w.append(weight[i])
            rows_v.append(j); cols_v.append(var_idx(i, j)); data_v.append(volume[i])
    A_weight = sp.csr_matrix((data_w, (rows_w, cols_w)), shape=(n_ulds, n_vars))
    A_volume = sp.csr_matrix((data_v, (rows_v, cols_v)), shape=(n_ulds, n_vars))
    weight_ub = np.array([uld_capacities[uid][0] for uid in uld_ids])
    volume_ub = np.array([uld_capacities[uid][1] * volume_utilization_factor for uid in uld_ids])
    weight_constraint = LinearConstraint(A_weight, -np.inf, weight_ub)
    volume_constraint = LinearConstraint(A_volume, -np.inf, volume_ub)

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)

    print(f'[knapsack] solving: {n_items} Economy packages x {n_ulds} ULDs = {n_vars} binary variables...')
    result = milp(
        c, constraints=[assign_constraint, weight_constraint, volume_constraint],
        integrality=integrality, bounds=bounds,
        options={'time_limit': time_limit, 'disp': False},
    )
    if result.x is None:
        print(f'[knapsack] solver failed: {result.message}')
        return {}
    print(f'[knapsack] status: {result.message}  captured delay_cost: {-result.fun:,.0f}')

    x = result.x.reshape(n_items, n_ulds)
    assignment = {}
    for i in range(n_items):
        j = np.argmax(x[i])
        if x[i, j] > 0.5:
            assignment[pkgs.loc[i, 'Package_ID']] = uld_ids[j]
    return assignment
