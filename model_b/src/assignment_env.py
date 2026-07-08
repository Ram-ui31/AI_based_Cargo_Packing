"""Per-instance assignment sampling + reward via the frozen Phase-A placement policy.

Sampling is a single structured decision per instance (not a multi-step MDP):
the network scores every package against every ULD slot + NONE in one forward
pass, then actions are drawn *sequentially* (priority packages first, then by
decreasing volume) with a running per-ULD weight pre-filter, mirroring how
the actual (harder) capacity constraint will bite. The real, authoritative
feasibility check is always the frozen placement policy run afterward -- the
pre-filter here only keeps obviously-overloaded picks out of the sample.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rl_packer", "src"))

import numpy as np
import torch

from assignment_policy import (
    MAX_N_ULDS, MAX_N_PKGS, N_CLASSES, NONE_CLASS,
    uld_features, package_features, compute_context, compute_masks,
)
from geometry import Heightmap
from placement_env import PlacementEnv


def build_inputs(instance):
    n_ulds = len(instance.ulds)
    n_pkgs = min(len(instance.packages), MAX_N_PKGS)
    uld_feats = uld_features(instance.ulds)
    pkg_feats = package_features(instance.packages)
    context = compute_context(instance.packages, instance.ulds, instance.K)
    masks = compute_masks(instance.packages, instance.ulds)

    key_padding_mask = np.zeros(MAX_N_ULDS + MAX_N_PKGS, dtype=bool)
    key_padding_mask[n_ulds:MAX_N_ULDS] = True
    key_padding_mask[MAX_N_ULDS + n_pkgs:] = True
    return uld_feats, pkg_feats, context, masks, key_padding_mask, n_ulds, n_pkgs


def rollout(policy, instance, greedy: bool = False, replay_actions: dict[int, int] | None = None):
    """Runs the assignment policy once over an instance.

    If `replay_actions` (a dict from package-row-position `i` to the chosen
    class `a`) is given, those exact actions are reused instead of
    sampling/argmax -- used by PPO to recompute log-probs of a fixed action
    sequence under updated parameters.

    Returns (assignment, total_logprob, mean_entropy, chosen_actions, aux) where
    aux carries the raw logits/masks needed for feasibility_hinge_loss().
    """
    uld_feats, pkg_feats, context, masks, kpm, n_ulds, n_pkgs = build_inputs(instance)

    logits = policy(
        torch.from_numpy(uld_feats).unsqueeze(0),
        torch.from_numpy(pkg_feats).unsqueeze(0),
        torch.from_numpy(context).unsqueeze(0),
        torch.from_numpy(kpm).unsqueeze(0),
    )[0]  # (MAX_N_PKGS, N_CLASSES)

    pkgs = instance.packages.iloc[:n_pkgs].reset_index(drop=True)
    volumes = (pkgs["Length"] * pkgs["Width"] * pkgs["Height"]).to_numpy()
    weights = pkgs["Weight"].to_numpy()
    is_priority = (pkgs["Type"] == "Priority").to_numpy()
    uld_weight_limit = instance.ulds["Weight_Limit"].to_numpy()[:n_ulds]
    uld_volume = (instance.ulds["Length"] * instance.ulds["Width"] * instance.ulds["Height"]).to_numpy()[:n_ulds]

    order = np.lexsort((-volumes, -is_priority.astype(int)))  # priority first, then by -volume

    running_weight = np.zeros(n_ulds)
    running_volume = np.zeros(n_ulds)
    assignment = {}
    chosen_actions: dict[int, int] = {}
    logprobs, entropies = [], []
    for i in order:
        row_logits = logits[i].clone()
        row_mask = torch.from_numpy(masks[i])
        row_logits = row_logits.masked_fill(~row_mask, -1e9)
        for j in range(n_ulds):
            if running_weight[j] + weights[i] > uld_weight_limit[j]:
                row_logits[j] = -1e9
            if running_volume[j] + volumes[i] > uld_volume[j]:
                row_logits[j] = -1e9
        if row_logits[:n_ulds].max().item() <= -1e8 and not is_priority[i]:
            # nothing feasible left for this economy package -- force NONE
            row_logits[:] = -1e9
            row_logits[NONE_CLASS] = 0.0

        dist = torch.distributions.Categorical(logits=row_logits)
        i_key = int(i)
        if replay_actions is not None:
            a = replay_actions[i_key]
        elif greedy:
            a = int(torch.argmax(row_logits).item())
        else:
            a = int(dist.sample().item())
        logprobs.append(dist.log_prob(torch.tensor(a)))
        entropies.append(dist.entropy())
        chosen_actions[i_key] = a

        pkg_id = pkgs.loc[i, "Package_ID"]
        if a == NONE_CLASS:
            assignment[pkg_id] = None
        else:
            assignment[pkg_id] = a
            running_weight[a] += weights[i]
            running_volume[a] += volumes[i]

    total_logprob = torch.stack(logprobs).sum()
    mean_entropy = torch.stack(entropies).mean()
    aux = dict(logits=logits, masks=masks, is_priority=is_priority, n_pkgs=n_pkgs, n_ulds=n_ulds)
    return assignment, total_logprob, mean_entropy, chosen_actions, aux


def sample_assignment(policy, instance, greedy: bool = False):
    """Convenience wrapper for evaluation: just the assignment dict."""
    with torch.no_grad():
        assignment, _, _, _, _ = rollout(policy, instance, greedy=greedy)
    return assignment


def feasibility_hinge_loss(aux: dict, margin: float = 1.0) -> torch.Tensor:
    """Penalizes the network for preferring NONE over a dimensionally-feasible
    ULD by more than `margin`, for economy packages -- computed from the
    static dim-fit mask (ground truth), not an assumed target. Found the
    network's own static argmax already rejects 30-46% of economy packages
    to NONE even when every one of them fits somewhere, which a single
    scalar cost/advantage over ~200+ joint decisions is too diffuse a
    signal to unlearn quickly on its own."""
    n_pkgs, n_ulds = aux["n_pkgs"], aux["n_ulds"]
    logits = aux["logits"][:n_pkgs]
    masks = torch.from_numpy(aux["masks"][:n_pkgs])
    is_priority = torch.from_numpy(aux["is_priority"][:n_pkgs])

    uld_mask = masks[:, :n_ulds]
    has_feasible = uld_mask.any(dim=1) & ~is_priority
    if not has_feasible.any():
        return torch.tensor(0.0)

    none_logit = logits[:, NONE_CLASS]
    uld_logits = logits[:, :n_ulds].masked_fill(~uld_mask, -1e9)
    max_feasible_logit = uld_logits.max(dim=1).values
    hinge = torch.relu(none_logit - max_feasible_logit + margin)
    return hinge[has_feasible].mean()


def soft_spread_loss(aux: dict) -> torch.Tensor:
    """Differentiable proxy for priority spread: expected number of distinct
    ULDs receiving at least one priority package, computed directly from the
    network's own softmax distribution over priority packages (independence
    approximation across packages) -- NOT from the sparse whole-rollout
    REINFORCE cost signal. Mirrors the mechanism the prior-art clusterer
    (A) used during its own training (an explicit differentiable
    spread_loss term, scaled by K), which our from-scratch clusterer (B)
    never had -- it only ever saw spread through the diffuse downstream
    cost, which turned out much weaker at teaching spread calibration.
    Caller scales this by (a function of) K before adding to the loss."""
    n_pkgs, n_ulds = aux["n_pkgs"], aux["n_ulds"]
    logits = aux["logits"][:n_pkgs]
    masks = torch.from_numpy(aux["masks"][:n_pkgs, :n_ulds])
    is_priority = torch.from_numpy(aux["is_priority"][:n_pkgs])

    prio_positions = is_priority.nonzero(as_tuple=True)[0]
    if len(prio_positions) == 0:
        return torch.tensor(0.0)

    prio_logits = logits[prio_positions][:, :n_ulds].masked_fill(~masks[prio_positions], -1e9)
    probs = torch.softmax(prio_logits, dim=-1)  # (n_prio, n_ulds)
    prob_uld_unused = torch.prod(1 - probs, dim=0)  # (n_ulds,) independence approx
    soft_spread = (1 - prob_uld_unused).sum()
    return soft_spread


def evaluate_assignment(instance, assignment: dict, placement_policy) -> dict:
    packages_df, ulds_df, K = instance.packages, instance.ulds, instance.K
    pkg_lookup = packages_df.set_index("Package_ID")
    uld_rows = list(ulds_df.itertuples())
    n_ulds = len(uld_rows)
    hms = [Heightmap(length=int(u.Length), width=int(u.Width), height=int(u.Height),
                      weight_limit=float(u.Weight_Limit)) for u in uld_rows]

    by_uld = [[] for _ in range(n_ulds)]
    left_behind = set()
    none_by_clusterer = set()
    for pid in packages_df["Package_ID"]:
        a = assignment.get(pid)
        if a is None or not (0 <= a < n_ulds):
            left_behind.add(pid)
            none_by_clusterer.add(pid)
        else:
            by_uld[a].append(pid)

    def run(hm, pkg_ids):
        if not pkg_ids:
            return [], []
        sub = packages_df[packages_df["Package_ID"].isin(pkg_ids)]
        env = PlacementEnv(None, sub, hm=hm)
        while not env.done:
            cands = env.candidates()
            if not cands:
                env.close_out()
                break
            idx, _, _ = placement_policy.select(cands, env.hm, env.pool, greedy=True)
            env.step(cands[idx])
        return env.placed_ids, env.left_behind_ids

    # Phase 1: priority packages first, per their assigned ULD (never sacrificed for economy)
    dropped_priority = []
    priority_placed_uld: dict[str, int] = {}
    for uld_idx in range(n_ulds):
        prio_ids = [p for p in by_uld[uld_idx] if pkg_lookup.loc[p, "Type"] == "Priority"]
        placed, dropped = run(hms[uld_idx], prio_ids)
        for pid in placed:
            priority_placed_uld[pid] = uld_idx
        dropped_priority.extend(dropped)

    # Fallback: retry any dropped priority package in whichever other ULD has the most free volume
    still_dropped_priority = []
    for pid in dropped_priority:
        order = sorted(range(n_ulds), key=lambda i: -(hms[i].volume - hms[i].volume_used))
        placed_ok = False
        for uld_idx in order:
            placed, _ = run(hms[uld_idx], [pid])
            if placed:
                priority_placed_uld[pid] = uld_idx
                placed_ok = True
                break
        if not placed_ok:
            still_dropped_priority.append(pid)

    # Phase 2: economy packages into their assigned (possibly already partly-filled) ULD
    dropped_by_packer = set()
    for uld_idx in range(n_ulds):
        econ_ids = [p for p in by_uld[uld_idx] if pkg_lookup.loc[p, "Type"] == "Economy"]
        _, dropped = run(hms[uld_idx], econ_ids)
        left_behind.update(dropped)
        dropped_by_packer.update(dropped)
    dropped_by_packer.update(still_dropped_priority)  # priority the packer couldn't fit anywhere

    delay_cost = float(pkg_lookup.loc[list(left_behind), "Delay_Cost"].sum()) if left_behind else 0.0
    spread = len(set(priority_placed_uld.values()))
    cost = K * spread + delay_cost

    return dict(
        cost=cost, spread=spread, delay_cost=delay_cost,
        left_behind=sorted(left_behind), priority_dropped=still_dropped_priority,
        n_priority=int((packages_df["Type"] == "Priority").sum()),
        utilization=[hm.utilization() for hm in hms],
        n_none_by_clusterer=len(none_by_clusterer),
        n_dropped_by_packer=len(dropped_by_packer),
        n_assigned_by_clusterer=len(packages_df) - len(none_by_clusterer),
    )
