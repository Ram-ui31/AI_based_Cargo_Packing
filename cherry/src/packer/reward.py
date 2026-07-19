import torch
import torch.nn.functional as F

from .config import MAX_N_ULDS, MAX_N_PKGS, N_ULD_CLASSES


def compute_packing_cost(placements, packages_df, k_value):
    """
    Cost = Σ delay_cost(all unplaced packages) + K * spread
    where spread = number of ULDs that contain at least one Priority package.

    Unplaced packages include:
      - reason='clusterer_none'  : clusterer assigned to NONE
      - reason='packer_unfit'    : assigned to a ULD but packer couldn't place it

    Returns:
        total_cost         : float
        delay_cost         : float
        spread_cost        : float — K × n_priority_ulds
        n_priority_ulds    : int
        unplaced_priority  : list[str] — Package_IDs of unplaced Priority packages
        unplaced_economy   : list[str] — Package_IDs of unplaced Economy packages
    """
    pkg_lookup        = packages_df.set_index('Package_ID').to_dict('index')
    delay_cost        = 0
    prio_uld_ids      = set()
    unplaced_priority = []
    unplaced_economy  = []

    for p in placements:
        pid = p['Package_ID']
        uid = p['ULD_ID']
        pkg = pkg_lookup.get(pid, {})
        if uid == 'NONE':
            delay_cost += pkg.get('Delay_Cost', 0)
            if pkg.get('Type') == 'Priority':
                unplaced_priority.append(pid)
            else:
                unplaced_economy.append(pid)
        else:
            if pkg.get('Type') == 'Priority':
                prio_uld_ids.add(uid)

    n_p         = len(prio_uld_ids)
    spread_cost = k_value * n_p
    total       = delay_cost + spread_cost
    return total, delay_cost, spread_cost, n_p, unplaced_priority, unplaced_economy


def rl_capacity_violation_penalty(logits, n_pkgs, n_ulds,
                                   pkg_weights, uld_weight_limits,
                                   pkg_volumes=None, uld_volumes=None):
    """
    Differentiable capacity-overflow penalty for ONE instance, computed from
    raw (pre weight/volume-mask) logits.

    model.sample_actions() hard-masks weight/volume before sampling, so the
    policy never receives a gradient for wanting to overflow a ULD. This
    penalty is added on top of the REINFORCE loss so the network is pushed
    toward lower probability mass on overflowing ULDs, not just silently
    corrected by masking.

    logits            : (MAX_N_PKGS, N_ULD_CLASSES) raw model output
    n_pkgs, n_ulds    : ints — actual sizes (<= MAX_N_PKGS / MAX_N_ULDS)
    pkg_weights       : list[float], length n_pkgs
    uld_weight_limits : list[float], length n_ulds
    pkg_volumes       : list[float] or None
    uld_volumes       : list[float] or None

    Returns: (weight_penalty, volume_penalty) — scalar tensors averaged over
    the n_ulds real ULDs. volume_penalty is 0.0 if volumes aren't provided.
    """
    device = logits.device

    probs     = F.softmax(logits[:n_pkgs], dim=-1)         # (n_pkgs, N_ULD_CLASSES)
    uld_probs = probs[:, :n_ulds]                           # (n_pkgs, n_ulds)

    pkg_w  = torch.tensor(pkg_weights,      dtype=torch.float32, device=device)
    uld_wl = torch.tensor(uld_weight_limits, dtype=torch.float32, device=device).clamp(min=1e-6)

    expected_weight = torch.einsum('p,pu->u', pkg_w, uld_probs)
    weight_overflow = F.relu(expected_weight - uld_wl) / uld_wl
    weight_penalty  = weight_overflow.mean()

    if pkg_volumes is not None and uld_volumes is not None:
        pkg_v  = torch.tensor(pkg_volumes, dtype=torch.float32, device=device)
        uld_v  = torch.tensor(uld_volumes, dtype=torch.float32, device=device).clamp(min=1e-6)
        expected_volume = torch.einsum('p,pu->u', pkg_v, uld_probs)
        volume_overflow = F.relu(expected_volume - uld_v) / uld_v
        volume_penalty  = volume_overflow.mean()
    else:
        volume_penalty = torch.tensor(0.0, device=device)

    return weight_penalty, volume_penalty


# ── Auxiliary losses ported from cargoism/git/model_b(c)/src/assignment_env.py ─
# Both operate on ONE instance's raw (pre-mask) logits, mirroring how
# train_rl.py's rollout loop already processes instances one at a time.

def feasibility_hinge_loss(logits, dim_mask, priority_mask, n_pkgs, n_ulds, margin=1.0):
    """
    Penalizes the network for preferring NONE over a dimensionally-feasible
    ULD for an Economy package, by more than `margin` -- computed from the
    static dim-fit mask (ground truth), not the downstream cost. The sparse
    whole-rollout REINFORCE signal alone was found (in model_b) too diffuse
    to unlearn a policy that statically rejects a large fraction of
    dimensionally-feasible economy packages to NONE.

    logits        : (MAX_N_PKGS, N_ULD_CLASSES) raw model output for ONE instance
    dim_mask      : (MAX_N_PKGS, MAX_N_ULDS) bool -- True = package fits in ULD (dims only)
    priority_mask : (MAX_N_PKGS,) bool -- True = Economy (this codebase's convention:
                    priority_mask marks packages ALLOWED to go to NONE, i.e. Economy)
    """
    logits       = logits[:n_pkgs]
    uld_mask     = dim_mask[:n_pkgs, :n_ulds]
    is_economy   = priority_mask[:n_pkgs]
    has_feasible = uld_mask.any(dim=1) & is_economy
    if not has_feasible.any():
        return torch.tensor(0.0, device=logits.device)

    none_logit          = logits[:, MAX_N_ULDS]
    uld_logits           = logits[:, :n_ulds].masked_fill(~uld_mask, -1e9)
    max_feasible_logit   = uld_logits.max(dim=1).values
    hinge = torch.relu(none_logit - max_feasible_logit + margin)
    return hinge[has_feasible].mean()


def soft_spread_loss(logits, dim_mask, priority_mask, n_pkgs, n_ulds):
    """
    Differentiable proxy for priority spread: expected number of distinct
    ULDs receiving at least one priority package, computed directly from the
    policy's own softmax over priority packages (independence approximation
    across packages) -- NOT from the sparse whole-rollout REINFORCE cost
    signal. Caller scales this by (a function of) K before adding to the
    loss, giving spread a direct, dense gradient tied to K.

    priority_mask : (MAX_N_PKGS,) bool -- True = Economy (see note above); Priority
                    packages are ~priority_mask.
    """
    logits      = logits[:n_pkgs]
    is_priority = ~priority_mask[:n_pkgs]
    prio_positions = is_priority.nonzero(as_tuple=True)[0]
    if len(prio_positions) == 0:
        return torch.tensor(0.0, device=logits.device)

    uld_mask    = dim_mask[:n_pkgs, :n_ulds]
    prio_logits = logits[prio_positions][:, :n_ulds].masked_fill(~uld_mask[prio_positions], -1e9)
    probs       = torch.softmax(prio_logits, dim=-1)      # (n_prio, n_ulds)
    prob_uld_unused = torch.prod(1 - probs, dim=0)          # (n_ulds,) independence approx
    soft_spread = (1 - prob_uld_unused).sum()
    return soft_spread


def soft_spread_loss_ipr(logits, dim_mask, priority_mask, n_pkgs, n_ulds, eps=1e-6):
    """
    Alternative differentiable spread proxy using an inverse participation
    ratio (Herfindahl-index reciprocal) over the AGGREGATE expected priority
    mass per ULD, instead of soft_spread_loss's per-package independence
    approximation.

    Why: soft_spread_loss's `prod(1-p)` term decays to ~0 (saturating
    soft_spread at the n_ulds ceiling) EXPONENTIALLY in the number of
    priority packages -- verified on a real 54-priority/5-ULD instance,
    backprop through it gave a k_proj.weight gradient of ~2e-6, i.e. the
    K-conditioning path was structurally starved of signal no matter how
    training was run (multi-K data, contrastive loss, larger coefficients
    all failed for this reason, not for lack of training time). Aggregating
    into per-ULD mass first (m[u] = sum_i probs[i,u]) and taking the
    reciprocal Herfindahl index (1 / sum(q_u^2), q = m / n_priority) keeps
    the gradient magnitude roughly O(1/n_priority) instead of exponential --
    confirmed empirically: 20 gradient steps on this same instance took
    soft_spread_ipr from 5.0 (uniform) to ~1.0 (fully concentrated), versus
    soft_spread_loss moving essentially not at all over the same steps.

    Range is [1, n_ulds], same semantics as soft_spread_loss (1 = every
    priority package in a single ULD, n_ulds = spread maximally thin).
    """
    logits      = logits[:n_pkgs]
    is_priority = ~priority_mask[:n_pkgs]
    prio_positions = is_priority.nonzero(as_tuple=True)[0]
    n_prio = len(prio_positions)
    if n_prio == 0:
        return torch.tensor(0.0, device=logits.device)

    uld_mask    = dim_mask[:n_pkgs, :n_ulds]
    prio_logits = logits[prio_positions][:, :n_ulds].masked_fill(~uld_mask[prio_positions], -1e9)
    probs       = torch.softmax(prio_logits, dim=-1)      # (n_prio, n_ulds)
    mass        = probs.sum(dim=0)                          # (n_ulds,) expected count per ULD
    q           = mass / n_prio
    concentration = (q * q).sum()
    return 1.0 / (concentration + eps)
