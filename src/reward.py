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
