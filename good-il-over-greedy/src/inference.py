import torch

from .config import MAX_N_ULDS, MAX_SAFE_PKGS, MAX_SAFE_ULDS, DEVICE
from .data_utils import build_tensors, chunk_dataframe

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED INFERENCE — run the model on an instance of ANY size
#
# This is the function real usage should call instead of `model(...)`
# directly. It transparently handles instances that exceed the model's
# trained capacity by chunking (see src/data_utils.py for the chunking
# rationale) and stitching the partial results back into one global
# assignment, carrying running weight/volume usage forward between package
# chunks so a later chunk can't double-book a ULD that an earlier chunk
# already filled.
#
# For instances within the model's normal capacity, this is equivalent to a
# single direct forward pass (just with a bit of extra bookkeeping), so it's
# safe to use unconditionally — including for "normal" instances.
# ─────────────────────────────────────────────────────────────────────────────

def assign_uld_chunk(model, packages_df, ulds_df, device,
                      weight_used_init=None, volume_used_init=None):
    """
    Run one forward pass of the model on a single (packages_df, ulds_df)
    pair that already fits within MAX_N_PKGS / MAX_N_ULDS, applying weight
    and volume masking on top of the model's raw output (the supervised
    forward() pass does not hard-mask weight/volume — see
    src/losses.py — so we enforce it explicitly here at inference time).

    weight_used_init / volume_used_init: optional list[float] of capacity
    already consumed in each ULD before this chunk runs (used when chunking
    packages — see run_inference()).

    Returns:
        assignment   : {Package_ID: ULD_ID | 'NONE'}
        weight_used  : list[float], updated running weight per ULD
        volume_used  : list[float], updated running volume per ULD
    """
    n_ulds = len(ulds_df)
    n_pkgs = len(packages_df)
    uld_ids = ulds_df['ULD_ID'].tolist()
    uld_weight_limits = ulds_df['Weight_Limit'].tolist()
    uld_volumes = (ulds_df['Length'] * ulds_df['Width'] * ulds_df['Height']).tolist()

    weight_used = list(weight_used_init) if weight_used_init is not None else [0.0] * n_ulds
    volume_used = list(volume_used_init) if volume_used_init is not None else [0.0] * n_ulds

    tensors = build_tensors(packages_df, ulds_df, device)
    model.eval()
    with torch.no_grad():
        logits = model(
            tensors['uld_feats'], tensors['pkg_feats'], tensors['key_padding_mask'],
            torch.tensor([n_ulds], device=device),
            tensors['dim_mask'], tensors['priority_mask'], tensors['tightness'],
        ).squeeze(0)   # (MAX_N_PKGS, N_ULD_CLASSES)

    pkg_records = packages_df.to_dict('records')
    assignment = {}

    for i in range(n_pkgs):
        pkg = pkg_records[i]
        pid = pkg['Package_ID']
        pw_ = pkg['Weight']
        pvol = pkg['Length'] * pkg['Width'] * pkg['Height']
        is_priority = (str(pkg['Type']).upper() == 'PRIORITY')

        lg = logits[i].clone()

        # Enforce weight/volume capacity explicitly (not hard-masked inside
        # the model's forward() — see note above).
        for j in range(n_ulds):
            if weight_used[j] + pw_ > uld_weight_limits[j] + 1e-6:
                lg[j] = -1e9
            if volume_used[j] + pvol > uld_volumes[j] + 1e-6:
                lg[j] = -1e9

        all_blocked = all(lg[j].item() <= -1e8 for j in range(n_ulds))

        if is_priority and all_blocked:
            # Mirror GreedyLabeller's intent: a priority package should not
            # go to NONE if there's genuinely no room anywhere. Fall back to
            # the least-loaded ULD by remaining weight headroom, same as the
            # model's own sample_actions() fallback, and flag it loudly,
            # exactly like GreedyLabeller.label() does.
            action = max(range(n_ulds), key=lambda j: uld_weight_limits[j] - weight_used[j])
            print(f'  WARNING: Priority package {pid} forced into ULD '
                  f'{uld_ids[action]} despite exceeding capacity '
                  f'- no eligible ULD found')
        elif all_blocked:
            action = MAX_N_ULDS   # NONE
        else:
            action = int(torch.argmax(lg).item())

        if action == MAX_N_ULDS or action >= n_ulds:
            assignment[pid] = 'NONE'
        else:
            assignment[pid] = uld_ids[action]
            weight_used[action] += pw_
            volume_used[action] += pvol

    return assignment, weight_used, volume_used


def run_inference(model, packages_df, ulds_df, device=None,
                   max_pkgs=None, max_ulds=None):
    """
    Real-use-case-ready inference entry point: produces a correct
    {Package_ID: ULD_ID | 'NONE'} assignment for an instance of ANY size,
    chunking internally when the instance exceeds the model's trained
    capacity (max_pkgs / max_ulds, defaulting to MAX_SAFE_PKGS / MAX_SAFE_ULDS)
    instead of crashing or truncating.

    Strategy (see src/data_utils.py for rationale):
      - ULDs are split into groups of <= max_ulds.
      - For each ULD group, packages are split into chunks of <= max_pkgs.
      - Packages already assigned to a ULD in an earlier ULD group are
        excluded from later groups.
      - Running weight/volume usage per ULD carries over between package
        chunks within the same ULD group, so capacity is respected globally.
      - Any package still unassigned after every ULD group has had a chance
        gets 'NONE'.
    """
    device = device or DEVICE
    max_pkgs = MAX_SAFE_PKGS if max_pkgs is None else max_pkgs
    max_ulds = MAX_SAFE_ULDS if max_ulds is None else max_ulds

    full_assignment = {}
    remaining_pkgs_df = packages_df.reset_index(drop=True)
    uld_chunks = chunk_dataframe(ulds_df.reset_index(drop=True), max_ulds)

    for uld_chunk in uld_chunks:
        if len(remaining_pkgs_df) == 0:
            break

        n_ulds_here = len(uld_chunk)
        weight_used = [0.0] * n_ulds_here
        volume_used = [0.0] * n_ulds_here

        pkg_chunks = chunk_dataframe(remaining_pkgs_df, max_pkgs)

        for pkg_chunk in pkg_chunks:
            chunk_assignment, weight_used, volume_used = assign_uld_chunk(
                model, pkg_chunk, uld_chunk, device,
                weight_used_init=weight_used, volume_used_init=volume_used,
            )
            full_assignment.update(chunk_assignment)

        # Packages this ULD group couldn't place move on to the next group.
        still_unassigned_ids = {pid for pid, uid in full_assignment.items() if uid == 'NONE'}
        remaining_pkgs_df = remaining_pkgs_df[
            remaining_pkgs_df['Package_ID'].isin(still_unassigned_ids)
        ].reset_index(drop=True)

    return full_assignment
