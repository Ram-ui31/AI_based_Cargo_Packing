"""Bridge to the already-proven TransformerClusterer (package -> ULD assignment)
from Desktop/test/uld-rl-finetuning, reusing its RL-finetuned checkpoint
(Desktop/clustering_v2/transformer_rl_v2_K.pt) instead of our from-scratch
Phase B. Assignment (this module) is combined with OUR frozen Phase A
placement policy for the actual geometry, via evaluate_assignment().
"""

from __future__ import annotations

import os
import sys

import torch

OLD_REPO = os.path.expanduser("~/Desktop/test/uld-rl-finetuning")
if OLD_REPO not in sys.path:
    sys.path.insert(0, OLD_REPO)

from src.model import TransformerClusterer  # noqa: E402
from src.data_utils import build_tensors, needs_chunking, chunk_dataframe  # noqa: E402
from src.config import MAX_N_ULDS, MAX_SAFE_PKGS, MAX_SAFE_ULDS  # noqa: E402

CHECKPOINT_PATH = os.path.expanduser("~/Desktop/clustering_v2/transformer_rl_v2_K.pt")


def load_old_clusterer(checkpoint_path: str = CHECKPOINT_PATH, device: str = "cpu") -> TransformerClusterer:
    model = TransformerClusterer()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _greedy_decode_chunk(model: TransformerClusterer, packages_df, ulds_df, device: str = "cpu") -> dict:
    """Deterministic (argmax) decode for one chunk that fits the model's fixed
    MAX_N_PKGS/MAX_N_ULDS shapes. Mirrors src.data_utils.il_sample_assignment's
    sequential weight/volume capacity masking, but always argmax, not sampled."""
    tensors = build_tensors(packages_df, ulds_df, device)
    n_ulds = len(ulds_df)
    n_pkgs = len(packages_df)

    with torch.no_grad():
        logits_batch = model.forward(
            tensors["uld_feats"], tensors["pkg_feats"], tensors["key_padding_mask"],
            torch.tensor([n_ulds], device=device),
            tensors["dim_mask"], tensors["priority_mask"], tensors["tightness"],
        ).squeeze(0)  # (MAX_N_PKGS, N_ULD_CLASSES)

    pkg_ids = packages_df["Package_ID"].tolist()
    uld_wt_limits = ulds_df["Weight_Limit"].tolist()
    pkg_weights = packages_df["Weight"].tolist()
    uld_volumes = (ulds_df["Length"] * ulds_df["Width"] * ulds_df["Height"]).tolist()
    pkg_volumes = (packages_df["Length"] * packages_df["Width"] * packages_df["Height"]).tolist()

    weight_used = [0.0] * n_ulds
    volume_used = [0.0] * n_ulds
    assignment: dict[str, int | None] = {}

    for i in range(n_pkgs):
        lg = logits_batch[i].clone()
        is_priority = lg[MAX_N_ULDS].item() < -1e8

        for j in range(n_ulds):
            if weight_used[j] + pkg_weights[i] > uld_wt_limits[j]:
                lg[j] = -1e9
            if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                lg[j] = -1e9

        def _relative_overflow(j, i=i):
            wt_over = max(0.0, (weight_used[j] + pkg_weights[i] - uld_wt_limits[j]) / max(uld_wt_limits[j], 1e-6))
            vol_over = max(0.0, (volume_used[j] + pkg_volumes[i] - uld_volumes[j]) / max(uld_volumes[j], 1e-6))
            return max(wt_over, vol_over)

        if is_priority and all(lg[j].item() < -1e8 for j in range(n_ulds)):
            fallback = min(range(n_ulds), key=_relative_overflow)
            lg[fallback] = 0.0

        action = int(torch.argmax(lg).item())
        if is_priority and action == MAX_N_ULDS:
            action = min(range(n_ulds), key=_relative_overflow)

        assignment[pkg_ids[i]] = action if action < n_ulds else None
        if action < n_ulds:
            weight_used[action] += pkg_weights[i]
            volume_used[action] += pkg_volumes[i]

    return assignment


def greedy_decode(model: TransformerClusterer, packages_df, ulds_df, device: str = "cpu") -> dict:
    """Chunk-safe deterministic decode -> {Package_ID: uld_idx (0-based, into
    ulds_df row order) | None}. Handles instances bigger than the model's
    fixed MAX_N_PKGS/MAX_N_ULDS by chunking, same pattern as the original
    il_sample_assignment_safe."""
    if not needs_chunking(packages_df, ulds_df, MAX_SAFE_PKGS, MAX_SAFE_ULDS):
        return _greedy_decode_chunk(model, packages_df, ulds_df, device)

    assignment: dict[str, int | None] = {}
    remaining = packages_df.reset_index(drop=True)
    uld_chunks = chunk_dataframe(ulds_df.reset_index(drop=True), MAX_SAFE_ULDS)

    uld_offset = 0
    for uld_chunk in uld_chunks:
        if len(remaining) == 0:
            break
        for pkg_chunk in chunk_dataframe(remaining, MAX_SAFE_PKGS):
            local = _greedy_decode_chunk(model, pkg_chunk, uld_chunk, device)
            for pid, local_idx in local.items():
                assignment[pid] = (uld_offset + local_idx) if local_idx is not None else None
        uld_offset += len(uld_chunk)
        resolved_now = {pid for pid in remaining["Package_ID"] if assignment.get(pid) is not None}
        remaining = remaining[~remaining["Package_ID"].isin(resolved_now)].reset_index(drop=True)

    for pid in remaining["Package_ID"]:
        assignment.setdefault(pid, None)
    return assignment
