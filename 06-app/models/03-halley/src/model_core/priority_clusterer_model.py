import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
    ULD_FEAT_DIM, PKG_FEAT_DIM,
    MAX_N_ULDS, MAX_N_PKGS, N_ULD_CLASSES,
)


class TransformerClusterer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model

        # ── Base projection layers (shapes must never change) ─────────────────
        self.uld_proj = nn.Linear(ULD_FEAT_DIM, d_model)   # 7 -> 128
        self.pkg_proj = nn.Linear(PKG_FEAT_DIM, d_model)   # 9 -> 128

        # ── Positional and type embeddings ────────────────────────────────────
        self.uld_pos_embed = nn.Embedding(MAX_N_ULDS, d_model)
        self.pkg_pos_embed = nn.Embedding(MAX_N_PKGS, d_model)
        self.type_embed    = nn.Embedding(2, d_model)   # 0=ULD, 1=package

        # ── Tightness injection ───────────────────────────────────────────────
        # Projects scalar tightness -> d_model; broadcast-added to every token.
        self.tightness_proj = nn.Linear(1, d_model, bias=False)

        # ── K injection (dual path) ────────────────────────────────────────────
        # K (the spread-cost coefficient, cost = K*n_priority_ulds + delay_cost)
        # was previously invisible to the model entirely -- it only appeared in
        # the RL reward, never as a model input. A first attempt injected K only
        # via this one broadcast projection (mirroring tightness) and it was not
        # enough: the policy converged to a roughly fixed spread regardless of
        # K, because the signal gets diluted by the time it passes through the
        # full transformer trunk to reach the classification head. Fixed by
        # ALSO concatenating the raw (log-normalized) K value directly onto the
        # per-package encoding right before output_head (see forward()) --
        # a short, low-noise path straight into the final decision, alongside
        # this diffuse one that still lets K modulate the shared representation.
        self.k_proj = nn.Linear(1, d_model, bias=False)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Output head ───────────────────────────────────────────────────────
        # Input is d_model + 1: the transformer's per-package encoding plus
        # the raw K scalar concatenated directly (see k_proj's docstring for
        # why K needs this second, direct path in addition to the broadcast one).
        self.output_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_ULD_CLASSES),
        )
        self.dropout_layer = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _build_embeddings(self, uld_feats, pkg_feats, tightness, k_feat):
        """tightness, k_feat : (B, 1) float tensors, normalised to [0, 1]"""
        B      = uld_feats.shape[0]
        device = uld_feats.device

        uld_emb = self.uld_proj(uld_feats)   # (B, MAX_N_ULDS, d_model)
        pkg_emb = self.pkg_proj(pkg_feats)   # (B, MAX_N_PKGS, d_model)

        uld_pos = torch.arange(MAX_N_ULDS, device=device).unsqueeze(0)
        pkg_pos = torch.arange(MAX_N_PKGS, device=device).unsqueeze(0)
        uld_emb = uld_emb + self.uld_pos_embed(uld_pos)
        pkg_emb = pkg_emb + self.pkg_pos_embed(pkg_pos)

        uld_types = torch.zeros(B, MAX_N_ULDS, dtype=torch.long, device=device)
        pkg_types = torch.ones( B, MAX_N_PKGS, dtype=torch.long, device=device)
        uld_emb   = uld_emb + self.type_embed(uld_types)
        pkg_emb   = pkg_emb + self.type_embed(pkg_types)

        # tightness/k_feat: (B,1) -> unsqueeze -> (B,1,1) -> Linear -> (B,1,d_model)
        # broadcast-add to full sequence (B, MAX_SEQ_LEN, d_model)
        tight_emb = self.tightness_proj(tightness.unsqueeze(1))   # (B, 1, d_model)
        k_emb     = self.k_proj(k_feat.unsqueeze(1))               # (B, 1, d_model)
        seq = torch.cat([uld_emb, pkg_emb], dim=1)                # (B, MAX_SEQ_LEN, d_model)
        seq = seq + tight_emb + k_emb                              # broadcast add
        seq = self.dropout_layer(seq)
        return seq

    def _apply_hard_masks(self, logits, n_ulds_batch, dim_mask, priority_mask):
        """
        1. Range mask  — ULD slots >= n_ulds are invalid
        2. Dim mask    — package doesn't physically fit in this ULD
        3. Priority    — Priority packages cannot go to NONE (index MAX_N_ULDS)
        """
        NEG_INF   = -1e9
        B         = logits.shape[0]
        uld_range = torch.arange(MAX_N_ULDS, device=logits.device).unsqueeze(0).expand(B, -1)
        range_mask = (uld_range >= n_ulds_batch.unsqueeze(1)).unsqueeze(1).expand(-1, MAX_N_PKGS, -1)
        logits[:, :, :MAX_N_ULDS] = logits[:, :, :MAX_N_ULDS].masked_fill(range_mask, NEG_INF)
        logits[:, :, :MAX_N_ULDS] = logits[:, :, :MAX_N_ULDS].masked_fill(~dim_mask,  NEG_INF)
        logits[:, :, MAX_N_ULDS]  = logits[:, :, MAX_N_ULDS ].masked_fill(~priority_mask, NEG_INF)
        return logits

    def forward(self, uld_feats, pkg_feats, key_padding_mask,
                n_ulds_batch, dim_mask, priority_mask, tightness, k_feat):
        seq     = self._build_embeddings(uld_feats, pkg_feats, tightness, k_feat)
        out     = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        pkg_out = out[:, MAX_N_ULDS:, :]                                    # (B, MAX_N_PKGS, d_model)
        k_raw   = k_feat.unsqueeze(1).expand(-1, MAX_N_PKGS, -1)            # (B, MAX_N_PKGS, 1)
        logits  = self.output_head(torch.cat([pkg_out, k_raw], dim=-1))
        logits  = self._apply_hard_masks(logits, n_ulds_batch, dim_mask, priority_mask)
        return logits

    def sample_actions(self, uld_feats, pkg_feats, key_padding_mask,
                       n_ulds, dim_mask, priority_mask, tightness, k_feat,
                       n_pkgs, pkg_weights, uld_weight_limits,
                       temperature=1.0, pkg_volumes=None, uld_volumes=None):
        """
        Stochastic forward pass for REINFORCE.

        Returns:
            actions    : (MAX_N_PKGS,) long
            log_probs  : (MAX_N_PKGS,) float (differentiable)
            entropy    : scalar
            logits_full: (MAX_N_PKGS, N_ULD_CLASSES)
            weight_used: list[float] per-ULD weight after assignment
            volume_used: list[float] per-ULD volume after assignment
        """
        logits_batch = self.forward(
            uld_feats, pkg_feats, key_padding_mask,
            torch.tensor([n_ulds], device=uld_feats.device),
            dim_mask, priority_mask, tightness, k_feat,
        ).squeeze(0)

        entropy     = -(F.softmax(logits_batch[:n_pkgs], dim=-1) *
                        F.log_softmax(logits_batch[:n_pkgs], dim=-1)).sum()
        actions     = torch.zeros(MAX_N_PKGS, dtype=torch.long, device=uld_feats.device)
        log_probs   = torch.zeros(MAX_N_PKGS, device=uld_feats.device)
        weight_used = [0.0] * n_ulds
        volume_used = [0.0] * n_ulds
        track_vol   = pkg_volumes is not None and uld_volumes is not None

        # This loop is sequential by construction (each package's masking
        # depends on the running weight_used/volume_used from every package
        # before it), so it can't be vectorized across packages -- but the
        # masking condition itself (weight_used[j] + pkg_weights[i] > limit)
        # is plain Python arithmetic, needing no tensor read at all. The only
        # reason this loop used to touch the GPU per package was reading
        # logit *values* via .item() to decide "is this ULD masked" / to
        # sample an action -- on MPS, each .item() call forces a full
        # GPU command-buffer sync, and with up to MAX_N_PKGS packages this
        # was the dominant cost of an RL training epoch. Fix: pull the whole
        # (n_pkgs, N_ULD_CLASSES) logits slice to CPU ONCE up front (a single
        # sync), do all per-package decision-making (masking, sampling) on
        # that CPU copy, and only touch the original on-device, grad-tracked
        # logits_batch once per package for the final differentiable
        # log_probs[i] computation itself -- semantically identical to the
        # original per-.item() version, just without the per-package sync.
        logits_cpu = logits_batch[:n_pkgs].detach().to('cpu')

        for i in range(n_pkgs):
            lg_cpu      = logits_cpu[i].clone()
            is_priority = (lg_cpu[MAX_N_ULDS].item() < -1e8)

            for j in range(n_ulds):
                if weight_used[j] + pkg_weights[i] > uld_weight_limits[j]:
                    lg_cpu[j] = -1e9
            if track_vol:
                for j in range(n_ulds):
                    if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                        lg_cpu[j] = -1e9

            # Priority fallback: if all ULDs blocked, force into whichever
            # ULD this package would overflow LEAST relative to its own capacity.
            override_idx = None
            if is_priority and all(lg_cpu[j].item() < -1e8 for j in range(n_ulds)):
                def _relative_overflow(j):
                    wt_over = max(0.0, (weight_used[j] + pkg_weights[i] - uld_weight_limits[j])
                                  / max(uld_weight_limits[j], 1e-6))
                    if track_vol:
                        vol_over = max(0.0, (volume_used[j] + pkg_volumes[i] - uld_volumes[j])
                                      / max(uld_volumes[j], 1e-6))
                        return max(wt_over, vol_over)
                    return wt_over
                override_idx = min(range(n_ulds), key=_relative_overflow)
                lg_cpu[override_idx] = 0.0

            probs  = F.softmax(lg_cpu / temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()

            # Final safety: priority must not go to NONE.
            if is_priority and action == MAX_N_ULDS:
                def _relative_overflow2(j):
                    wt_over = max(0.0, (weight_used[j] + pkg_weights[i] - uld_weight_limits[j])
                                  / max(uld_weight_limits[j], 1e-6))
                    if track_vol:
                        vol_over = max(0.0, (volume_used[j] + pkg_volumes[i] - uld_volumes[j])
                                      / max(uld_volumes[j], 1e-6))
                        return max(wt_over, vol_over)
                    return wt_over
                action = min(range(n_ulds), key=_relative_overflow2)

            actions[i] = action

            # Rebuild the same mask on the ORIGINAL on-device tensor (pure
            # Python booleans already known above -- no device read needed)
            # so log_probs stays differentiable through the real logits_batch.
            lg = logits_batch[i].clone()
            for j in range(n_ulds):
                if weight_used[j] + pkg_weights[i] > uld_weight_limits[j]:
                    lg[j] = -1e9
            if track_vol:
                for j in range(n_ulds):
                    if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                        lg[j] = -1e9
            if override_idx is not None:
                lg[override_idx] = 0.0
            log_probs[i] = F.log_softmax(lg / temperature, dim=-1)[action]

            if action < n_ulds:
                weight_used[action] += pkg_weights[i]
                if track_vol:
                    volume_used[action] += pkg_volumes[i]

        return actions, log_probs, entropy, logits_batch, weight_used, volume_used
