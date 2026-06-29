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
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
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

    def _build_embeddings(self, uld_feats, pkg_feats, tightness):
        """tightness : (B, 1) float tensor, normalised to [0, 1]"""
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

        # tightness: (B,1) -> unsqueeze -> (B,1,1) -> Linear -> (B,1,d_model)
        # broadcast-add to full sequence (B, MAX_SEQ_LEN, d_model)
        tight_emb = self.tightness_proj(tightness.unsqueeze(1))   # (B, 1, d_model)
        seq = torch.cat([uld_emb, pkg_emb], dim=1)                # (B, MAX_SEQ_LEN, d_model)
        seq = seq + tight_emb                                      # broadcast add
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
                n_ulds_batch, dim_mask, priority_mask, tightness):
        seq     = self._build_embeddings(uld_feats, pkg_feats, tightness)
        out     = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        pkg_out = out[:, MAX_N_ULDS:, :]
        logits  = self.output_head(pkg_out)
        logits  = self._apply_hard_masks(logits, n_ulds_batch, dim_mask, priority_mask)
        return logits

    def sample_actions(self, uld_feats, pkg_feats, key_padding_mask,
                       n_ulds, dim_mask, priority_mask, tightness,
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
            dim_mask, priority_mask, tightness,
        ).squeeze(0)

        entropy     = -(F.softmax(logits_batch[:n_pkgs], dim=-1) *
                        F.log_softmax(logits_batch[:n_pkgs], dim=-1)).sum()
        actions     = torch.zeros(MAX_N_PKGS, dtype=torch.long, device=uld_feats.device)
        log_probs   = torch.zeros(MAX_N_PKGS, device=uld_feats.device)
        weight_used = [0.0] * n_ulds
        volume_used = [0.0] * n_ulds
        track_vol   = pkg_volumes is not None and uld_volumes is not None

        for i in range(n_pkgs):
            lg          = logits_batch[i].clone()
            is_priority = (lg[MAX_N_ULDS].item() < -1e8)

            for j in range(n_ulds):
                if weight_used[j] + pkg_weights[i] > uld_weight_limits[j]:
                    lg[j] = -1e9
            if track_vol:
                for j in range(n_ulds):
                    if volume_used[j] + pkg_volumes[i] > uld_volumes[j]:
                        lg[j] = -1e9

            # Priority fallback: if all ULDs blocked, force into whichever
            # ULD this package would overflow LEAST relative to its own capacity.
            if is_priority and all(lg[j].item() < -1e8 for j in range(n_ulds)):
                def _relative_overflow(j):
                    wt_over = max(0.0, (weight_used[j] + pkg_weights[i] - uld_weight_limits[j])
                                  / max(uld_weight_limits[j], 1e-6))
                    if track_vol:
                        vol_over = max(0.0, (volume_used[j] + pkg_volumes[i] - uld_volumes[j])
                                      / max(uld_volumes[j], 1e-6))
                        return max(wt_over, vol_over)
                    return wt_over
                fallback     = min(range(n_ulds), key=_relative_overflow)
                lg[fallback] = 0.0

            probs  = F.softmax(lg / temperature, dim=-1)
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

            actions[i]   = action
            log_probs[i] = F.log_softmax(lg / temperature, dim=-1)[action]
            if action < n_ulds:
                weight_used[action] += pkg_weights[i]
                if track_vol:
                    volume_used[action] += pkg_volumes[i]

        return actions, log_probs, entropy, logits_batch, weight_used, volume_used
