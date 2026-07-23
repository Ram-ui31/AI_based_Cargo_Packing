import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    MAX_N_PKGS, N_ULD_CLASSES, IGNORE_INDEX,
    N_EPOCHS, BATCH_SIZE, LR, PATIENCE,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, ULD_FEAT_DIM, PKG_FEAT_DIM,
    LAMBDA_WEIGHT_PENALTY, LAMBDA_VOLUME_PENALTY, DEVICE,
)
from .data_utils import ClusteringDataset, collate_fn
from .labeller import DEFAULT_LABELLER
from .losses import capacity_violation_penalty


def train_il(
    model,
    train_dir,
    test_dir,
    train_meta_path,
    test_meta_path,
    labeller   = None,
    n_epochs   = N_EPOCHS,
    batch_size = BATCH_SIZE,
    lr         = LR,
    patience   = PATIENCE,
    save_path  = None,
    log_path   = None,
    device     = DEVICE,
    lambda_weight = LAMBDA_WEIGHT_PENALTY,
    lambda_volume = LAMBDA_VOLUME_PENALTY,
):
    labeller = labeller or DEFAULT_LABELLER

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/20)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print("Building datasets...")
    train_ds = ClusteringDataset(train_dir, train_meta_path, labeller, device)
    val_ds   = ClusteringDataset(test_dir,  test_meta_path,  labeller, device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn)
    print(f"Train: {len(train_ds)} instances | Val: {len(val_ds)} instances")

    history          = []
    best_val_loss    = float('inf')
    patience_counter = 0

    epoch_bar = tqdm(range(n_epochs), desc='IL Training', unit='epoch')

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_ce, train_wpen, train_vpen = 0.0, 0.0, 0.0, 0.0
        train_correct, train_total = 0, 0

        for batch in train_dl:
            uld_f  = batch['uld_feats'].to(device)
            pkg_f  = batch['pkg_feats'].to(device)
            kpm    = batch['key_padding_mask'].to(device)
            n_ulds = batch['n_ulds_batch'].to(device)
            dm     = batch['dim_mask'].to(device)
            pm     = batch['priority_mask'].to(device)
            tight  = batch['tightness'].to(device)
            labels = batch['labels'].to(device)   # (B, MAX_N_PKGS)

            logits = model(uld_f, pkg_f, kpm, n_ulds, dm, pm, tight)
            # logits: (B, MAX_N_PKGS, N_ULD_CLASSES) -> reshape for CrossEntropy
            B = logits.shape[0]
            ce_loss = criterion(
                logits.view(B * MAX_N_PKGS, N_ULD_CLASSES),
                labels.view(B * MAX_N_PKGS),
            )

            weight_pen, volume_pen = capacity_violation_penalty(
                logits, pkg_f, uld_f, n_ulds, labels
            )
            loss = ce_loss + lambda_weight * weight_pen + lambda_volume * volume_pen

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss  += loss.item()    * B
            train_ce    += ce_loss.item() * B
            train_wpen  += weight_pen.item() * B
            train_vpen  += volume_pen.item() * B

            # Flatten labels and masks for accuracy calculation
            labels_flat = labels.view(-1)
            mask_flat   = (labels_flat != IGNORE_INDEX)
            preds_flat  = logits.argmax(dim=-1).view(-1)

            train_correct += (preds_flat[mask_flat] == labels_flat[mask_flat]).sum().item()
            train_total   += mask_flat.sum().item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)
        avg_train_ce   = train_ce   / len(train_ds)
        avg_train_wpen = train_wpen / len(train_ds)
        avg_train_vpen = train_vpen / len(train_ds)
        train_acc      = train_correct / max(train_total, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_wpen, val_vpen = 0.0, 0.0
        priority_correct, priority_total = 0, 0

        with torch.no_grad():
            for batch in val_dl:
                uld_f  = batch['uld_feats'].to(device)
                pkg_f  = batch['pkg_feats'].to(device)
                kpm    = batch['key_padding_mask'].to(device)
                n_ulds = batch['n_ulds_batch'].to(device)
                dm     = batch['dim_mask'].to(device)
                pm     = batch['priority_mask'].to(device)
                tight  = batch['tightness'].to(device)
                labels = batch['labels'].to(device)

                logits = model(uld_f, pkg_f, kpm, n_ulds, dm, pm, tight)
                B      = logits.shape[0]
                ce_loss = criterion(
                    logits.view(B * MAX_N_PKGS, N_ULD_CLASSES),
                    labels.view(B * MAX_N_PKGS),
                )
                weight_pen, volume_pen = capacity_violation_penalty(
                    logits, pkg_f, uld_f, n_ulds, labels
                )
                loss = ce_loss + lambda_weight * weight_pen + lambda_volume * volume_pen

                val_loss += loss.item() * B
                val_wpen += weight_pen.item() * B
                val_vpen += volume_pen.item() * B

                # Flatten labels and masks for accuracy calculation
                labels_flat = labels.view(-1)
                mask_flat   = (labels_flat != IGNORE_INDEX)
                preds_flat  = logits.argmax(dim=-1).view(-1)

                val_correct += (preds_flat[mask_flat] == labels_flat[mask_flat]).sum().item()
                val_total   += mask_flat.sum().item()

                # Priority-specific accuracy
                # priority_mask from batch: True = Economy, so ~pm = Priority
                priority_mask_batch_flat = batch['priority_mask'].to(device).view(-1)
                prio_mask_flat = mask_flat & ~priority_mask_batch_flat
                priority_correct += (preds_flat[prio_mask_flat] == labels_flat[prio_mask_flat]).sum().item()
                priority_total   += prio_mask_flat.sum().item()

        avg_val_loss   = val_loss / len(val_ds)
        avg_val_wpen   = val_wpen / len(val_ds)
        avg_val_vpen   = val_vpen / len(val_ds)
        val_acc        = val_correct  / max(val_total, 1)
        prio_acc       = priority_correct / max(priority_total, 1)
        current_lr     = scheduler.get_last_lr()[0]

        row = {
            'epoch':           epoch + 1,
            'train_loss':      round(avg_train_loss, 6),
            'train_ce':        round(avg_train_ce,   6),
            'train_weight_pen':round(avg_train_wpen, 6),
            'train_volume_pen':round(avg_train_vpen, 6),
            'val_loss':        round(avg_val_loss,   6),
            'val_weight_pen':  round(avg_val_wpen,   6),
            'val_volume_pen':  round(avg_val_vpen,   6),
            'train_acc':       round(train_acc,       4),
            'val_acc':         round(val_acc,         4),
            'priority_acc':    round(prio_acc,        4),
            'lr':              round(current_lr,      8),
        }
        history.append(row)

        epoch_bar.set_postfix(
            tr_loss=f'{avg_train_loss:.4f}',
            val_loss=f'{avg_val_loss:.4f}',
            val_acc=f'{val_acc:.1%}',
            prio_acc=f'{prio_acc:.1%}',
            w_pen=f'{avg_val_wpen:.4f}',
            v_pen=f'{avg_val_vpen:.4f}',
            lr=f'{current_lr:.1e}',
        )

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss':         best_val_loss,
                'val_acc':          val_acc,
                'priority_acc':     prio_acc,
                'config': {
                    'd_model': D_MODEL, 'n_heads': N_HEADS,
                    'n_layers': N_LAYERS, 'd_ff': D_FF,
                    'uld_feat_dim': ULD_FEAT_DIM, 'pkg_feat_dim': PKG_FEAT_DIM,
                    'lambda_weight_penalty': lambda_weight,
                    'lambda_volume_penalty': lambda_volume,
                },
            }, save_path)
            epoch_bar.write(f'  Saved  val_loss={best_val_loss:.4f}  '
                            f'val_acc={val_acc:.1%}  prio_acc={prio_acc:.1%}  '
                            f'w_pen={avg_val_wpen:.4f}  v_pen={avg_val_vpen:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_bar.write(f'\n[Early Stop] No improvement for {patience} epochs.')
                break

    df = pd.DataFrame(history)
    if log_path:
        df.to_csv(log_path, index=False)
        print(f"Log saved -> {log_path}")
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    return df
