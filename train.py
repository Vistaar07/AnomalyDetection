import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from loss import BoundaryAwareTailWeightedLoss
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from evaluate import calculate_xview2_metrics
from datetime import datetime


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    train_dataset = xBDDataset(config.PROCESSED_TRAIN_DIR, is_train=True)
    val_dataset   = xBDDataset(config.PROCESSED_TEST_DIR,  is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    model     = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareTailWeightedLoss().to(device)

    # Single optimizer group — no layerwise LR.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )

    # Cosine annealing over the main training phase (warmup to freeze epoch).
    # T_0=46 = FREEZE_EPOCH(50) - WARMUP_EPOCHS(4).
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=46, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler('cuda')

    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{run_id}.csv"

    hp_header = (
        f"# RUN ID: {run_id}\n"
        f"# BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"# EPOCHS: {config.EPOCHS}\n"
        f"# LEARNING_RATE: {config.LEARNING_RATE}\n"
        f"# CUTMIX_PROB: {config.CUTMIX_PROB}\n"
    )

    start_epoch = 0
    best_score  = -1.0
    history     = []

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'latest_checkpoint_{run_id}.pth')
    best_model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{run_id}.pth')

    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score  = checkpoint.get('best_score', -1.0)

    WARMUP_EPOCHS           = 4
    FREEZE_EPOCH            = 50
    EARLY_STOPPING_PATIENCE = 20
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.EPOCHS):

        # Soft LR reduction at epoch 50 for fine-tuning phase.
        if epoch == FREEZE_EPOCH:
            print("\n>>> LR REDUCTION: Transitioning to fine-tuning phase <<<")
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * 0.05

        # Linear warmup for first 4 epochs, then cosine takes over.
        if epoch < WARMUP_EPOCHS:
            warmup_lr = config.LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in pbar:
            pre,   post   = batch['pre'].to(device),   batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask,  edge   = batch['mask'].to(device),   batch['edge'].to(device)

            # CutMix: valid on normalized tensors since it only swaps spatial regions.
            if config.CUTMIX_PROB > 0 and np.random.rand() < config.CUTMIX_PROB:
                lam        = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(pre.size(0)).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(pre.size(), lam)

                pre[:,  :, bbx1:bbx2, bby1:bby2] = pre[rand_index,  :, bbx1:bbx2, bby1:bby2]
                post[:, :, bbx1:bbx2, bby1:bby2] = post[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask[:,    bbx1:bbx2, bby1:bby2] = mask[rand_index,    bbx1:bbx2, bby1:bby2]
                edge[:,    bbx1:bbx2, bby1:bby2] = edge[rand_index,    bbx1:bbx2, bby1:bby2]

            optimizer.zero_grad()

            with autocast('cuda'):
                mask_out, edge_out = model(pre, post, g_pre, g_post)
                loss = criterion(mask_out, mask, edge_out, edge)

            scaler.scale(loss).backward()

            # Unscale before clipping so gradients are in their true magnitude
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step() natively checks for NaNs. If found, it skips the step automatically.
            scaler.step(optimizer)

            # scaler.update() adjusts the scale factor properly based on if the step was skipped.
            scaler.update()

            # Safely add the loss for logging, skipping if it exploded this iteration
            if not torch.isnan(loss) and not torch.isinf(loss):
                train_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss      = 0.0
        valid_batches = 0
        global_cm     = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre,   post   = batch['pre'].to(device),   batch['post'].to(device)
                g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
                mask,  edge   = batch['mask'].to(device),   batch['edge'].to(device)

                with autocast('cuda'):
                    mask_out, edge_out = model(pre, post, g_pre, g_post)
                    loss = criterion(mask_out, mask, edge_out, edge)

                if torch.isnan(loss):
                    continue

                val_loss      += loss.item()
                valid_batches += 1

                preds   = torch.argmax(mask_out, dim=1)
                preds   = torch.clamp(preds, 0, config.NUM_CLASSES - 1).view(-1)
                targets = mask.view(-1)

                k = (targets >= 0) & (targets < config.NUM_CLASSES)
                cm = torch.bincount(
                    config.NUM_CLASSES * targets[k] + preds[k],
                    minlength=config.NUM_CLASSES ** 2
                ).reshape(config.NUM_CLASSES, config.NUM_CLASSES)

                global_cm += cm.cpu().numpy()

        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')

        f1_loc, f1_dmg_classes, f1_dmg_harmonic, xview2_score = calculate_xview2_metrics(global_cm)

        # Cosine scheduler steps every epoch between warmup and freeze.
        if epoch >= WARMUP_EPOCHS and epoch < FREEZE_EPOCH:
            scheduler.step(epoch - WARMUP_EPOCHS)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Localization F1:      {f1_loc:.4f}")
        print(f"No Damage F1:         {f1_dmg_classes[0]:.4f}")
        print(f"Minor Damage F1:      {f1_dmg_classes[1]:.4f}")
        print(f"Major Damage F1:      {f1_dmg_classes[2]:.4f}")
        print(f"Destroyed F1:         {f1_dmg_classes[3]:.4f}")
        print(f"Damage F1 (Harmonic): {f1_dmg_harmonic:.4f}")
        print(f"xView2 Score:         {xview2_score:.4f}")
        print(f"LR:                   {current_lr}")

        history.append({
            'epoch':        epoch + 1,
            'train_loss':   avg_train_loss,
            'val_loss':     avg_val_loss,
            'f1_loc':       f1_loc,
            'f1_no_damage': f1_dmg_classes[0],
            'f1_minor':     f1_dmg_classes[1],
            'f1_major':     f1_dmg_classes[2],
            'f1_destroyed': f1_dmg_classes[3],
            'f1_harmonic':  f1_dmg_harmonic,
            'xview2_score': xview2_score,
            'lr':           current_lr
        })

        with open(log_filename, 'w') as f:
            f.write(hp_header)
        pd.DataFrame(history).to_csv(log_filename, mode='a', index=False)

        # ---------------- MODEL SAVING ----------------
        if xview2_score > best_score:
            best_score = xview2_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New best model saved! Score: {best_score:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement} epoch(s). Best: {best_score:.4f}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score':           best_score
        }, checkpoint_path)


if __name__ == '__main__':
    train()