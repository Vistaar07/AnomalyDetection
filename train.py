import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from loss import BoundaryAwareOrdinalFocalLoss
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import kornia.augmentation as K
from evaluate import calculate_xview2_metrics
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    train_dataset = xBDDataset(config.PROCESSED_TRAIN_DIR, is_train=True)
    val_dataset = xBDDataset(config.PROCESSED_TEST_DIR, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareOrdinalFocalLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda')

    # Augmentations
    color_aug = K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5)

    spatial_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=30.0, p=0.3),
        data_keys=["input", "input", "mask", "mask"]
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{run_id}.csv"

    hp_header = (
        f"# RUN ID: {run_id}\n"
        f"# BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"# EPOCHS: {config.EPOCHS}\n"
        f"# LEARNING_RATE: {config.LEARNING_RATE}\n"
        f"# LAMBDA_FOCAL: {config.LAMBDA_FOCAL}\n"
        f"# LAMBDA_DICE: {config.LAMBDA_DICE}\n"
        f"# LAMBDA_ORDINAL: {config.LAMBDA_ORDINAL}\n"
        f"# LAMBDA_BOUNDARY: {config.LAMBDA_BOUNDARY}\n"
        f"# CUTMIX_PROB: {config.CUTMIX_PROB}\n"
    )

    start_epoch = 0
    best_score = -1.0
    history = []

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'latest_checkpoint_{run_id}.pth')
    best_model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{run_id}.pth')

    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', -1.0)

    WARMUP_EPOCHS = 4
    epochs_without_improvement = 0
    EARLY_STOPPING_PATIENCE = 15

    for epoch in range(start_epoch, config.EPOCHS):

        # Warmup LR
        if epoch < WARMUP_EPOCHS:
            warmup_lr = config.LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in pbar:
            pre, post = batch['pre'].to(device), batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask, edge = batch['mask'].to(device), batch['edge'].to(device)

            # Augmentations
            pre = color_aug(pre)
            post = color_aug(post)

            mask_k = mask.unsqueeze(1).float()
            edge_k = edge.unsqueeze(1).float()

            pre, post, mask_k, edge_k = spatial_aug(pre, post, mask_k, edge_k)

            mask = mask_k.squeeze(1).long()
            edge = edge_k.squeeze(1).float()

            # CutMix (keep configurable)
            if config.CUTMIX_PROB > 0 and np.random.rand() < config.CUTMIX_PROB:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(pre.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(pre.size(), lam)

                pre[:, :, bbx1:bbx2, bby1:bby2] = pre[rand_index, :, bbx1:bbx2, bby1:bby2]
                post[:, :, bbx1:bbx2, bby1:bby2] = post[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask[:, bbx1:bbx2, bby1:bby2] = mask[rand_index, bbx1:bbx2, bby1:bby2]
                edge[:, bbx1:bbx2, bby1:bby2] = edge[rand_index, bbx1:bbx2, bby1:bby2]

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                mask_out, edge_out = model(pre, post, g_pre, g_post)
                loss = criterion(mask_out, mask, edge_out, edge)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        valid_batches = 0
        global_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre, post = batch['pre'].to(device), batch['post'].to(device)
                g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
                mask, edge = batch['mask'].to(device), batch['edge'].to(device)

                with autocast('cuda', dtype=torch.bfloat16):
                    mask_out, edge_out = model(pre, post, g_pre, g_post)
                    loss = criterion(mask_out, mask, edge_out, edge)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item()
                valid_batches += 1

                preds = torch.argmax(mask_out, dim=1).cpu().numpy()
                true = mask.cpu().numpy()

                flattened_true = true.flatten()
                flattened_preds = preds.flatten()

                batch_cm = np.bincount(
                    config.NUM_CLASSES * flattened_true + flattened_preds,
                    minlength=config.NUM_CLASSES ** 2
                ).reshape(config.NUM_CLASSES, config.NUM_CLASSES)

                global_cm += batch_cm

        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')

        f1_loc, f1_dmg_classes, f1_dmg_harmonic, xview2_score = calculate_xview2_metrics(global_cm)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step(-xview2_score)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Localization F1: {f1_loc:.4f}")
        print(f"Minor Damage F1: {f1_dmg_classes[1]:.4f}")
        print(f"Damage F1 (Harmonic): {f1_dmg_harmonic:.4f}")
        print(f"xView2 Score: {xview2_score:.4f}")
        print(f"LR: {current_lr}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'xview2_score': xview2_score,
            'lr': current_lr
        })

        with open(log_filename, 'w') as f:
            f.write(hp_header)
        pd.DataFrame(history).to_csv(log_filename, mode='a', index=False)

        # ---------------- MODEL SAVING ----------------
        if xview2_score > best_score:
            best_score = xview2_score
            torch.save(model.state_dict(), best_model_path)
            print("New Best Model Saved (based on F1)!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered! No F1 improvement in {EARLY_STOPPING_PATIENCE} epochs.")
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score
        }, checkpoint_path)


if __name__ == '__main__':
    train()