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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    # Load Train and Validation (using test dir for now)
    train_dataset = xBDDataset(config.PROCESSED_TRAIN_DIR, is_train=True)
    val_dataset = xBDDataset(config.PROCESSED_TEST_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareOrdinalFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Added Scheduler: Reduces LR when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))

    for epoch in range(start_epoch, config.EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in pbar:
            pre, post = batch['pre'].to(device), batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask, edge = batch['mask'].to(device), batch['edge'].to(device)

            if config.CUTMIX_PROB > 0 and np.random.rand() < config.CUTMIX_PROB:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(pre.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(pre.size(), lam)
                pre[:, :, bbx1:bbx2, bby1:bby2] = pre[rand_index, :, bbx1:bbx2, bby1:bby2]
                post[:, :, bbx1:bbx2, bby1:bby2] = post[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask[:, bbx1:bbx2, bby1:bby2] = mask[rand_index, bbx1:bbx2, bby1:bby2]
                edge[:, bbx1:bbx2, bby1:bby2] = edge[rand_index, bbx1:bbx2, bby1:bby2]

            optimizer.zero_grad()
            with autocast():
                mask_out, edge_out = model(pre, post, g_pre, g_post)
                loss = criterion(mask_out, mask, edge_out, edge)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre, post = batch['pre'].to(device), batch['post'].to(device)
                g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
                mask, edge = batch['mask'].to(device), batch['edge'].to(device)

                with autocast():
                    mask_out, edge_out = model(pre, post, g_pre, g_post)
                    loss = criterion(mask_out, mask, edge_out, edge)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")

        # Save History to CSV
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        })
        pd.DataFrame(history).to_csv('training_log.csv', index=False)

        # Save Best Model based on Validation
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print("⭐ New Best Model Saved!")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_val_loss
        }, checkpoint_path)

if __name__ == '__main__':
    train()