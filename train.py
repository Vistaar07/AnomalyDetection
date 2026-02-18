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
import datetime

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

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareOrdinalFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler('cuda')

    # --- NEW LOGGING SETUP ---
    # Create a unique Run ID based on the current date and time
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{run_id}.csv"

    # Format the Hyperparameters as commented lines for the CSV header
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
    print(f"Logging this run to: {log_filename}")

    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'latest_checkpoint_{run_id}.pth')
    best_model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_{run_id}.pth')

    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))

    for epoch in range(start_epoch, config.EPOCHS):

        # --- LINEAR WARMUP (First 2 Epochs) ---
        if epoch < 2 and start_epoch == 0:
            warmup_lr = 1e-5 + (config.LEARNING_RATE - 1e-5) * (epoch / 2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

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
            with autocast('cuda'):
                mask_out, edge_out = model(pre, post, g_pre, g_post)
                loss = criterion(mask_out, mask, edge_out, edge)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- SAFE VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            # Disabled autocast here to prevent FP16 NaN overflows during evaluation
            for batch in tqdm(val_loader, desc="Validating"):
                pre, post = batch['pre'].to(device), batch['post'].to(device)
                g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
                mask, edge = batch['mask'].to(device), batch['edge'].to(device)

                mask_out, edge_out = model(pre, post, g_pre, g_post)
                loss = criterion(mask_out, mask, edge_out, edge)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item()
                valid_batches += 1

        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')

        # Step the scheduler only after warmup is complete
        if epoch >= 2:
            scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        })
        # --- NEW SAVING LOGIC ---
        # 1. Write the HP header first (overwrites file)
        with open(log_filename, 'w') as f:
            f.write(hp_header)
        # 2. Append the DataFrame underneath the header
        pd.DataFrame(history).to_csv(log_filename, mode='a', index=False)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Update to use the new unique best_model_path
            torch.save(model.state_dict(), best_model_path)
            print("⭐ New Best Model Saved!")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_val_loss
        }, checkpoint_path)

if __name__ == '__main__':
    train()