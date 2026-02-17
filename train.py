import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from loss import BoundaryAwareOrdinalFocalLoss
import numpy as np
from torch.amp import autocast, GradScaler

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

    # CHANGED: Now points to the specific train directory
    dataset = xBDDataset(config.PROCESSED_TRAIN_DIR, is_train=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareOrdinalFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resumed from Epoch {start_epoch}")

    model.train()

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for batch in progress_bar:
            pre, post = batch['pre'].to(device), batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask, edge = batch['mask'].to(device), batch['edge'].to(device)

            r = np.random.rand(1)
            if config.CUTMIX_PROB > 0 and r < config.CUTMIX_PROB:
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

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

if __name__ == '__main__':
    train()