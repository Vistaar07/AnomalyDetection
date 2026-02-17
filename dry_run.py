import torch
from torch.cuda.amp import autocast, GradScaler
from model import GLCrossNet
from loss import BoundaryAwareOrdinalFocalLoss
import config

def format_memory(bytes_val):
    return f"{bytes_val / (1024 ** 3):.2f} GB"

def run_memory_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        print("CUDA is not available. Cannot test VRAM usage.")
        return

    print(f"Testing on: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 1. Initialize Model, Loss, and Optimizer
    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    criterion = BoundaryAwareOrdinalFocalLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize the AMP GradScaler
    scaler = GradScaler()

    # 2. Create Dummy Data (Simulating one batch)
    # Shape: [Batch Size, Channels, Height, Width]
    batch_size = config.BATCH_SIZE
    c, h, w = 3, config.TILE_SIZE, config.TILE_SIZE

    print(f"\nSimulating Batch Size: {batch_size} with Tile Size: {h}x{w}")

    pre = torch.randn(batch_size, c, h, w, device=device)
    post = torch.randn(batch_size, c, h, w, device=device)
    g_pre = torch.randn(batch_size, c, h, w, device=device)
    g_post = torch.randn(batch_size, c, h, w, device=device)

    mask = torch.randint(0, config.NUM_CLASSES, (batch_size, h, w), device=device, dtype=torch.long)
    edge = torch.randint(0, 2, (batch_size, h, w), device=device, dtype=torch.float32)

    model.train()
    optimizer.zero_grad()

    # 3. Forward Pass with AMP
    with autocast():
        mask_out, edge_out = model(pre, post, g_pre, g_post)
        loss = criterion(mask_out, mask, edge_out, edge)

    # 4. Backward Pass with Scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 5. Report VRAM Usage
    allocated = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)

    print("\n--- VRAM Usage Report ---")
    print(f"Current Allocated: {format_memory(allocated)}")
    print(f"Peak VRAM Used:    {format_memory(peak)}")

    if peak < (8 * 1024**3):
        print("\n✅ SUCCESS: Peak memory is safely under 8.00 GB.")
    else:
        print("\n❌ WARNING: Peak memory exceeded 8.00 GB. You will OOM. Reduce BATCH_SIZE in config.py.")

if __name__ == '__main__':
    run_memory_test()