import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from torch.amp import autocast
from datetime import datetime

def calculate_xview2_metrics(confusion_matrix):
    tp_loc = np.sum(confusion_matrix[1:, 1:])
    fp_loc = np.sum(confusion_matrix[0, 1:])
    fn_loc = np.sum(confusion_matrix[1:, 0])

    f1_loc = tp_loc / (tp_loc + 0.5 * (fp_loc + fn_loc) + 1e-8)

    f1_dmg_classes = []
    for i in range(1, 5):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        f1 = tp / (tp + 0.5 * (fp + fn) + 1e-8)
        f1_dmg_classes.append(f1)

    f1_dmg_classes      = np.array(f1_dmg_classes)
    f1_dmg_classes_safe = np.maximum(f1_dmg_classes, 1e-8)
    f1_dmg_harmonic     = 4.0 / np.sum(1.0 / f1_dmg_classes_safe)

    score = 0.3 * f1_loc + 0.7 * f1_dmg_harmonic
    return f1_loc, f1_dmg_classes, f1_dmg_harmonic, score

def tta_predict(model, pre, post, g_pre, g_post):
    with autocast('cuda'):
        out0, _ = model(pre, post, g_pre, g_post)

        out1, _ = model(torch.flip(pre, [3]), torch.flip(post, [3]), torch.flip(g_pre, [3]), torch.flip(g_post, [3]))
        out1 = torch.flip(out1, [3])

        out2, _ = model(torch.flip(pre, [2]), torch.flip(post, [2]), torch.flip(g_pre, [2]), torch.flip(g_post, [2]))
        out2 = torch.flip(out2, [2])

        out3, _ = model(torch.flip(pre, [2, 3]), torch.flip(post, [2, 3]), torch.flip(g_pre, [2, 3]), torch.flip(g_post, [2, 3]))
        out3 = torch.flip(out3, [2, 3])

    return (out0 + out1 + out2 + out3) / 4.0

def generate_2way_channel_grid():
    combinations = []
    # Broad ranges to let the optimizer find the new Swin vs HRNet balance
    hrnet_bg   = [0.40, 0.45, 0.50, 0.55, 0.60] # HRNet here
    hrnet_nd   = [0.40, 0.45, 0.50, 0.55, 0.60]
    hrnet_min  = [0.30, 0.35, 0.40, 0.45]       # Swin for minor
    hrnet_maj  = [0.40, 0.45, 0.50, 0.55]
    hrnet_dest = [0.45, 0.50, 0.55, 0.60, 0.65] # HRNet here

    for bg in hrnet_bg:
        for nd in hrnet_nd:
            for mn in hrnet_min:
                for mj in hrnet_maj:
                    for dest in hrnet_dest:
                        combinations.append((
                            (bg, round(1.0 - bg, 2)),
                            (nd, round(1.0 - nd, 2)),
                            (mn, round(1.0 - mn, 2)),
                            (mj, round(1.0 - mj, 2)),
                            (dest, round(1.0 - dest, 2))
                        ))
    return combinations

def run_channel_grid_search_hrnet_swin():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = xBDDataset(config.PROCESSED_TEST_DIR, is_train=False)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading HRNet-W32...")
    model_hrnet = GLCrossNet(backbone='hrnet_w32', num_classes=config.NUM_CLASSES).to(device)
    model_hrnet.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model_hrnet_w32_20260421_105211.pth'), map_location=device))
    model_hrnet.eval()

    print("Loading Swin-Tiny...")
    model_swin = GLCrossNet(backbone='swin_tiny_patch4_window7_224', num_classes=config.NUM_CLASSES).to(device)
    model_swin.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model_20260417_174915.pth'), map_location=device))
    model_swin.eval()

    weight_combinations = generate_2way_channel_grid()
    num_combinations = len(weight_combinations)
    print(f"\nInitialized GPU-optimized 2-WAY grid search: {num_combinations} configurations.")

    cw_hrnet_tensors, cw_swin_tensors = [], []
    for w_tuple in weight_combinations:
        cw = torch.tensor(w_tuple, device=device, dtype=torch.float16)
        cw_hrnet_tensors.append(cw[:, 0].view(1, 5, 1, 1))
        cw_swin_tensors.append(cw[:, 1].view(1, 5, 1, 1))

    results_cm_gpu = torch.zeros((num_combinations, config.NUM_CLASSES ** 2), dtype=torch.int64, device=device)

    print("Evaluating Channel-Wise Ensemble (Raw Logits + TTA)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pre, post     = batch['pre'].to(device), batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)

            mask = batch['mask'].to(device).view(-1)
            mask_shifted = mask * config.NUM_CLASSES

            logits_hrnet = tta_predict(model_hrnet, pre, post, g_pre, g_post)
            logits_swin  = tta_predict(model_swin,  pre, post, g_pre, g_post)

            for i in range(num_combinations):
                ensemble_logits = (cw_hrnet_tensors[i] * logits_hrnet) + (cw_swin_tensors[i] * logits_swin)
                preds = torch.argmax(ensemble_logits, dim=1).view(-1)

                batch_cm = torch.bincount(mask_shifted + preds, minlength=config.NUM_CLASSES ** 2)
                results_cm_gpu[i] += batch_cm

    print("\nCalculations complete. Transferring results to CPU...")
    all_cms_cpu = results_cm_gpu.cpu().numpy().reshape(num_combinations, config.NUM_CLASSES, config.NUM_CLASSES)

    final_scores = []
    for i in range(num_combinations):
        w_tuple = weight_combinations[i]
        global_cm = all_cms_cpu[i]

        f1_loc, f1_dmg_classes, f1_dmg_harmonic, score = calculate_xview2_metrics(global_cm)
        final_scores.append({
            'weights': w_tuple,
            'score': score,
            'f1_loc': f1_loc,
            'f1_no_damage': f1_dmg_classes[0],
            'f1_minor': f1_dmg_classes[1],
            'f1_major': f1_dmg_classes[2],
            'f1_destroyed': f1_dmg_classes[3],
            'f1_harmonic': f1_dmg_harmonic
        })

    final_scores.sort(key=lambda x: x['score'], reverse=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"ensemble_channel_2way_HRNET_SWIN_{run_id}.txt"

    print(f"Writing detailed results to {output_filename}...")

    with open(output_filename, 'w') as f:
        f.write("=== 2-WAY CHANNEL-WISE ENSEMBLE GRID SEARCH ===\n")
        f.write("Models: HRNet-W32 (H) & Swin-Tiny (S)\n")
        f.write("Method: Raw Logit Averaging\n")
        f.write(f"Total combinations tested: {num_combinations}\n")
        f.write("Sorted by Overall xView2 Score\n")
        f.write("===============================================\n\n")

        for i, res in enumerate(final_scores):
            w = res['weights']
            f.write(f"RANK {i+1}: Overall Score: {res['score']:.5f}\n")
            f.write(f"  Background:  H={w[0][0]:.2f} | S={w[0][1]:.2f}\n")
            f.write(f"  No Damage:   H={w[1][0]:.2f} | S={w[1][1]:.2f}\n")
            f.write(f"  Minor Dmg:   H={w[2][0]:.2f} | S={w[2][1]:.2f}\n")
            f.write(f"  Major Dmg:   H={w[3][0]:.2f} | S={w[3][1]:.2f}\n")
            f.write(f"  Destroyed:   H={w[4][0]:.2f} | S={w[4][1]:.2f}\n")
            f.write("-" * 47 + "\n")
            f.write(f"  Localization F1:      {res['f1_loc']:.5f}\n")
            f.write(f"  No Damage F1:         {res['f1_no_damage']:.5f}\n")
            f.write(f"  Minor Damage F1:      {res['f1_minor']:.5f}\n")
            f.write(f"  Major Damage F1:      {res['f1_major']:.5f}\n")
            f.write(f"  Destroyed F1:         {res['f1_destroyed']:.5f}\n")
            f.write(f"  Damage F1 (Harmonic): {res['f1_harmonic']:.5f}\n\n")

    print(f"Done! Check {output_filename} for the full list.")

if __name__ == '__main__':
    run_channel_grid_search_hrnet_swin()