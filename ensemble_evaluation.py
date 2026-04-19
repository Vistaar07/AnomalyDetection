import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from torch.amp import autocast

# --- ENSEMBLE WEIGHTS ---
CONV_WEIGHT = 0.30
SWIN_WEIGHT = 0.70

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

    f1_dmg_classes     = np.array(f1_dmg_classes)
    f1_dmg_classes_safe = np.maximum(f1_dmg_classes, 1e-8)
    f1_dmg_harmonic    = 4.0 / np.sum(1.0 / f1_dmg_classes_safe)

    score = 0.3 * f1_loc + 0.7 * f1_dmg_harmonic
    return f1_loc, f1_dmg_classes, f1_dmg_harmonic, score


def tta_predict(model, pre, post, g_pre, g_post):
    """
    Test-Time Augmentation: average predictions over 4 flip variants.
    """
    with autocast('cuda'):
        # Original
        out0, _ = model(pre, post, g_pre, g_post)

        # Horizontal flip
        out1, _ = model(
            torch.flip(pre,  [3]),
            torch.flip(post, [3]),
            torch.flip(g_pre,  [3]),
            torch.flip(g_post, [3])
        )
        out1 = torch.flip(out1, [3])

        # Vertical flip
        out2, _ = model(
            torch.flip(pre,  [2]),
            torch.flip(post, [2]),
            torch.flip(g_pre,  [2]),
            torch.flip(g_post, [2])
        )
        out2 = torch.flip(out2, [2])

        # Both flips
        out3, _ = model(
            torch.flip(pre,  [2, 3]),
            torch.flip(post, [2, 3]),
            torch.flip(g_pre,  [2, 3]),
            torch.flip(g_post, [2, 3])
        )
        out3 = torch.flip(out3, [2, 3])

    # Average logits then argmax - more stable than averaging softmax
    avg_out = (out0 + out1 + out2 + out3) / 4.0
    return avg_out


def evaluate_ensemble():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = xBDDataset(config.PROCESSED_TEST_DIR, is_train=False)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    # 1. INITIALIZE & LOAD CONVNEXT
    print("Loading ConvNeXt-Tiny Model...")
    model_conv = GLCrossNet(backbone='convnext_tiny', num_classes=config.NUM_CLASSES).to(device)
    conv_ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'best_model_20260415_131313.pth')
    model_conv.load_state_dict(torch.load(conv_ckpt_path, map_location=device))
    model_conv.eval()

    # 2. INITIALIZE & LOAD SWIN
    print("Loading Swin-Tiny Model...")
    model_swin = GLCrossNet(backbone='swin_tiny_patch4_window7_224', num_classes=config.NUM_CLASSES).to(device)
    swin_ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'best_model_20260417_174915.pth')
    model_swin.load_state_dict(torch.load(swin_ckpt_path, map_location=device))
    model_swin.eval()

    global_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

    print("Evaluating Ensemble on Test Data (with TTA)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pre,   post   = batch['pre'].to(device),   batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask          = batch['mask'].cpu().numpy()

            # Get TTA logits from both models
            logits_conv = tta_predict(model_conv, pre, post, g_pre, g_post)
            logits_swin = tta_predict(model_swin, pre, post, g_pre, g_post)

            # Weighted Logit Ensemble
            ensemble_logits = (CONV_WEIGHT * logits_conv) + (SWIN_WEIGHT * logits_swin)

            # Extract final predictions
            preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy()

            flattened_true  = mask.flatten()
            flattened_preds = preds.flatten()

            batch_cm = np.bincount(
                config.NUM_CLASSES * flattened_true + flattened_preds,
                minlength=config.NUM_CLASSES ** 2
            ).reshape(config.NUM_CLASSES, config.NUM_CLASSES)

            global_cm += batch_cm

    f1_loc, f1_dmg_classes, f1_dmg_harmonic, xview2_score = calculate_xview2_metrics(global_cm)

    print("\n--- Ensemble Evaluation Results (ConvNeXt + Swin) ---")
    print(f"Ensemble Weights:       Conv={CONV_WEIGHT:.2f} | Swin={SWIN_WEIGHT:.2f}")
    print(f"Localization F1:        {f1_loc:.4f}")
    print(f"No Damage F1:           {f1_dmg_classes[0]:.4f}")
    print(f"Minor Damage F1:        {f1_dmg_classes[1]:.4f}")
    print(f"Major Damage F1:        {f1_dmg_classes[2]:.4f}")
    print(f"Destroyed F1:           {f1_dmg_classes[3]:.4f}")
    print(f"Damage F1 (Harmonic):   {f1_dmg_harmonic:.4f}")
    print(f"-------------------------------------------------------")
    print(f"Overall xView2 Score:   {xview2_score:.4f}")


if __name__ == '__main__':
    evaluate_ensemble()