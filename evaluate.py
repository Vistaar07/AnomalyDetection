import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import xBDDataset
from model import GLCrossNet
from torch.amp import autocast

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

    f1_dmg_classes = np.array(f1_dmg_classes)
    f1_dmg_classes_safe = np.maximum(f1_dmg_classes, 1e-8)
    f1_dmg_harmonic = 4.0 / np.sum(1.0 / f1_dmg_classes_safe)

    score = 0.3 * f1_loc + 0.7 * f1_dmg_harmonic

    return f1_loc, f1_dmg_classes, f1_dmg_harmonic, score

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CHANGED: Now points to the specific test directory
    dataset = xBDDataset(config.PROCESSED_TEST_DIR, is_train=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GLCrossNet(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model_20260218_213512.pth'), map_location=device))
    model.eval()

    global_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

    print("Evaluating Model on Test Data...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pre, post = batch['pre'].to(device), batch['post'].to(device)
            g_pre, g_post = batch['g_pre'].to(device), batch['g_post'].to(device)
            mask = batch['mask'].cpu().numpy()

            with autocast('cuda'):
                mask_out, _ = model(pre, post, g_pre, g_post)

            preds = torch.argmax(mask_out, dim=1).cpu().numpy()

            flattened_true = mask.flatten()
            flattened_preds = preds.flatten()

            batch_cm = np.bincount(
                config.NUM_CLASSES * flattened_true + flattened_preds,
                minlength=config.NUM_CLASSES**2
            ).reshape(config.NUM_CLASSES, config.NUM_CLASSES)

            global_cm += batch_cm

    f1_loc, f1_dmg_classes, f1_dmg_harmonic, xview2_score = calculate_xview2_metrics(global_cm)

    print("\n--- Evaluation Results ---")
    print(f"Localization F1:        {f1_loc:.4f}")
    print(f"No Damage F1:           {f1_dmg_classes[0]:.4f}")
    print(f"Minor Damage F1:        {f1_dmg_classes[1]:.4f}")
    print(f"Major Damage F1:        {f1_dmg_classes[2]:.4f}")
    print(f"Destroyed F1:           {f1_dmg_classes[3]:.4f}")
    print(f"Damage F1 (Harmonic):   {f1_dmg_harmonic:.4f}")
    print(f"--------------------------")
    print(f"Overall xView2 Score:   {xview2_score:.4f}")

if __name__ == '__main__':
    evaluate()