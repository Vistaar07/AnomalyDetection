import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class BoundaryAwareOrdinalFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        smooth = 1e-6
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)
        # One-hot encode the target masks
        target_one_hot = F.one_hot(target, num_classes=config.NUM_CLASSES).permute(0, 3, 1, 2).float()

        # Calculate intersection and union
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, pred_masks, true_masks, pred_edges, true_edges):
        # 1. FOCAL LOSS (With Label Smoothing to prevent overconfidence)
        ce_loss = F.cross_entropy(pred_masks, true_masks, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        # 2. MULTI-CLASS DICE LOSS
        dice = self.dice_loss(pred_masks, true_masks)

        # 3. ORDINAL LOSS
        num_classes = pred_masks.shape[1]
        probs = torch.softmax(pred_masks, dim=1)
        class_ids = torch.arange(num_classes, device=pred_masks.device).view(1, -1, 1, 1)
        pred_expected = (probs * class_ids).sum(dim=1)

        true_classes = true_masks.float()
        mask_fg = true_classes > 0

        if mask_fg.sum() > 0:
            ordinal_loss = F.l1_loss(pred_expected[mask_fg], true_classes[mask_fg])
        else:
            ordinal_loss = torch.tensor(0.0, device=pred_masks.device)

        # 4. BOUNDARY LOSS
        true_edges = true_edges.unsqueeze(1).float()
        pred_edges = pred_edges.unsqueeze(1).float()
        edge_loss = self.bce(pred_edges, true_edges)

        # TOTAL COMBINED LOSS
        total_loss = (
                config.LAMBDA_FOCAL * focal_loss
                + config.LAMBDA_DICE * dice
                + config.LAMBDA_ORDINAL * ordinal_loss
                + config.LAMBDA_BOUNDARY * edge_loss
        )

        return total_loss