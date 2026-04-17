import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class BoundaryAwareTailWeightedLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss()
        self.register_buffer('class_weights', torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32))

        # Ordinal EMD loss reintroduced in a stable CDF form.
        # EMD penalises predictions proportional to ordinal distance from ground truth, pushing the model to at least get severity direction right.

    def dice_loss(self, pred, target):
        smooth = 1.0
        pred   = F.softmax(pred, dim=1)
        target_one_hot = (
            F.one_hot(target, num_classes=config.NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .float()
        )

        intersection   = (pred * target_one_hot).sum(dim=(2, 3))
        union          = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice           = (2. * intersection + smooth) / (union + smooth)

        dice_per_class = dice.mean(dim=0)
        weights        = self.class_weights / self.class_weights.sum()

        return 1 - (dice_per_class * weights).sum()

    def ordinal_emd_loss(self, pred, target):
        # Force float32 — AMP runs this in float16 by default, and softmax
        # outputs near zero in float16 cause cumsum to produce NaN/inf gradients
        # which explode the model. float32 cast here is the fix.
        pred = pred.float()

        probs = F.softmax(pred, dim=1)
        # Clamp to prevent log-of-zero in any downstream ops and stabilise cumsum
        probs = torch.clamp(probs, min=1e-6, max=1.0)
        C = probs.shape[1]

        target_one_hot = (
            F.one_hot(target, num_classes=C)
            .permute(0, 3, 1, 2)
            .float()
        )

        pred_cdf   = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_one_hot, dim=1)

        return torch.mean(torch.abs(pred_cdf - target_cdf))

    def forward(self, pred_masks, true_masks, pred_edges, true_edges):
        pred_masks = pred_masks.float()
        pred_edges = pred_edges.float()

        # 1. FOCAL LOSS
        # Get raw cross-entropy for accurate pt extraction
        ce_loss_raw = F.cross_entropy(
            pred_masks,
            true_masks,
            reduction='none'
        )
        pt = torch.exp(-ce_loss_raw).clamp(min=1e-6, max=1.0)

        # Get weighted cross-entropy to apply the focal scale against
        ce_loss_weighted = F.cross_entropy(
            pred_masks,
            true_masks,
            weight=self.class_weights,
            reduction='none'
        )

        # Combine them
        focal_loss = ((1 - pt) ** self.gamma * ce_loss_weighted).mean()

        # 2. MULTI-CLASS WEIGHTED DICE LOSS
        dice = self.dice_loss(pred_masks, true_masks)

        # 3. BOUNDARY LOSS
        true_edges = true_edges.unsqueeze(1).float()
        edge_loss  = self.bce(pred_edges, true_edges)

        # 4. ORDINAL EMD LOSS (weight=0.5, lightweight guide not a dominant term)
        emd = self.ordinal_emd_loss(pred_masks, true_masks)

        total_loss = (
                config.LAMBDA_FOCAL      * focal_loss
                + config.LAMBDA_DICE     * dice
                + config.LAMBDA_BOUNDARY * edge_loss
                + 0.5                    * emd
        )

        return total_loss