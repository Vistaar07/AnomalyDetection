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

        # label_smoothing has been removed.
        # When combined with focal loss, label_smoothing distorts the per-sample
        # probability estimate (pt = exp(-ce)) that the focal term depends on.
        # Smoothing artificially raises ce even for confident correct predictions,
        # which inflates (1-pt)^gamma and over-penalises easy examples — the
        # opposite of what focal loss is designed to do.
        # Use one or the other; here focal loss is the primary mechanism for
        # handling class imbalance, so label_smoothing is dropped.

    def dice_loss(self, pred, target):
        smooth     = 1.0
        pred       = F.softmax(pred, dim=1)
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

    def forward(self, pred_masks, true_masks, pred_edges, true_edges):
        pred_masks = pred_masks.float()
        pred_edges = pred_edges.float()

        # 1. FOCAL LOSS
        # Standard weighted cross-entropy (no label_smoothing) so that pt is a
        # true probability estimate and the focal modulation is mathematically correct.
        ce_loss    = F.cross_entropy(
            pred_masks,
            true_masks,
            weight=self.class_weights,
            reduction='none'
        )
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # 2. MULTI-CLASS WEIGHTED DICE LOSS
        dice = self.dice_loss(pred_masks, true_masks)

        # 3. BOUNDARY LOSS
        true_edges = true_edges.unsqueeze(1).float()
        edge_loss  = self.bce(pred_edges, true_edges)

        total_loss = (
                config.LAMBDA_FOCAL    * focal_loss
                + config.LAMBDA_DICE   * dice
                + config.LAMBDA_BOUNDARY * edge_loss
        )

        return total_loss