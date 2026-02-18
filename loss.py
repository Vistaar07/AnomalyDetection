import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class BoundaryAwareOrdinalFocalLoss(nn.Module):
    # Updated default gamma to 2.5.
    def __init__(self, gamma=2.5, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

        # Register class weights so they automatically map to the GPU
        self.register_buffer('class_weights', torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32))

    def dice_loss(self, pred, target):
        smooth = 1e-6

        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(
            target, num_classes=config.NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)

        #proper reduction
        dice_per_class = dice.mean(dim=0)

        weights = self.class_weights / self.class_weights.sum()

        return 1 - (dice_per_class * weights).sum()

    # FIX: Indented to be a method of the class
    def forward(self, pred_masks, true_masks, pred_edges, true_edges):
        # 1. FOCAL LOSS (Now using CLASS_WEIGHTS inside cross_entropy)
        ce_loss = F.cross_entropy(
            pred_masks,
            true_masks,
            weight=self.class_weights,  # Applies the [1.0, 1.5, 3.5, 2.5, 2.0] balancing
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

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