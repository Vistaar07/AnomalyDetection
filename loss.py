import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareOrdinalFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_masks, true_masks, pred_edges, true_edges):
        # -----------------------------
        # 1. FOCAL LOSS
        # -----------------------------
        ce_loss = F.cross_entropy(pred_masks, true_masks, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        # -----------------------------
        # 2. ORDINAL LOSS (DIFFERENTIABLE)
        # -----------------------------
        num_classes = pred_masks.shape[1]
        probs = torch.softmax(pred_masks, dim=1)
        class_ids = torch.arange(num_classes, device=pred_masks.device).view(1, -1, 1, 1)
        pred_expected = (probs * class_ids).sum(dim=1)

        true_classes = true_masks.float()
        mask_fg = true_classes > 0

        if mask_fg.sum() > 0:
            ordinal_loss = F.l1_loss(
                pred_expected[mask_fg],
                true_classes[mask_fg]
            )
        else:
            ordinal_loss = torch.tensor(0.0, device=pred_masks.device)

        # -----------------------------
        # 3. BOUNDARY LOSS
        # -----------------------------
        true_edges = true_edges.unsqueeze(1).float()
        pred_edges = pred_edges.unsqueeze(1).float()
        edge_loss = self.bce(pred_edges, true_edges)

        # -----------------------------
        # TOTAL
        # -----------------------------
        import config
        total_loss = (
            config.LAMBDA_FOCAL_DICE * focal_loss
            + config.LAMBDA_ORDINAL * ordinal_loss
            + config.LAMBDA_BOUNDARY * edge_loss
        )

        return total_loss
