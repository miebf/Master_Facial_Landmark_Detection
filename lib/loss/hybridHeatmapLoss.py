import torch
import torch.nn as nn

from lib.loss.heatmapFocalLoss import HeatmapFocalLoss
from lib.loss.awingLoss import AWingLoss
from conf import *

class HybridHeatmapLoss(nn.Module):
    """alpha x AWing + beta x Focal on the same heatmaps."""
    def __init__(self, alpha_aw=0.5, alpha_focal=1.0,
                 aw_cfg={}, focal_cfg={}):
        super().__init__()
        self.aw = AWingLoss(**aw_cfg)
        self.focal = HeatmapFocalLoss(**focal_cfg)
        self.alpha_aw = alpha_aw
        self.alpha_focal = alpha_focal

    def forward(self, pred_heatmap, gt_heatmap):
        loss_aw   = self.aw(pred_heatmap, gt_heatmap)
        loss_foc  = self.focal(pred_heatmap, gt_heatmap)
        total_loss = self.alpha_aw * loss_aw + self.alpha_focal * loss_foc
        return total_loss

    def __repr__(self):
        return (f"HybridHeatmapLoss(α_aw={self.alpha_aw}, α_focal={self.alpha_focal}, "
                f"AWing={self.aw}, Focal={self.focal})")