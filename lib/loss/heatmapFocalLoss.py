import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: [B, N, H, W] with gt in [0,1], pred in (0,1)
        Returns a single scalar: mean over all pixels in the batch.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        pos_loss = - ((1 - pred) ** self.alpha) * torch.log(pred + self.eps) * pos_inds
        neg_loss = - ((1 - gt) ** self.beta)   * (pred ** self.alpha)   * torch.log(1 - pred + self.eps) * neg_inds

        loss = pos_loss + neg_loss
        return loss.mean()   # mean over B×N×H×W