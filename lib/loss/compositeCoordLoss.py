import torch
import torch.nn as nn

from lib.loss.anisotropicDirectionLoss import AnisotropicDirectionLoss
from lib.loss.visibilityWeightedMSELoss import VisibilityWeightedMSELoss
from conf import *

class CompositeCoordLoss(nn.Module):
    def __init__(self, loss_lambda=2.0, edge_info=None, alpha=1.0, beta=0.1):
        super().__init__()
        self.ani_loss = AnisotropicDirectionLoss(loss_lambda=loss_lambda, edge_info=edge_info)
        self.vis_loss = VisibilityWeightedMSELoss()
        self.alpha = alpha  # Weight for AnisotropicDirectionLoss
        self.beta = beta    # Weight for VisibilityWeightedMSELoss

    def forward(self, output, target, heatmap=None, landmarks=None, visibility=None):
        loss_ani = self.ani_loss(output, target, heatmap=heatmap, landmarks=landmarks)
        loss_vis = self.vis_loss(output, target, visibility=visibility)
        total_loss = self.alpha * loss_ani + self.beta * loss_vis

        return total_loss