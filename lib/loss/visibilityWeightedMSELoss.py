import torch
import torch.nn as nn

class VisibilityWeightedMSELoss(nn.Module):
    def __init__(self):
        super(VisibilityWeightedMSELoss, self).__init__()

    def forward(self, output, target, visibility):
        # output, target: shape [batch, n_points, ...]
        # visibility: shape [batch, n_points] (1 for visible, 0 for occluded)

        # Expand visibility to match output shape
        visibility = visibility.unsqueeze(-1).expand_as(target)
        diff = (output - target) * visibility
        loss = (diff ** 2).sum() / (visibility.sum() + 1e-8)  # avoid div by zero
        return loss
