
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights



def _make_grid(h, w):
    yy, xx = torch.meshgrid(
    torch.arange(h).float() / (h - 1) * 2 - 1,
    torch.arange(w).float() / (w - 1) * 2 - 1,
    indexing='ij'  # <-- add this
    )
    return yy, xx


def get_coords_from_heatmap(heatmap):
    batch, npoints, h, w = heatmap.shape

    yy, xx = _make_grid(h, w)
    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum  # batch x npoints
    coords = torch.stack([xx_coord, yy_coord], dim=-1)

    return coords

class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels=256, out_channels=256):
        super().__init__()
        self.up1 = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'


class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        super().__init__()

        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # use the torchvision-provided weights & architecture
        if pretrained:
            # this will download (or use cached) ImageNet-pretrained weights
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            backbone = mobilenet_v3_large(weights=None)

        # grab only the feature‐extracting layers
        self.features = backbone.features  

        # figure out output channel count dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            out = self.features(dummy)
            self.out_channels = out.shape[1]

    def forward(self, x):
        return self.features(x)


class StageHead(nn.Module):
    def __init__(self, in_channels, num_heats, num_edges, num_points):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Output heads
        self.heat_out = nn.Conv2d(256, num_heats, 1)
        self.edge_out = nn.Conv2d(256, num_edges, 1)
        self.point_out = nn.Conv2d(256, num_points, 1)

    def forward(self, x):
        feat = self.conv(x)
        heat = self.heat_out(feat)
        edge = self.edge_out(feat)
        point = self.point_out(feat)
        return feat, heat, edge, point


class ADNetMobileV3MultiStage(nn.Module):
    def __init__(self, classes_num, edge_info, nstack=2, pretrained=True):
        super().__init__()
        self.num_heats = classes_num[0]
        self.num_edges = classes_num[1]
        self.num_points = classes_num[2]
        self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)
        self.nstack = nstack

        self.backbone = MobileNetFeatureExtractor(pretrained=pretrained)
        in_channels = self.backbone.out_channels  # typically 960

        # Add decoder to upsample backbone features from 8x8 to 64x64
        self.decoder = UpsampleDecoder(in_channels=in_channels, mid_channels=256, out_channels=256)

        # Now StageHead expects 256 channels from decoder output
        self.stages = nn.ModuleList([
            StageHead(256, self.num_heats, self.num_edges, self.num_points)
            for _ in range(nstack)
        ])

        # Merge layers for intermediate stages (same 256 channels as StageHead input)
        self.merge_features = nn.ModuleList([
            nn.Conv2d(256, 256, 1) for _ in range(nstack - 1)
        ])
        self.merge_heatmaps = nn.ModuleList([
            nn.Conv2d(self.num_heats, 256, 1) for _ in range(nstack - 1)
        ])
        self.merge_edgemaps = nn.ModuleList([
            nn.Conv2d(self.num_edges, 256, 1) for _ in range(nstack - 1)
        ])
        self.merge_pointmaps = nn.ModuleList([
            nn.Conv2d(self.num_points, 256, 1) for _ in range(nstack - 1)
        ])

        self.heatmap_act = Activation("in+relu", self.num_heats)
        self.edgemap_act = Activation("sigmoid", self.num_edges)
        self.pointmap_act = Activation("sigmoid", self.num_points)

    def set_inference(self, inference):
        self.inference = inference

    def forward(self, x):
        base_feat = self.backbone(x)  # [B, C, 8, 8]
        base_feat = self.decoder(base_feat)  # [B, 256, 64, 64]

        outputs = []

        for i in range(self.nstack):
            feat, heatmaps0, edgemaps0, pointmaps0 = self.stages[i](base_feat)

            heatmaps = self.heatmap_act(heatmaps0)
            edgemaps = self.edgemap_act(edgemaps0)
            pointmaps = self.pointmap_act(pointmaps0)

            edge_point_attention_mask = self.e2h_transform(edgemaps) * pointmaps
            landmarks = get_coords_from_heatmap(edge_point_attention_mask * heatmaps)

            # Merge features back for next stage except last
            if i < self.nstack - 1:
                base_feat = base_feat + \
                            self.merge_features[i](feat) + \
                            self.merge_heatmaps[i](heatmaps) + \
                            self.merge_edgemaps[i](edgemaps) + \
                            self.merge_pointmaps[i](pointmaps)

            outputs.append(landmarks)
            outputs.append(edgemaps)
            outputs.append(pointmaps)

        final_landmarks = outputs[-3]
        return outputs, edge_point_attention_mask, final_landmarks
        
