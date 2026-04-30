import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FundusSwinBackbone(nn.Module):
    def __init__(self, model_name="swin_base_patch4_window12_384", pretrained=False, token_grid=12):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.token_grid = token_grid
        self.out_dim = getattr(self.backbone, "num_features", 1024)
        self.project = nn.Identity() if self.out_dim == 1024 else nn.Linear(self.out_dim, 1024)
        self.norm = nn.LayerNorm(1024)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            if feats.shape[1] == self.out_dim:
                feats = F.adaptive_avg_pool2d(feats, (self.token_grid, self.token_grid))
                feats = feats.flatten(2).transpose(1, 2)
            else:
                feats = feats.permute(0, 3, 1, 2)
                feats = F.adaptive_avg_pool2d(feats, (self.token_grid, self.token_grid))
                feats = feats.flatten(2).transpose(1, 2)
        elif feats.dim() == 3:
            token_count = self.token_grid * self.token_grid
            if feats.shape[1] != token_count:
                b, n, c = feats.shape
                side = max(int(n**0.5), 1)
                feats = feats.transpose(1, 2).reshape(b, c, side, side)
                feats = F.adaptive_avg_pool2d(feats, (self.token_grid, self.token_grid))
                feats = feats.flatten(2).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected feature shape from fundus backbone: {tuple(feats.shape)}")

        feats = self.project(feats)
        feats = self.norm(feats)
        pooled = feats.mean(dim=1)
        return feats, pooled


def build_model():
    return FundusSwinBackbone()
