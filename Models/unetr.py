import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNETR_base_3DNet(nn.Module):
    def __init__(self, num_classes=2, token_grid=(6, 6, 6)):
        super().__init__()
        self.token_grid = token_grid
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _ConvBlock3D(64, 128, stride=1)
        self.layer2 = _ConvBlock3D(128, 256, stride=2)
        self.layer3 = _ConvBlock3D(256, 512, stride=2)
        self.proj = nn.Conv3d(512, 768, kernel_size=1)
        self.norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        x = F.adaptive_avg_pool3d(x, self.token_grid)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled
