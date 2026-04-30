import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class OCTResNet3DBackbone(nn.Module):
    def __init__(self, num_classes=2, token_grid=(6, 6, 6), target_dim=768):
        super().__init__()
        self.token_grid = token_grid
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResidualBlock3D(64, 128), _ResidualBlock3D(128, 128))
        self.layer2 = nn.Sequential(_ResidualBlock3D(128, 256, stride=2), _ResidualBlock3D(256, 256))
        self.layer3 = nn.Sequential(_ResidualBlock3D(256, 512, stride=2), _ResidualBlock3D(512, 512))
        self.layer4 = nn.Sequential(_ResidualBlock3D(512, 512), _ResidualBlock3D(512, 512))
        self.proj = nn.Conv3d(512, target_dim, kernel_size=1)
        self.norm = nn.LayerNorm(target_dim)
        self.head = nn.Linear(target_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        x = F.adaptive_avg_pool3d(x, self.token_grid)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled


def build_model(num_classes=2):
    return OCTResNet3DBackbone(num_classes=num_classes)
