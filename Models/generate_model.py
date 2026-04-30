import torch.nn as nn


class _BasicBlock3D(nn.Module):
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


class _SimpleResNet3D(nn.Module):
    def __init__(self, in_channels=1, nb_class=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(_BasicBlock3D(64, 64), _BasicBlock3D(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock3D(64, 128, stride=2), _BasicBlock3D(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock3D(128, 256, stride=2), _BasicBlock3D(256, 256))
        self.layer4 = nn.Sequential(_BasicBlock3D(256, 512, stride=2), _BasicBlock3D(512, 512))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, nb_class)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def generate_model(
    model_type="resnet",
    model_depth=10,
    input_W=128,
    input_H=128,
    input_D=128,
    resnet_shortcut="B",
    no_cuda=True,
    gpu_id=None,
    pretrain_path=None,
    nb_class=2,
):
    return _SimpleResNet3D(in_channels=1, nb_class=nb_class)
