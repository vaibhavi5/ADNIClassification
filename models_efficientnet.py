import torch
import torch.nn as nn
from math import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = Swish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.use_se = se_ratio is not None

        expand_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv3d(in_channels, expand_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm3d(expand_channels)
        self.expand_act = Swish()

        depthwise_padding = kernel_size // 2
        self.depthwise_conv = nn.Conv3d(expand_channels, expand_channels, kernel_size, stride, depthwise_padding, groups=expand_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm3d(expand_channels)
        self.depthwise_act = Swish()

        if self.use_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv3d(expand_channels, num_squeezed_channels, kernel_size=1)
            self.se_expand = nn.Conv3d(num_squeezed_channels, expand_channels, kernel_size=1)

        self.project_conv = nn.Conv3d(expand_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = x

        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)

        if self.use_se:
            se = x.mean(dim=(2, 3, 4), keepdim=True)
            se = self.se_reduce(se)
            se = self.se_expand(se)
            x = torch.sigmoid(se) * x

        x = self.project_conv(x)
        x = self.project_bn(x)

        if identity.size(1) == x.size(1) and identity.size(2) == x.size(2) and identity.size(3) == x.size(3) and identity.size(4) == x.size(4):
            x += identity

        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, width_multiplier=1.0, depth_multiplier=1.0):
        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        channels = [int(c * width_multiplier) for c in channels]
        depths = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]

        self.features = nn.Sequential(
            ConvBlock(num_channels, channels[0], kernel_size=3, stride=1, padding=1),
            MBConvBlock(channels[0], channels[1], kernel_size=3, stride=strides[0], expand_ratio=1, se_ratio=0.25),
            MBConvBlock(channels[1], channels[2], kernel_size=3, stride=strides[1], expand_ratio=6, se_ratio=0.25),
            MBConvBlock(channels[2], channels[3], kernel_size=5, stride=strides[2], expand_ratio=6, se_ratio=0.25),
            MBConvBlock(channels[3], channels[4], kernel_size=3, stride=strides[3], expand_ratio=6, se_ratio=0.25),
            MBConvBlock(channels[4], channels[5], kernel_size=5, stride=strides[4], expand_ratio=6, se_ratio=0.25),
            MBConvBlock(channels[5], channels[6], kernel_size=3, stride=strides[5], expand_ratio=6, se_ratio=0.25),
            ConvBlock(channels[6], channels[7], kernel_size=1, stride=strides[6], padding=0),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(channels[7], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]
