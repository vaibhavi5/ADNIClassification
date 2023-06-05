import torch
import torch.nn as nn
from math import *

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(XceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + shortcut
        x = self.relu(x)

        return x

class XceptionNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1):
        super(XceptionNet, self).__init__()
        self.entry_flow = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 728, stride=2),
        )

        self.middle_flow = nn.Sequential(
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
            XceptionBlock(728, 728),
        )

        self.exit_flow = nn.Sequential(
            XceptionBlock(728, 1024, stride=2),
            SeparableConv2d(1024, 1536, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
