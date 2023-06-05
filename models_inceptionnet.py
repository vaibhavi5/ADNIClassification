import torch.nn as nn
import torch

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.branch1x1 = nn.Sequential(
            nn.Conv3d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm3d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 conv branch
        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm3d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 conv branch
        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm3d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm3d(out_pool),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)
        out = torch.cat([out1x1, out3x3, out5x5, out_pool], dim=1)
        return out

class InceptionNet3D(nn.Module):
    def __init__(self, num_classes=2, num_channels=1):
        super(InceptionNet3D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64*num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv3d(64*num_channels, 64*num_channels, kernel_size=1),
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64*num_channels, 192*num_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(192*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            InceptionModule(192*num_channels, 64*num_channels, 96*num_channels, 128*num_channels, 16*num_channels, 32*num_channels, 32*num_channels),
            InceptionModule(256*num_channels, 128*num_channels, 128*num_channels, 192*num_channels, 32*num_channels, 96*num_channels, 64*num_channels),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            InceptionModule(480*num_channels, 192*num_channels, 96*num_channels, 208*num_channels, 16*num_channels, 48*num_channels, 64*num_channels),
            InceptionModule(512*num_channels, 160*num_channels, 112*num_channels, 224*num_channels, 24*num_channels, 64*num_channels, 64*num_channels),
            InceptionModule(512*num_channels, 128*num_channels, 128*num_channels, 256*num_channels, 24*num_channels, 64*num_channels, 64*num_channels),
            InceptionModule(512*num_channels, 112*num_channels, 144*num_channels, 288*num_channels, 32*num_channels, 64*num_channels, 64*num_channels),
            InceptionModule(528*num_channels, 256*num_channels, 160*num_channels, 320*num_channels, 32*num_channels, 128*num_channels, 128*num_channels),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            
            InceptionModule(832*num_channels, 256*num_channels, 160*num_channels, 320*num_channels, 32*num_channels, 128*num_channels, 128*num_channels),
            InceptionModule(832*num_channels, 384*num_channels, 192*num_channels, 384*num_channels, 48*num_channels, 128*num_channels, 128*num_channels),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(832*num_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]
