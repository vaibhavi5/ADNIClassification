import torch.nn as nn
from math import *

num_channels = 1 #Depending if its multimodal or just fMRI 

class AlexNet3D_Dropout(nn.Module): #AlexNet3D_Deeper_Dropout
    def __init__(self, num_classes=2):  #This is dependent on three way = 3, two way = 2 and regression = 1
        super(AlexNet3D_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64*num_channels, kernel_size=5,
                      stride=2, padding=0, groups=num_channels), #kernel 5 and padding 0 for uni, kernel 3 and padding 1 for multi
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64*num_channels, 128*num_channels, kernel_size=3,
                      stride=1, padding=0, groups=num_channels),
            nn.BatchNorm3d(128*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128*num_channels, 192*num_channels,
                      kernel_size=3, padding=1, groups=num_channels),
            nn.BatchNorm3d(192*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(192*num_channels, 384*num_channels,
                      kernel_size=3, padding=1, groups=num_channels),
            nn.BatchNorm3d(384*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(384*num_channels, 256*num_channels,
                      kernel_size=3, padding=1, groups=num_channels),
            nn.BatchNorm3d(256*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(256*num_channels, 256*num_channels,
                      kernel_size=3, padding=1, groups=num_channels),
            nn.BatchNorm3d(256*num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d([1,1,1]),
        )
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256*num_channels, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]

