import torch
import torch.nn as nn
import copy
from typing import Union, Optional


class ResidualConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        
        # Build main convolution path
        conv_layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=2 if downsample else 1, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # Removed inplace to avoid potential issues
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        ]
        self.conv = nn.Sequential(*conv_layers)

        # Handle identity mapping
        if downsample or in_channels != out_channels:
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.identity_downsample = None  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
        x += identity
        x = nn.ReLU(inplace=True)(x)
        return x



    
class ResNet18(nn.Module):
    """
    ResNet-18
    输入：[3, 224, 224]
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.make_layers(64, 64, 2)
        self.conv3 = self.make_layers(64, 128, 2, downsample=True)
        self.conv4 = self.make_layers(128, 256, 2, downsample=True)
        self.conv5 = self.make_layers(256, 512, 2, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layers(self, in_channels, out_channels, num_layers: int, downsample:bool = False) -> nn.Module:
        downsample = [downsample]+[False]*(num_layers-1)
        layers = nn.Sequential()
        for i in range(num_layers):
            layers.add_module(f"block{i}" ,ResidualConnectionBlock(in_channels, out_channels, downsample[i]))
            in_channels = out_channels
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet34(nn.Module):
    """
    ResNet-34
    输入：[3, 224, 224]
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self.make_layers(64, 64, 3)
        self.conv3 = self.make_layers(64, 128, 4, downsample=True)
        self.conv4 = self.make_layers(128, 256, 6, downsample=True)
        self.conv5 = self.make_layers(256, 512, 3, downsample=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layers(self, in_channels, out_channels, num_layers: int, downsample:bool = False) -> nn.Module:
        downsample = [downsample]+[False]*(num_layers-1)
        layers = nn.Sequential()
        for i in range(num_layers):
            layers.add_module(f"block{i}" ,ResidualConnectionBlock(in_channels, out_channels, downsample[i]))
            in_channels = out_channels
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
