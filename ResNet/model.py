import torch
import torch.nn as nn
import copy
from typing import Union, Optional


class ResidualConnectionBlock(nn.Module):
    def __init__(self, layers: Union[list[nn.Module],nn.Module], times:Union[list[int],int] = 1, IdentityLayers: nn.Module = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        if isinstance(layers, nn.Module):
            layers = [layers]
        if isinstance(times, int):
            times = [times]*len(layers)
        if len(layers) != len(times):
            raise ValueError("The length of layers and times must be equal.")
        for i, layer in enumerate(layers):
            repeat_count = times[i]
            for _ in range(repeat_count):
                self.layers.append(copy.deepcopy(layer))
        self.IdentityLayers = IdentityLayers
        self.relu = nn.ReLU(inplace=True)


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.IdentityLayers:
            identity = self.IdentityLayers(x.clone())
        else:
            identity = x.clone()

        for layer in self.layers:
            x = layer(x)
        
        x = x + identity
        x = self.relu(x)
        return x
    
class ResNet18(nn.Module):
    """
    ResNet-18
    输入：[3, 224, 224]
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        ), 2)
        self.conv3 = ResidualConnectionBlock(
            layers = [nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
            ),nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )], 
            times=[1,1],
            IdentityLayers = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
            ))
        self.conv4 = ResidualConnectionBlock(
            layers = [nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
            ),nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
            )],
            times=[1,1],
            IdentityLayers = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
            )
        )
        self.conv5 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        ), 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
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
        self.conv2 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        ), 3)
        self.conv3 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        ), 4)
        self.conv4 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        ), 6)
        self.conv5 = ResidualConnectionBlock(nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
        ), 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

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
