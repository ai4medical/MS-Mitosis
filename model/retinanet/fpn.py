"""RetinaFPN in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import *


class FPN(nn.Module):
    def __init__(self, resnet_size: int):
        super().__init__()
        self.resnet_size = resnet_size
        assert self.resnet_size in [18, 34, 50, 101, 152]

        # self.backbone = eval(
        #     f"resnet{resnet_size}(weights=ResNet{resnet_size}_Weights.DEFAULT)"
        # )

        self.backbone = eval(
            f"resnet{resnet_size}()"
        )

        if self.resnet_size <= 34:
            fpn_sizes = [
                self.backbone.layer4[-1].conv2.out_channels,
                self.backbone.layer3[-1].conv2.out_channels,
                self.backbone.layer2[-1].conv2.out_channels,
            ]
        else:
            fpn_sizes = [
                self.backbone.layer4[-1].conv3.out_channels,
                self.backbone.layer3[-1].conv3.out_channels,
                self.backbone.layer2[-1].conv3.out_channels,
            ]

        self.conv6 = nn.Conv2d(fpn_sizes[0], 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            fpn_sizes[0], 256, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            fpn_sizes[1], 256, kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            fpn_sizes[2], 256, kernel_size=1, stride=1, padding=0
        )

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        # Bottom-up
        # with torch.no_grad():
        c1 = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.backbone.layer1(c1)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        # print(p3.shape, p4.shape, p5.shape, p6.shape, p7.shape)
        # assert False
        return p3, p4, p5, p6, p7
