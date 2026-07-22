from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.projection(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3DBranch(nn.Module):
    """Native 3-D U-Net branch used for volumetric fault segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.enc1 = ConvBlock3D(in_channels, channels[0])
        self.enc2 = ConvBlock3D(channels[0], channels[1])
        self.enc3 = ConvBlock3D(channels[1], channels[2])
        self.enc4 = ConvBlock3D(channels[2], channels[3])
        self.bottleneck = ConvBlock3D(channels[3], channels[4])
        self.pool = nn.MaxPool3d(2)

        self.up4 = UpBlock3D(channels[4], channels[3], channels[3])
        self.up3 = UpBlock3D(channels[3], channels[2], channels[2])
        self.up2 = UpBlock3D(channels[2], channels[1], channels[1])
        self.up1 = UpBlock3D(channels[1], channels[0], channels[0])
        self.classifier = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(volume)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        x = self.up4(bottleneck, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.classifier(x)
