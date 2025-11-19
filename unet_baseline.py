import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Simplified U-Net for baseline and gaze-guided segmentation.
    in_ch = 1 (baseline) or 2 (with gaze heatmap)
    """
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))

        # Decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base * 2, base)

        # Output
        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return self.outc(x)
