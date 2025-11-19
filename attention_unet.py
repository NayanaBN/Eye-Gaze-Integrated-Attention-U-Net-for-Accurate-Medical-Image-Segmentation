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


class AttentionBlock(nn.Module):
    """Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttUNet(nn.Module):
    """
    Attention U-Net for segmentation.
    in_ch = 1 (baseline) or 2 (gaze-guided)
    """
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_ch, base)                         # out: base
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))   # out: base*2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))  # out: base*4

        # Decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=base * 2, F_l=base * 2, F_int=base)  # both 64 channels if base=32
        self.conv1 = DoubleConv(base * 4, base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=base, F_l=base, F_int=base // 2)     # both 32 channels if base=32
        self.conv2 = DoubleConv(base * 2, base)

        # Output
        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)   # [B, base, H, W]
        x2 = self.down1(x1)  # [B, base*2, H/2, W/2]
        x3 = self.down2(x2)  # [B, base*4, H/4, W/4]

        # Decoder
        g1 = self.up1(x3)       # [B, base*2, H/2, W/2]
        x2_att = self.att1(g1, x2)
        x = torch.cat([g1, x2_att], dim=1)
        x = self.conv1(x)

        g2 = self.up2(x)        # [B, base, H, W]
        x1_att = self.att2(g2, x1)
        x = torch.cat([g2, x1_att], dim=1)
        x = self.conv2(x)

        return self.outc(x)
