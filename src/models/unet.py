import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseUNet(nn.Module):
    """
    Abstract base class for U-Net variants.
    """
    def build_conv_block(self, in_ch, out_ch):
        """Factory method for convolution blocks."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def build_up_block(self, in_ch, out_ch):
        """
        Factory method for upsampling blocks.
        Uses Bilinear Upsampling + Conv to avoid checkerboard artifacts 
        (better than ConvTranspose2d).
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class BioCPPNet(BaseUNet):
    """
    U-Net architecture for bioacoustic source separation.
    Designed for 250kHz sampling rates.
    """
    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, hidden_dim: int = 64
    ):
        super().__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.build_conv_block(in_channels, hidden_dim)
        self.enc2 = self.build_conv_block(hidden_dim, hidden_dim * 2)
        self.enc3 = self.build_conv_block(hidden_dim * 2, hidden_dim * 4)

        # Bottleneck
        self.bottleneck = self.build_conv_block(hidden_dim * 4, hidden_dim * 8)

        # Decoder (Expanding Path)
        # Input to up-block is previous layer output (dim*8). 
        # Output should match skip connection (dim*4).
        self.up3 = self.build_up_block(hidden_dim * 8, hidden_dim * 4)
        self.dec3 = self.build_conv_block(hidden_dim * 8, hidden_dim * 4)

        self.up2 = self.build_up_block(hidden_dim * 4, hidden_dim * 2)
        self.dec2 = self.build_conv_block(hidden_dim * 4, hidden_dim * 2)

        self.up1 = self.build_up_block(hidden_dim * 2, hidden_dim)
        self.dec1 = self.build_conv_block(hidden_dim * 2, hidden_dim)

        self.final = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        # Handle padding if size doesn't match due to odd input dim
        if u3.shape != e3.shape:
            u3 = F.interpolate(
                u3, size=e3.shape[2:], mode="bilinear", align_corners=False
            )

        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape != e2.shape:
            u2 = F.interpolate(
                u2, size=e2.shape[2:], mode="bilinear", align_corners=False
            )

        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = F.interpolate(
                u1, size=e1.shape[2:], mode="bilinear", align_corners=False
            )

        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)
