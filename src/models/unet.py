import torch
import torch.nn as nn
import torch.nn.functional as F

class BioCPPNet(nn.Module):
    """
    U-Net architecture for bioacoustic source separation.
    Designed for 250kHz sampling rates.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._conv_block(in_channels, hidden_dim)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)
        self.enc3 = self._conv_block(hidden_dim * 2, hidden_dim * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(hidden_dim * 4, hidden_dim * 8)
        
        # Decoder (Expanding Path)
        self.up3 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(hidden_dim * 8, hidden_dim * 4)
        
        self.up2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(hidden_dim * 4, hidden_dim * 2)
        
        self.up1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(hidden_dim * 2, hidden_dim)
        
        self.final = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
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
        # Pad if needed or crop? U-Net usually pads input to handle size changes
        # Assuming input dimensions are powers of 2 for simplicity here
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)
