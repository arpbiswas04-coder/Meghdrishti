import torch
import torch.nn as nn
import torch.nn.functional as F

class CloudDetectionHead(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        
        # Encoder (3 blocks)
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256)
        
        # Decoder (3 blocks)
        self.dec3 = self._block(256 + 128, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64 + 32, 32)
        
        # Output
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, kernel_size=2, stride=2)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        u3 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        # Logits (no sigmoid, applied outside)
        logits = self.out_conv(d1)
        
        return logits
