import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_utils import weights_init

class PatchGAN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids=None):
        super().__init__()
        self.gpu_ids = gpu_ids
        total_channels = in_ch + out_ch
        
        self.D1 = PatchGAN(total_channels)
        self.D2 = PatchGAN(total_channels)
        
        self.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids and len(self.gpu_ids) > 1:
            d1_out = nn.parallel.data_parallel(self.D1, x, self.gpu_ids)
            x_down = F.avg_pool2d(x, kernel_size=2)
            d2_out = nn.parallel.data_parallel(self.D2, x_down, self.gpu_ids)
        else:
            d1_out = self.D1(x)
            x_down = F.avg_pool2d(x, kernel_size=2)
            d2_out = self.D2(x_down)
            
        return d1_out, d2_out