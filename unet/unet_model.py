# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts_gn import *

class UNetStar(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetStar, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64,bilinear=True)
        self.up2 = up(128, 32,bilinear=True)
        self.up3 = up(64, 32,bilinear=True)
        self.features = nn.Conv2d(32,128,3,padding=1)
        self.out_ray = outconv(128, n_classes)
        self.final_activation_ray = nn.ReLU()
        self.out_prob = outconv(128, 1)
        self.final_activation_prob = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.features(x)
        out_ray = self.out_ray(x)
        out_ray = self.final_activation_ray(out_ray)
        out_prob = self.out_prob(x)
        out_prob = self.final_activation_prob(out_prob)
        return [out_ray,out_prob]