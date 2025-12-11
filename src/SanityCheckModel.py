import torch
import torch.nn as nn
import torch.nn.functional as F

class baseline_arch(nn.Module):
    def __init__(self, upscale=1):
        self.upscale = upscale
        super().__init__()

    def forward(self, x):
        """
        x: [B, 3, H, W] RGB image
        returns: same tensor as input
        """
        if (self.upscale != 1):
            return F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return x