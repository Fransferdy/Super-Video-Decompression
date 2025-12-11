import torch
import torch.nn as nn
import torch.nn.functional as F
from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


@ARCH_REGISTRY.register()
class baseline_arch(nn.Module):
    def __init__(self, upscale=upscale):
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