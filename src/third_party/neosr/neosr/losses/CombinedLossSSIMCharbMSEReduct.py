import torch
import torch.nn as nn
from neosr.losses.ssim_loss import mssim_loss
from neosr.utils.registry import LOSS_REGISTRY

class CombinedLossSSIMCharbMSE(nn.Module):
    def __init__(self, alpha_charb=0.3, alpha_mse=0.1, alpha_ssim=0.6, eps=1e-6):
        """
        Combines Charbonnier, MSE, and SSIM loss into a single scalar.
        """
        super().__init__()
        self.alpha_charb = alpha_charb
        self.alpha_mse   = alpha_mse
        self.alpha_ssim  = alpha_ssim
        self.eps = eps
        self.mse = nn.MSELoss(reduction="mean")
        self.ms_ssim = mssim_loss()

    def charbonnier_loss(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps**2).mean()

    def forward(self, pred, target):
        """
        pred, target: [B, C, H, W] in [0,1]
        Returns: scalar loss
        """
        # ---- Charbonnier Loss ----
        charb = self.charbonnier_loss(pred, target)
        #print("charb:", charb)
        # ---- MSE Loss ----
        mse_loss = self.mse(pred, target) * 14 #increase mse sensibility
        #print("MSE:", mse_loss)
        # ---- SSIM Loss ----
        ssim_loss = self.ms_ssim(pred, target)  
        #print("SSIM:", ssim_loss)
        # ---- Weighted combination ----
        total_loss = (self.alpha_charb * charb) + (self.alpha_mse * mse_loss) + (self.alpha_ssim * ssim_loss)
        #print("Combined:", total_loss)
        return total_loss