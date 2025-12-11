import torch
import torch.nn as nn
from neosr.losses.ssim_loss import mssim_loss
from neosr.utils.registry import LOSS_REGISTRY
from torch import Tensor, nn
import torch.nn.functional as F

class GaussianFilter2D(nn.Module):
    def _get_gaussian_window1d(self) -> Tensor:
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d) -> Tensor:
        return torch.matmul(
            gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d
        )

    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        
        padding: int | None = None,
    ) -> None:
        """2D Gaussian Filer.

        Args:
        ----
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.

        """
        super().__init__()
        self.window_size = window_size
        if window_size % 2 != 1:
            msg = "Window size must be odd."
            raise ValueError(msg)
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma
        
        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(
            name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            input=x,
            weight=self.gaussian_window,
            stride=1,
            padding=self.padding,
            groups=x.shape[1],
        )


class CombinedLossSSIMCharbMSE(nn.Module):
    def __init__(self, alpha_charb=0.3, alpha_mse=0.1, alpha_ssim=0.6, eps=1e-6,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
        L: int = 1,
        padding: int | None = None):
        """
        Combines Charbonnier, MSE, and SSIM loss into a single scalar.
        """
        super().__init__()
        self.alpha_charb = alpha_charb
        self.alpha_mse   = alpha_mse
        self.alpha_ssim  = alpha_ssim
        self.eps = eps
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.mse = nn.MSELoss(reduction="none")
        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )
        
    def _ssim(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l1 = A1 / B1
        cs = A2 / B2
        ssim = l1 * cs

        return ssim

    def charbonnier_loss(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps**2)

    def forward(self, pred, target):
        """
        pred, target: [B, C, H, W] in [0,1]
        Returns: scalar loss
        """
        #print("in_shape:", pred.shape)
        #print("expected_out_shape: [B,patches]")
        # Flatten patches: [B, C*patch_size*patch_size, num_patches]
        
        # ---- Charbonnier Loss Per Patch----
        charb = self.charbonnier_loss(pred, target)
        #print("charb:", charb.mean(dim=(1, 2, 3)).mean())
        # ---- MSE Loss Per Patch----
        mse_loss = self.mse(pred, target) * 20 #increase mse sensibility
        #print("MSE:", mse_loss.mean(dim=(1, 2, 3)).mean())
        # ---- SSIM Loss Per Patch----
        ssim_loss = 1.0 - self._ssim(pred, target)  
        #print("SSIM:", ssim_loss.mean(dim=(1, 2, 3)).mean())
        # ---- Weighted combination ----
        total_loss = (self.alpha_charb * charb) + (self.alpha_mse * mse_loss) + (self.alpha_ssim * ssim_loss)
        
        # ---- Reduce to one value per patch ----
        #total_loss = total_loss.mean(dim=(1, 2, 3))  # -> [B]
        #print("Combined:", total_loss.mean())
        return total_loss

