import torch
import torch.nn.functional as F
from neosr.utils.registry import LOSS_REGISTRY
import torch.nn as nn
from neosr.losses.CombinedLossSSIMCharbMSE import CombinedLossSSIMCharbMSE
'''
Loss, that goes from patch to patch of the image(supposed 32x32 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''

@LOSS_REGISTRY.register()
class patch_difference_variation_combined_loss(nn.Module):
    def __init__(self, patch_size=20, upscale=1, alpha=7, loss_weight: float = 1.0,alpha_charb=0.3, alpha_mse=0.1, alpha_ssim=0.6, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.alpha = alpha
        self.upscale=upscale
        self.combinedLoss = CombinedLossSSIMCharbMSE(alpha_charb, alpha_mse, alpha_ssim, eps)
    
    def patch_difference_variation_loss_fn(self,pred, target,source):
        B, C, H, W = target.shape
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        #fold = torch.nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        sourceCopy = None
        if (self.upscale!=1):
            sourceCopy = F.interpolate(source, scale_factor=self.upscale, mode='bicubic')
        else:
            sourceCopy = source

        # Unfold directly gives [B, C*patch_size*patch_size, num_patches]
        pred_patches = unfold(pred)   # [B, C*P*P, N]
        target_patches = unfold(target)
        source_patches = unfold(sourceCopy)

        num_patches = pred_patches.shape[-1]
        P = self.patch_size

        # Compute variance per patch (over C*P*P dimension)
        #target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]
        target_diff = (abs(target_patches - source_patches)).mean(dim=1)
        target_var = target_patches.var(dim=1, unbiased=False)

        # Reshape for loss computation: [B*N, C, P, P]
        pred_patches_flat = pred_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)
        target_patches_flat = target_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)

        # Combined loss per patch
        patch_loss = self.combinedLoss(pred_patches_flat, target_patches_flat).mean(dim=(1, 2, 3))
        patch_loss = patch_loss.view(B, num_patches)

        # Normalize variance and weight
        diff_norm = target_diff / (target_diff.max(dim=1, keepdim=True)[0] + 1e-8)
        var_norm = target_var / (target_var.max(dim=1, keepdim=True)[0] + 1e-8)
        weights = 1.0 + diff_norm + var_norm
        

        return ((weights * patch_loss)).mean() * self.alpha
    
    def forward(self, pred, target,source):
        return self.patch_difference_variation_loss_fn(pred, target,source) * self.loss_weight
        
