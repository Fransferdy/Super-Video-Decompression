import torch
import torch.nn.functional as F
from neosr.utils.registry import LOSS_REGISTRY
import torch.nn as nn
from neosr.losses.CombinedLossSSIMCharbMSEReduct import CombinedLossSSIMCharbMSE
'''
Loss, that goes from patch to patch of the image(supposed 32x32 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''

@LOSS_REGISTRY.register()
class patch_variance_combined_loss(nn.Module):
    def __init__(self, patch_size=20, alpha=40, loss_weight: float = 1.0,alpha_charb=0.3, alpha_mse=0.1, alpha_ssim=0.6, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.alpha = alpha
        self.combinedLoss = CombinedLossSSIMCharbMSE(alpha_charb, alpha_mse, alpha_ssim, eps)
    
    def patch_variance_loss_fn(self,pred, target):
        B, C, H, W = target.shape
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        #fold = torch.nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

        # Flatten patches: [B, C*patch_size*patch_size, num_patches]
        pred_patches = unfold(pred)
        target_patches = unfold(target)
        num_patches = pred_patches.shape[-1]
        
        pred_patches_img = pred_patches.transpose(1, 2).reshape(B, num_patches, C, self.patch_size, self.patch_size)
        target_patches_img = target_patches.transpose(1, 2).reshape(B, num_patches, C, self.patch_size, self.patch_size)

        pred_patches_flat = pred_patches_img.reshape(B * num_patches, C, self.patch_size, self.patch_size)
        target_patches_flat = target_patches_img.reshape(B * num_patches, C, self.patch_size, self.patch_size)

        # Compute variance per patch (use unbiased=False for stability)
        target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]

        # Compute Loss per patch
        patch_loss = self.combinedLoss(pred_patches_flat, target_patches_flat)

        # Normalize variance (avoid division by zero)
        var_norm = target_var / (target_var.mean(dim=1, keepdim=True) + 1e-8)

        # Compute weighted loss
        weights = 1.0 + var_norm
        loss = (weights * patch_loss).mean() * self.alpha

        return loss
    
    def forward(self, pred, target):
        return self.patch_variance_loss_fn(pred, target) * self.loss_weight
        
