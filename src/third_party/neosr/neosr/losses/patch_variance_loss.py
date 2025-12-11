import torch
import torch.nn.functional as F
from neosr.utils.registry import LOSS_REGISTRY
import torch.nn as nn

'''
Loss, that goes from patch to patch of the image(supposed 32x32 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''
def patch_variance_loss_fn(pred, target, patch_size=20, alpha=430, loss_weight: float = 1.0):
    B, C, H, W = target.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    #fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Flatten patches: [B, C*patch_size*patch_size, num_patches]
    pred_patches = unfold(pred)
    target_patches = unfold(target)

    # Compute variance per patch (use unbiased=False for stability)
    target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]
    
    # Compute MSE per patch
    diff = (pred_patches - target_patches) * alpha
    patch_mse = (diff ** 2).mean(dim=1) / alpha  # [B, num_patches]
    #print(patch_mse.shape)
    # Normalize variance (avoid division by zero)
    var_norm = target_var / (target_var.mean(dim=1, keepdim=True) + 1e-8)
    #print(var_norm.shape)
    # Compute weighted loss
    weights = 1.0 + var_norm
    loss = (weights * patch_mse).mean()

    return loss  * loss_weight

@LOSS_REGISTRY.register()
class patch_variance_loss(nn.Module):
    def __init__(self, patch_size=20, alpha=430, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.alpha = alpha
    
    def forward(self, pred, target):
        return patch_variance_loss_fn(pred, target, self.patch_size, self.alpha, self.loss_weight)
        
