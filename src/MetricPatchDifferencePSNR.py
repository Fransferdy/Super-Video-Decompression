import torch
import torch.nn.functional as F

'''
PSNR Metric, that goes from patch to patch of the image(supposed 20x20 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''

from math import log10
def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR for a single pair of tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * log10((max_val ** 2) / mse.item())

def compute_psnr_batch(pred, target, max_val=1.0):
    """
    pred:   [N, C, H, W]
    target: [N, C, H, W]
    returns: tensor of shape [N]
    """

    # MSE per patch (no reduction over batch)
    mse = F.mse_loss(pred, target, reduction="none")   # shape [N, C, H, W]

    # average over C,H,W to get one MSE per patch
    mse = mse.mean(dim=[1, 2, 3])                      # shape [N]

    # avoid division by zero
    mse = torch.clamp(mse, min=1e-12)

    psnr = 10 * torch.log10((max_val ** 2) / mse)

    return psnr


def patch_difference_psnr(pred, target, source, upscale, patch_size=20):
    B, C, H, W = target.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    sourceCopy = None
    if (upscale!=1):
        sourceCopy = F.interpolate(source, scale_factor=upscale, mode='bicubic')
    else:
        sourceCopy = source

    # Unfold directly gives [B, C*patch_size*patch_size, num_patches]
    pred_patches = unfold(pred)   # [B, C*P*P, N]
    target_patches = unfold(target)
    source_patches = unfold(sourceCopy)

    num_patches = pred_patches.shape[-1]
    P = patch_size

    # Reshape for loss computation: [B*N, C, P, P]
    pred_patches_flat = pred_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)
    target_patches_flat = target_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)

    # Compute variance per patch (over C*P*P dimension)
    target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]
    target_diff = (abs(target_patches - source_patches)).mean(dim=1)

    # Compute MSE per patch
    patch_psnr = compute_psnr_batch(pred_patches_flat, target_patches_flat)
    # Normalize difference (avoid division by zero)
    diff_norm = target_diff / (target_diff.mean(dim=1, keepdim=True)[0] + 1e-8)
    # Compute weighted psnr
    weights = diff_norm
    psnr = (weights * patch_psnr).mean()
    var_norm = target_var / (target_var.mean(dim=1, keepdim=True)[0] + 1e-8)
    # Compute weighted psnr
    weights =  var_norm
    var_psnr = (weights * patch_psnr).mean()

    # Create a tensor filled with the variance values for each patch
    var_expanded = torch.repeat_interleave(
        var_norm.unsqueeze(1), patch_size * patch_size, dim=1
    )
    
    print("Non Weightedpsnr:", patch_psnr.mean())

    return psnr , var_psnr, patch_psnr.mean() #, variance_mask, diff_mask


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def load_image(path, size):
    # Open image, resize, convert to tensor in [0,1]
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),  # Converts to [C,H,W] float32 in [0,1]
    ])
    return transform(img).unsqueeze(0) 

def test_patch_difference_psnr():
    # Create two toy RGB images [B,3,H,W]
    '''
    B, C, H, W = 1, 3, 64, 64
    img1 = torch.zeros((B, C, H, W))
    img2 = torch.zeros((B, C, H, W))

    # Draw a white square in img1
    img1[:, :, 16:48, 16:48] = 1.0
    # Draw a slightly shifted square in img2
    img2[:, :, 20:52, 20:52] = 1.0
    '''   
    img1 = load_image('./frame_00256l.webp', size=(202,480))
    img2 = load_image('./frame_00256h.webp', size=(202,480))
    img1 #= img2
    
    #plt.imshow(img1.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    #plt.imshow(img2.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    diff_psnr,var_psnr,simple_psnr, variance_mask, difference_mask = patch_difference_psnr(img1, img2, img1,upscale=1, patch_size=20) #, mask
    print("Final Patch Difference psnr:",diff_psnr)
    print("Final Patch Variance psnr:",var_psnr)
    
    # Visualize (single image)
    import matplotlib.pyplot as plt
    plt.imshow(variance_mask[0,0].detach().cpu(), cmap='magma')
    plt.colorbar()
    plt.title("Variance Mask")
    plt.show()
    
    plt.imshow(difference_mask[0,0].detach().cpu(), cmap='magma')
    plt.colorbar()
    plt.title("Difference Mask")
    plt.show()
    
    # Normalize img2 for display (assuming it's a torch tensor [B, C, H, W])
    img2_np = img2[0].detach().cpu().permute(1, 2, 0).numpy()
    img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min())

    # Convert mask to numpy
    mask_np = difference_mask[0, 0].detach().cpu().numpy()
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())  # normalize 0–1
    
    # Overlay
    plt.figure(figsize=(10, 5))
    plt.imshow(img2_np, interpolation='nearest')
    plt.imshow(mask_np, cmap='magma', alpha=0.5)  # overlay with transparency
    plt.title("Difference Mask Overlay (α=0.5)")
    plt.axis("off")
    plt.show()
    
    mask_np = variance_mask[0, 0].detach().cpu().numpy()
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())  # normalize 0–1

    # Overlay
    plt.figure(figsize=(10, 5))
    plt.imshow(img2_np, interpolation='nearest')
    plt.imshow(mask_np, cmap='magma', alpha=0.5)  # overlay with transparency
    plt.title("Variance Mask Overlay (α=0.5)")
    plt.axis("off")
    plt.show()






test_patch_difference_psnr()
