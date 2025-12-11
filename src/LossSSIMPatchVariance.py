import torch
import torch.nn.functional as F
from ImprovedBaseLoss import ImprovedBaseLoss

'''
Loss, that goes from patch to patch of the image(supposed 32x32 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''
completeLoss =  ImprovedBaseLoss()
def patch_variance_loss(pred, target, patch_size=20, alpha=10):
    B, C, H, W = target.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # Unfold directly gives [B, C*patch_size*patch_size, num_patches]
    pred_patches = unfold(pred)   # [B, C*P*P, N]
    target_patches = unfold(target)

    num_patches = pred_patches.shape[-1]
    P = patch_size

    # Compute variance per patch (over C*P*P dimension)
    target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]

    # Reshape for loss computation: [B*N, C, P, P]
    pred_patches_flat = pred_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)
    target_patches_flat = target_patches.transpose(1, 2).reshape(B * num_patches, C, P, P)

    # Combined loss per patch
    patch_loss = completeLoss(pred_patches_flat, target_patches_flat).mean(dim=(1, 2, 3))
    patch_loss = patch_loss.view(B, num_patches)

    # Normalize variance and weight
    var_norm = target_var / (target_var.mean(dim=1, keepdim=True) + 1e-8)
    weights = 1.0 + var_norm

    return (weights * patch_loss).mean() * alpha
    '''
    B, C, H, W = target.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Flatten patches: [B, C*patch_size*patch_size, num_patches]
    pred_patches = unfold(pred)
    target_patches = unfold(target)
    num_patches = pred_patches.shape[-1]
    
    pred_patches_img = pred_patches.transpose(1, 2).reshape(B, num_patches, C, patch_size, patch_size)
    target_patches_img = target_patches.transpose(1, 2).reshape(B, num_patches, C, patch_size, patch_size)

    pred_patches_flat = pred_patches_img.reshape(B * num_patches, C, patch_size, patch_size)
    target_patches_flat = target_patches_img.reshape(B * num_patches, C, patch_size, patch_size)

    # Compute variance per patch (use unbiased=False for stability)
    target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]
    print("target_var:", target_var.shape)
    #compute loss
    patch_loss = completeLoss(pred_patches_flat, target_patches_flat).mean(dim=(1, 2, 3))
    patch_loss = patch_loss.view(B, num_patches)
    print("patch_loss:", patch_loss.shape)
    # Normalize variance (avoid division by zero)
    var_norm = target_var / (target_var.mean(dim=1, keepdim=True) + 1e-8)

    # Compute weighted loss
    weights = 1.0 + var_norm
    loss = (weights * patch_loss).mean()
    '''
    '''
    # Create a tensor filled with the variance values for each patch
    var_expanded = torch.repeat_interleave(
        target_var.unsqueeze(1), patch_size * patch_size, dim=1
    )

    # Fold back into image shape
    variance_mask = fold(var_expanded) / (patch_size * patch_size)
    # Normalize to [0,1] for visualization
    variance_mask = (variance_mask - variance_mask.min()) / (variance_mask.max() - variance_mask.min() + 1e-8)
    '''

    #return loss * alpha #, variance_mask


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Assume DifferentiableCanny and CannyEdgeLoss are already defined above
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

def test_patch_variance_loss():
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
    img1 = torch.cat((img1, img1), dim=0)
    img2 = torch.cat((img2, img2), dim=0)
    
    #plt.imshow(img1.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    #plt.imshow(img2.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    loss = patch_variance_loss(img1, img2, patch_size=20, alpha=10) #, mask
    print("Final Patch Variance Loss:",loss)
    '''
    # Visualize (single image)
    import matplotlib.pyplot as plt
    plt.imshow(mask[0,0].detach().cpu(), cmap='magma')
    plt.colorbar()
    plt.title("Variance Mask")
    plt.show()
    '''

   
test_patch_variance_loss()
