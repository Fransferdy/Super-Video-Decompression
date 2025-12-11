import torch
import torch.nn.functional as F

'''
Loss, that goes from patch to patch of the image(supposed 32x32 patches) 
and calculate the variance of pixels ( high variance = lots of different colors, low variance = few colors dominate the whole patch), 
then each patch is multiplied by a alpha variable(such as 5 or 10 or 100), 
calculate the MSE of the patch and divide by the alpha. 
For the whole image, we take a pondered weight of all MSEs
where patches with high variance have more weight and low variance less weight
'''
def patch_variance_loss(pred, target, patch_size=20, alpha=430):
    B, C, H, W = target.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Flatten patches: [B, C*patch_size*patch_size, num_patches]
    pred_patches = unfold(pred)
    target_patches = unfold(target)

    # Compute variance per patch (use unbiased=False for stability)
    target_var = target_patches.var(dim=1, unbiased=False)  # [B, num_patches]

    # Compute MSE per patch
    diff = (pred_patches - target_patches) * alpha
    patch_mse = (diff ** 2).mean(dim=1) / alpha  # [B, num_patches]

    # Normalize variance (avoid division by zero)
    var_norm = target_var / (target_var.mean(dim=1, keepdim=True) + 1e-8)

    # Compute weighted loss
    weights = 1.0 + var_norm
    loss = (weights * patch_mse).mean()

    
    # Create a tensor filled with the variance values for each patch
    var_expanded = torch.repeat_interleave(
        target_var.unsqueeze(1), patch_size * patch_size, dim=1
    )

    # Fold back into image shape
    variance_mask = fold(var_expanded) / (patch_size * patch_size)
    # Normalize to [0,1] for visualization
    variance_mask = (variance_mask - variance_mask.min()) / (variance_mask.max() - variance_mask.min() + 1e-8)
    

    return loss , variance_mask


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
    
    #plt.imshow(img1.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    #plt.imshow(img2.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    loss, mask = patch_variance_loss(img1, img2, patch_size=20, alpha=1) #, mask
    print("Final Patch Variance Loss:",loss)
    
    # Visualize (single image)
    import matplotlib.pyplot as plt
    plt.imshow(mask[0,0].detach().cpu(), cmap='magma')
    plt.colorbar()
    plt.title("Variance Mask")
    plt.show()
    

   
test_patch_variance_loss()
