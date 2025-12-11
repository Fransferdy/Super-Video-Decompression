
from CannyEdgeDetectorModel import DifferentiableCanny
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a smooth L1 variant)."""
    def __init__(self, eps=1e-3, alpha=45):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, pred, target):
        diff = (pred - target) * self.alpha
        loss = torch.sqrt(diff * diff + self.eps * self.eps) 
        return loss.mean()
    
def dice_Loss(edges_pred,edges_gt, eps=1e-6,alpha=190):
    P = edges_pred
    G = edges_gt

    intersection = (P * G).sum()
    #union = P.sum() + G.sum()

    #dice = (2 * intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (P.pow(2).sum() + G.pow(2).sum() + eps) 
    
    return  (1 - dice) * alpha

def soft_mask(edges_pred, edges_gt, threshold=0.5, sharpness=10.0):
    # Sigmoid acts like a "soft > threshold"
    pred_mask = torch.sigmoid(sharpness * (edges_pred - threshold))
    gt_mask   = torch.sigmoid(sharpness * (edges_gt   - threshold))
    # Soft "OR": 1 - (1-a)(1-b)
    return 1 - (1 - pred_mask) * (1 - gt_mask)

def masked_loss(pred, target, mask, base_loss=torch.nn.MSELoss(reduction='mean')):
    """
    Compute loss only on masked pixels by extracting them.
    
    pred:   [B, C, H, W]
    target: [B, C, H, W]
    mask:   [B, 1 or C, H, W] or [B, H, W]
    """
    # Ensure mask has same shape as pred
    if mask.dim() == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)  # [B,1,H,W]
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand_as(pred)  # broadcast to all channels

    # Select only masked pixels -> shape [N_masked, C]
    pred_masked   = pred[mask > 0.5]
    target_masked = target[mask > 0.5]
    #print("pred_masked shape:",pred_masked.shape)
    #print("pred shape:",pred.shape)

    if pred_masked.numel() == 0:
        # Avoid nan if mask is empty
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Apply base loss only on selected pixels
    return base_loss(pred_masked, target_masked)

class CannyEdgeLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, low_threshold=0.09, high_threshold=0.26, mask_threshold=0.5, loss_type="charbonnier", alpha=0.25, sharpness=2):
        super().__init__()
        self.alpha = alpha
        self.sharpness = sharpness
        self.canny = DifferentiableCanny(kernel_size, sigma, low_threshold, high_threshold)
        self.mask_threshold = mask_threshold
        if loss_type == "l1":
            self.criterion = F.l1_loss
        elif loss_type == "mse":
            self.criterion = F.mse_loss
        elif loss_type == "dice":
            self.criterion = dice_Loss
        elif loss_type == "charbonnier":
            self.criterion = CharbonnierLoss()
        else:
            raise ValueError(f"Unsupported loss_type {loss_type}, use 'l1' or 'mse'.")

    def forward(self, pred, target):
        """
        pred:   [B,3,H,W] generated RGB image in [0,1]
        target: [B,3,H,W] target RGB image in [0,1]
        """
        # Compute differentiable canny edges
        edges_pred = self.canny(pred)    # [B,3,H,W]
        edges_gt   = self.canny(target)  # [B,3,H,W]

        # Create relevance mask (OR between edges)
        mask = ((edges_pred > self.mask_threshold) | (edges_gt > self.mask_threshold)).float()  # [B,3,H,W]
        #mask = soft_mask(edges_pred, edges_gt, threshold=self.mask_threshold, sharpness=self.sharpness)
        # Difference
        #loss = self.criterion(edges_pred, edges_gt)
        diceLoss = dice_Loss(edges_pred, edges_gt) #masked_loss(edges_pred, edges_gt, mask, dice_Loss ) 
        imgLoss = self.criterion(pred, target) #masked_loss(pred, target, mask,self.criterion)
        
        maskedImageLoss = masked_loss(pred, target, mask,self.criterion)
        
        print("Dice Loss:",diceLoss)
        print("Masked Dice Loss:",masked_loss(edges_pred, edges_gt, mask, dice_Loss ) )
        print("Image Loss:",imgLoss)
        print("Masked Image Loss:",masked_loss(pred, target, mask,self.criterion))
        
        loss = maskedImageLoss + diceLoss

        return loss, mask, diceLoss, imgLoss
    
    
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

def test_canny_loss():
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
    canny = DifferentiableCanny(5, 1, 0.09, 0.26)
    pred = canny(img1)
    #plt.imshow(pred.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    pred = canny(img2)
    #plt.imshow(pred.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    
    # Instantiate loss
    loss_fn = CannyEdgeLoss(loss_type="charbonnier")

    # Forward pass
    loss, mask, diceLoss, imgLoss = loss_fn(img1, img2)
    print("Final Edge Loss:",loss)

    # Recompute edges + mask just for visualization
    edges_pred = loss_fn.canny(img1)
    edges_gt   = loss_fn.canny(img2)
    #mask = ((edges_pred > 0.5) | (edges_gt > 0.5)).float()
    #print("ImgLoss value:", imgLoss.item())
    #print("EdgeLoss value:", diceLoss.item())
    #print("Loss value:", loss.item())
    #print("Mask shape:", mask.shape)

    # Convert mask to numpy for plotting (take first batch, first channel)
    mask_np = mask[0,0].detach().cpu().numpy()

    #plt.imshow(mask_np, cmap="gray")
    #plt.title("Canny OR Mask")
    #plt.axis("off")
    #plt.show()
    
    
test_canny_loss()

'''
criterion = FractionBoostedCharbonnierLoss(eps=1e-3, boost_factor=3.0)

pred   = torch.tensor([1])  # predictions
target = torch.zeros_like(pred)                   # ground truth

loss = criterion(pred, target)
print("Loss:", loss.item())
'''