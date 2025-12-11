
from CannyEdgeDetectorModel import DifferentiableCanny
import torch
import torch.nn as nn
import torch.nn.functional as F
from neosr.utils.registry import LOSS_REGISTRY


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

@LOSS_REGISTRY.register()
class canny_edge_loss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, low_threshold=0.09, high_threshold=0.26, mask_threshold=0.5, loss_type="charbonnier", alpha=0.25, sharpness=2, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
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
        #imgLoss = self.criterion(pred, target) #masked_loss(pred, target, mask,self.criterion)
        
        maskedImageLoss = masked_loss(pred, target, mask,self.criterion)
        loss = (1 - self.alpha) * maskedImageLoss + self.alpha * diceLoss
        #loss = maskedImageLoss + diceLoss

        return loss * self.loss_weight