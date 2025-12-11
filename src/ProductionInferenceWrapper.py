import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBAWrapper(nn.Module):
    def __init__(self, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        """
        x: [N, H, W, 4], dtype=torch.uint8 (canvas ImageData)
        returns: [N, 4, H, W], dtype=torch.float16
        """
        # Move channels to NCHW
        # [N, H, W, 4] -> 
        # [N, 4, H, W]
        x = x.permute(0, 3, 1, 2)

        # Drop alpha
        rgb = x[:, :3, :, :]

        # Normalize
        rgb = rgb / 255.0

        # Forward through inner model (assume index 3 output is RGB)
        out = self.model(rgb)  # [N, 3, H, W]

        # Denormalize
        out = out.clamp(0.0, 1.0) * 255.0
        ##out = out * 255.0

        # Upsample by 4x (bilinear for ONNX compatibility)
        
        
        # Add back alpha = 255
        alpha = torch.ones_like(out[:, :1, :, :]) * 255.0
        out = torch.cat([out, alpha], dim=1)
        
        #out = F.interpolate(out, scale_factor=4, mode='nearest')

        # [N, 4, H, W] -> 
        # [N, H, W, 4]
        out = out.permute(0, 2, 3, 1)
        
        
        
        return out