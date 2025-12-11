import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableCanny(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Create Gaussian kernel
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)

        # Sobel kernels for gradient computation
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def create_gaussian_kernel(self, kernel_size, sigma):
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # shape [1,1,k,k]

    def forward(self, x):
        """
        x: [B,1,H,W] grayscale image in [0,1]
        returns: [B,1,H,W] soft edge map
        """
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB image with 3 channels"

        # Convert to grayscale
        x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # Gaussian smoothing
        x = F.conv2d(x, self.gaussian_kernel.to(x.device), padding=self.kernel_size//2)

        # Gradient computation
        Gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        Gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        grad_magnitude = torch.sqrt(Gx**2 + Gy**2 + 1e-6)
        grad_direction = torch.atan2(Gy, Gx)

        # Soft non-maximum suppression
        grad_x = grad_magnitude * torch.cos(grad_direction)
        grad_y = grad_magnitude * torch.sin(grad_direction)
        # Shifted versions for NMS
        def shift(tensor, dx, dy):
            # Ensure dx, dy are integers
            dx, dy = int(dx), int(dy)
            return F.pad(tensor, (1,1,1,1))[:, :, 1+dy:tensor.shape[2]+1+dy, 1+dx:tensor.shape[3]+1+dx]

        N = shift(grad_magnitude, 0, -1)
        S = shift(grad_magnitude, 0, 1)
        E = shift(grad_magnitude, 1, 0)
        W = shift(grad_magnitude, -1, 0)
        NE = shift(grad_magnitude, 1, -1)
        NW = shift(grad_magnitude, -1, -1)
        SE = shift(grad_magnitude, 1, 1)
        SW = shift(grad_magnitude, -1, 1)

        # Approximate NMS using softmax along gradient direction
        def soft_nms(mag, direction):
            # Using approximate max along gradient direction
            pos1 = torch.cos(direction) > 0
            pos2 = torch.sin(direction) > 0
            neighbors = 0.25*(N+S+E+W+NE+NW+SE+SW)
            return mag * torch.sigmoid(mag - neighbors)

        nms = soft_nms(grad_magnitude, grad_direction)

        # Soft double threshold
        low_mask = torch.sigmoid(10*(nms - self.low_threshold))
        high_mask = torch.sigmoid(10*(nms - self.high_threshold))

        # Combine masks: approximate hysteresis
        edges = torch.clamp(high_mask + low_mask * (1 - high_mask), 0, 1)
        #edges_rgb = edges.repeat(1, 3, 1, 1)
        edges_rgb = edges.expand(-1, 3, -1, -1)
         
        return edges_rgb