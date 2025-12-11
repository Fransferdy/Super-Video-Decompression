import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------
#   Sobel Filter module (from previous answer)
# ---------------------------------------------
import torch.nn as nn

class SobelFilter(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ])
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ])

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=channels
        )

        with torch.no_grad():
            for c in range(channels):
                self.conv.weight[2*c, 0] = sobel_x
                self.conv.weight[2*c+1, 0] = sobel_y

        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        edges = self.conv(x)       # (B,2,H,W)
        gx = edges[:, 0:1]         # keep dims
        gy = edges[:, 1:2]
        
        mag = torch.sqrt(gx**2 + gy**2 + 1e-8)  # avoid sqrt(0)

        return mag

# ---------------------------------------------
#   Load image
# ---------------------------------------------
image_path = "frame_00256l.webp"  # <-- replace with your image file

img = Image.open(image_path)  # grayscale for simplicity
to_tensor = T.ToTensor()
x = to_tensor(img).unsqueeze(0)            # shape: (1,1,H,W)

# ---------------------------------------------
#   Apply Sobel filter
# ---------------------------------------------
sobel = SobelFilter(channels=3)
mag = sobel(x)

# ---------------------------------------------
#   Display results
# ---------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Magnitude")
plt.imshow(mag.squeeze().numpy(), cmap='gray')
plt.axis('off')

plt.show()

model = SobelFilter(channels=3)

torch.save(model.state_dict(), "sobel.pth")