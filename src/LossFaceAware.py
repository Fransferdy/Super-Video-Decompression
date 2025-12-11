import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from retinaface import RetinaFace  # pip install retina-face
import cv2
from ImprovedBaseLoss import ImprovedBaseLoss

class FaceAwareLoss(nn.Module):
    def __init__(self, eps=1e-6, alpha=38):
        super().__init__()
        self.eps = eps
        self.detector = RetinaFace
        self.alpha = alpha
        self.loss_fn = ImprovedBaseLoss()

    def charbonnier_loss(self, pred, gt):
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps * self.eps).mean()
        return loss

    def extract_faces_batch(self, gt_imgs, pred_imgs):
        """
        Extract all detected faces from a batch of images.
        
        gt_imgs:   [B, C, H, W] tensor (ground truth)
        pred_imgs: [B, C, H, W] tensor (low-quality / predicted)
        
        Returns:
            gt_faces:   [N_faces, C, h, w] tensor
            pred_faces: [N_faces, C, h, w] tensor
            face_boxes: list of tuples (b_idx, x1, y1, x2, y2) for each face
        """
        gt_faces = []
        pred_faces = []
        #face_boxes = []

        B, C, H, W = gt_imgs.shape

        for b in range(B):
            # Convert GT image to numpy for face detection
            img_np = (gt_imgs[b].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            try:
                faces = self.detector.detect_faces(img_np)
                for face in faces.values():
                    x1, y1, x2, y2 = face["facial_area"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)

                    # Crop GT and predicted image
                    gt_crop = gt_imgs[b:b+1, :, y1:y2, x1:x2]      # shape [1, C, h, w]
                    pred_crop = pred_imgs[b:b+1, :, y1:y2, x1:x2]
                    
                    gt_crop_resized = F.interpolate(gt_crop, size=(64, 44), mode='bilinear', align_corners=False)
                    pred_crop_resized = F.interpolate(pred_crop, size=(64, 44), mode='bilinear', align_corners=False)

                    gt_faces.append(gt_crop_resized)
                    pred_faces.append(pred_crop_resized)
                    #face_boxes.append((b, x1, y1, x2, y2))

            except Exception as e:
                print(f"[FaceAwareLoss] Detection failed on image {b}: {e}")

        if len(gt_faces) == 0:
            # No faces detected, return empty tensors
            return torch.empty(0), torch.empty(0)#, []

        # Concatenate crops along batch dimension
        gt_faces = torch.cat(gt_faces, dim=0)       # [N_faces, C, h_i, w_i]
        pred_faces = torch.cat(pred_faces, dim=0)   # [N_faces, C, h_i, w_i]

        return gt_faces, pred_faces#, face_boxes

    def forward(self, pred, gt):
        """
        pred, gt: (B, C, H, W)
        """
        faces_gt,faces_pred = self.extract_faces_batch(gt, pred)
        print("Shapes:",faces_gt.shape, faces_pred.shape)
        loss = (self.loss_fn(faces_pred,faces_gt) ) * self.alpha
        return loss




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

def test_face_aware_loss(): 
    img1 = load_image('./frame_00256l.webp', size=(202,480))
    img2 = load_image('./frame_00256h.webp', size=(202,480))
    img1 #= img2
    
    # Convert back to numpy for visualization
    img_np = (img2[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Run RetinaFace detector on the ground truth image
    detector = RetinaFace
    faces = detector.detect_faces(img_np)

    # Draw bounding boxes
    img_with_boxes = img_np.copy()
    for face in faces:
        faceOb = faces[face]
        x1, y1, x2, y2 = faceOb['facial_area']
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    '''
    plt.figure(figsize=(8, 4))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Detected Faces (RetinaFace)")
    plt.axis("off")
    plt.show()
    '''
    faceAwareLoss = FaceAwareLoss(alpha=38)
    loss = faceAwareLoss(img1, img2) #, mask
    print("Final Face Aware Loss:",loss)
    '''
    # Visualize (single image)
    import matplotlib.pyplot as plt
    plt.imshow(mask[0,0].detach().cpu(), cmap='magma')
    plt.colorbar()
    plt.title("Face Mask")
    plt.show()
    '''

   
test_face_aware_loss()