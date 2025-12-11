import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from neosr.losses.CombinedLossSSIMCharbMSEReduct import CombinedLossSSIMCharbMSE
from neosr.utils.registry import LOSS_REGISTRY
import os
import json

@LOSS_REGISTRY.register()
class face_aware_combined_loss(nn.Module):
    def __init__(self, alpha=38, loss_weight: float = 1.0,alpha_charb=0.3, alpha_mse=0.1, alpha_ssim=0.6, eps=1e-6, face_file=None):
        super().__init__()
        self.alpha = alpha
        self.loss_fn = CombinedLossSSIMCharbMSE(alpha_charb, alpha_mse, alpha_ssim, eps)
        self.loss_weight = loss_weight
        self.face_file_path = face_file
        self.faces_pos = None
        if os.path.exists(self.face_file_path):
            self.faces_pos = json.load(open(self.face_file_path, "r"))
            print("✅Face File exists! :"+face_file)
        else:
            print("❌Face File not found! :"+face_file)

    def extract_faces_batch(self, gt_imgs, pred_imgs,gt_file_path,top_gt,left_gt,patch_size,scale):
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
        actualPatchSize = patch_size * scale
        
        for b in range(B):
            try:
                #print("Image Shape:",gt_imgs[b].shape,pred_imgs[b].shape)
                #print(type(gt_file_path[b]), gt_file_path[b])
                #print(type(str(gt_file_path[b])), str(gt_file_path[b]))
                #print(top_gt[b])
                #print(left_gt[b])
                file_name = os.path.basename(gt_file_path[b])
                image_patch_top = int(top_gt[b])
                image_patch_left = int(left_gt[b])
                #image_patch_right = image_patch_left + patch_size
                #image_patch_bottom = image_patch_top + patch_size
                cropHeight = 40 * scale
                cropWidth = 30 * scale
                
                faces = self.faces_pos[file_name]
                for face in faces:
                    x1 = face['x1']
                    x2 = face['x2']
                    y1 = face['y1']
                    y2 = face['y2']
                    #print(x1, y1, x2, y2)
                    #x1, y1 = max(0, x1), max(0, y1)
                    #x2, y2 = min(W, x2), min(H, y2)
                    
                    # Face coordinates relative to the patch
                    rel_x1 = x1 - image_patch_left
                    rel_y1 = y1 - image_patch_top
                    rel_x2 = x2 - image_patch_left
                    rel_y2 = y2 - image_patch_top
                    
                    if rel_x1 >= actualPatchSize or rel_y1 >= actualPatchSize:
                        continue
                    if rel_x2 <= 0 or rel_y2 <= 0:
                        continue
                    
                    x1, y1 = max(0, rel_x1), max(0, rel_y1)
                    x2, y2 = min(actualPatchSize, rel_x2), min(actualPatchSize, rel_y2)

                    # Crop GT and predicted image
                    gt_crop = gt_imgs[b:b+1, :, y1:y2, x1:x2]      # shape [1, C, h, w]
                    pred_crop = pred_imgs[b:b+1, :, y1:y2, x1:x2]
                    
                    gt_crop_resized = F.interpolate(gt_crop, size=(cropHeight, cropWidth), mode='bilinear', align_corners=False)
                    pred_crop_resized = F.interpolate(pred_crop, size=(cropHeight, cropWidth), mode='bilinear', align_corners=False)

                    gt_faces.append(gt_crop_resized)
                    pred_faces.append(pred_crop_resized)
                    #face_boxes.append((b, x1, y1, x2, y2))

            except Exception as e:
                print(f"[FaceAwareLoss] Image File Not Found Face failed on image {b}: {e}")

        if len(gt_faces) == 0:
            # No faces detected, return empty tensors
            #print("No Faces Detected!")
            return torch.empty(0), torch.empty(0)#, []
        #else:
            #print("Amount of Faces Detected:",len(gt_faces))

        # Concatenate crops along batch dimension
        gt_faces = torch.cat(gt_faces, dim=0)       # [N_faces, C, h_i, w_i]
        pred_faces = torch.cat(pred_faces, dim=0)   # [N_faces, C, h_i, w_i]

        return gt_faces, pred_faces#, face_boxes

    def forward(self, pred, gt,gt_file_path=None,top_gt = None,left_gt = None,patch_size = None,scale = None):
        """
        pred, gt: (B, C, H, W)
        """
        #print("GT File Path:",gt_file_path)
        #print("Patch Top:",top_gt)
        #print("patch_size:",patch_size)
        #print("scale:",scale)
        
        if (gt_file_path is None):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        if (self.faces_pos is None):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        faces_gt,faces_pred = self.extract_faces_batch(gt, pred,gt_file_path,top_gt,left_gt,patch_size,scale)
        if (faces_gt.shape[0] == 0) or (faces_pred.shape[0] == 0):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        #print("Shapes:",faces_gt.shape, faces_pred.shape)
        loss = (self.loss_fn(faces_pred,faces_gt)) * self.alpha
        return loss * self.loss_weight