import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os
import argparse

from math import log10

class compact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
    ----
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.

    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=1,
        act_type="prelu",
        **kwargs,
    ):
        super(compact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out

def compute_channel_importance(model: nn.Module, dataloader, device='cuda'):
    """
    Compute activation-based channel importance for all conv layers.
    Returns a dict mapping layer names to importance scores per output channel.
    """
    model.eval()
    model.to(device)

    activations = defaultdict(list)

    # Hook to capture conv outputs
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(
                layer.register_forward_hook(
                    lambda m, inp, out, n=name: activations[n].append(out.detach().abs().mean(dim=(0,2,3)))
                )
            )

    with torch.no_grad():
        for batch in dataloader:
            # Assume batch is [B,C,H,W], move to device
            batch = batch.to(device)
            _ = model(batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate importance over batches
    importance_dict = {}
    for name, acts in activations.items():
        acts_tensor = torch.stack(acts)  # [num_batches, out_channels]
        importance_dict[name] = acts_tensor.mean(dim=0)  # mean over batches

    return importance_dict


@torch.no_grad()
def make_identity_conv(conv: nn.Conv2d):
    """Set a Conv2d to act like an identity mapping (pass-through)."""
    conv.weight.zero_()
    cmin = min(conv.out_channels, conv.in_channels)
    center = conv.kernel_size[0] // 2
    for i in range(cmin):
        conv.weight[i, i, center, center] = 1.0
    if conv.bias is not None:
        conv.bias.zero_()

@torch.no_grad()
def init_identity_and_transfer(new_model: nn.Module, small_model: nn.Module):
    """
    Initializes all convs in new_model to identity and copies matching weights
    from small_model where shapes match.
    
    Works for 'compact' architecture.
    """
    # Get all Conv2d layers in order
    convs_new = [m for m in new_model.body if isinstance(m, nn.Conv2d)]
    convs_small = [m for m in small_model.body if isinstance(m, nn.Conv2d)]

    print(f"Initializing {len(convs_new)} convs in new model...")
    print(f"Transferring from small model with {len(convs_small)} convs")

    for i, conv_new in enumerate(convs_new):
        make_identity_conv(conv_new)
        if i < len(convs_small):
            conv_small = convs_small[i]

            Wn, Ws = conv_new.weight, conv_small.weight

            # Check if shapes match
            if Wn.shape == Ws.shape:
                conv_new.weight.copy_(Ws)
                if conv_new.bias is not None and conv_small.bias is not None:
                    conv_new.bias.copy_(conv_small.bias)
                print(f"  ✓ Copied layer {i} (same shape {Ws.shape})")
            else:
                # Partial copy for mismatched shapes
                oc = min(Wn.shape[0], Ws.shape[0])
                ic = min(Wn.shape[1], Ws.shape[1])
                conv_new.weight[:oc, :ic, :, :].copy_(Ws[:oc, :ic, :, :])
                if conv_new.bias is not None and conv_small.bias is not None:
                    conv_new.bias[:oc].copy_(conv_small.bias[:oc])
                print(f"  ~ Partially copied layer {i}: new {Wn.shape}, small {Ws.shape}")
        else:
            print(f"  - No matching layer {i} in small model (kept as identity)")

    print("✅ Identity initialization + weight transfer complete.")
    return new_model



def load_topk_channels(smaller_model: nn.Module, bigger_model: nn.Module,
                       importance_dict: dict, device='cuda'):
    """
    Load weights from bigger_model into smaller_model using top-K important channels
    based on activation importance.
    """
    small_state = smaller_model.state_dict()
    big_state = bigger_model.state_dict()

    for name, param in small_state.items():
        if name in big_state:
            big_param = big_state[name]

            # Handle Conv2d weight
            if param.ndim == 4:  # [out, in, kH, kW]
                # extract layer name for importance
                layer_name = name.rsplit('.',1)[0]  # remove .weight or .bias
                if layer_name in importance_dict:
                    importance = importance_dict[layer_name]
                    topk_idx = importance.topk(param.shape[0]).indices  # select top channels
                    print(layer_name)
                    print(topk_idx)
                    # slice weights
                    out_slices = topk_idx
                    in_slices = slice(0, param.shape[1])  # copy all input channels
                    small_state[name].copy_(big_param[out_slices, in_slices, :, :])
                    continue  # skip default copy

            # Handle Conv2d bias
            elif param.ndim == 1 and 'bias' in name:
                layer_name = name.rsplit('.',1)[0]
                if layer_name in importance_dict:
                    importance = importance_dict[layer_name]
                    topk_idx = importance.topk(param.shape[0]).indices
                    small_state[name].copy_(big_param[topk_idx])
                    continue

            # Handle PReLU
            elif isinstance(param, torch.Tensor) and param.ndim == 1:
                # slice according to num_features if present in bigger model
                small_state[name].copy_(big_param[:param.shape[0]])
                continue

            # fallback: copy if shape matches
            if param.shape == big_param.shape:
                small_state[name].copy_(big_param)

    smaller_model.load_state_dict(small_state)
    print("Top-K important channels loaded successfully.")
    
def cosine_similarity(a, b, eps=1e-8):
    """Compute cosine similarity between two 1D tensors."""
    return (a * b).sum() / (a.norm() * b.norm() + eps)

def cosine_similarity_safe(a, b, eps=1e-8):
    """Compute cosine similarity even if tensor lengths differ."""
    n = min(a.numel(), b.numel())
    a = a.flatten()[:n]
    b = b.flatten()[:n]
    return (a * b).sum() / (a.norm() * b.norm() + eps)

def build_frankenstein_model(new_model, model_small, model_big, importance_big,
                             total_features=35, small_features=24, upscale=1):
    """
    Construct intermediate model with features partially from small and big model.
    """
    model_medium = new_model

    state_small = model_small.state_dict()
    state_big = model_big.state_dict()
    state_med = model_medium.state_dict()
    sim_scores = []
    top_big_idx = []
    for name, param in state_med.items():
        if name not in state_big or name not in state_small:
            continue

        # Conv2d weights
        if param.ndim == 4:
            layer_name = name.rsplit('.',1)[0]
            print("Layer:",layer_name)

            # small and big weight tensors
            W_small = state_small[name]      # [C_small, in, kH, kW]
            W_big = state_big[name]          # [C_big, in, kH, kW]
            
            print("Layer S:",W_small.shape)
            print("Layer B:",W_big.shape)
            print("Layer M:",state_med[name].shape)
            
            #for last layer:
            if W_small.shape[0] == 3 * (upscale ** 2):
                print(f"Detected output layer: {layer_name}")

                # Input channels depend on previous feature size
                C_small_in = W_small.shape[1]

                # Copy first small input channels
                state_med[name][:, :C_small_in, :, :] = W_small

                # Copy the remaining big model input channels if needed
                remaining_in = total_features - C_small_in
                if remaining_in > 0:
                    state_med[name][:, C_small_in:total_features, :, :] = \
                        W_big[:, top_big_idx[:remaining_in], :, :]
                top_big_idx=[]
                # Skip rest of processing for this layer
                continue

            # Flatten each channel to a vector
            C_small = W_small.shape[0]
            C_big = W_big.shape[0]
            W_small_flat = W_small.view(C_small, -1)
            W_big_flat = W_big.view(C_big, -1)

            # Compute max similarity of each big channel to any small channel
            print("Layer, Big, Importance:",importance_big[layer_name])
            sim_scores = []
            for i in range(C_big):
                sims = []
                for j in range(C_small):
                    sims.append(cosine_similarity_safe(W_big_flat[i], W_small_flat[j]).item())
                max_sim = max(sims)
                # score = (1 - similarity) * activation_intensity
                diff = max(1 - max_sim, 0.01)
                act_intensity = importance_big[layer_name][i].item()
                score = diff * act_intensity
                sim_scores.append(score)
            #print("Sim Scores:",sim_scores)
            sim_scores = torch.tensor(sim_scores)
            # Sort by score descending
            sorted_idx = sim_scores.argsort(descending=True)

            # --- Select channels ---
            # 2) remaining channels from big model
            remaining = total_features - C_small
            top_big_idx = sorted_idx[:remaining]
            print("Conv2d:",top_big_idx)
            # --- Copy weights ---
            # 1) Small model channels
            #state_med[name][:C_small, :, :, :] = state_small[name]
            # 2) Selected big model channels
            #state_med[name][C_small:total_features, :, :, :] = state_big[name][top_big_idx, :, :, :]
            
            print("Layer S:",W_small.shape)
            print("Layer B:",W_big.shape)
            if W_small.shape[1] == 3:  # first layer (input is RGB)
                # Copy first 24 filters from small
                state_med[name][:C_small, :, :, :] = state_small[name]
                # Add 11 selected filters from big
                state_med[name][C_small:total_features, :, :, :] = state_big[name][top_big_idx, :, :, :]
            else:
                # Normal case (deeper layer)
                # Copy small outputs & small inputs
                C_in_small = W_small.shape[1]
                
                state_med[name][:C_small, :C_in_small, :, :] = state_small[name]
                # Copy new output channels from big (based on selected top_big_idx)
                state_med[name][C_small:total_features, :C_small, :, :] = state_big[name][top_big_idx, :C_small, :, :]

                # Fill the extra input connections for all outputs (new channels as input)
                extra_in = total_features - C_small
                if extra_in > 0:
                    # copy connections from big model using same top_big_idx pattern
                    state_med[name][:, C_small:total_features, :, :] = state_big[name][top_big_idx[:state_med[name].shape[0]], top_big_idx[:extra_in], :, :]
                
        # Conv2d bias
        elif param.ndim == 1 and 'bias' in name:
            layer_name = name.rsplit('.',1)[0]
            C_small = state_small[name].shape[0]
            C_big = state_big[name].shape[0]
            print("Bias:",top_big_idx)
            # Copy biases
            state_med[name][:C_small] = state_small[name]
            state_med[name][C_small:total_features] = state_big[name][top_big_idx]
                # Handle PReLU (or other 1D per-channel parameters)
        
        elif isinstance(param, torch.Tensor) and param.ndim == 1:
            # Get layer name
            layer_name = name.rsplit('.',1)[0]

            # Copy small model channels first
            C_small = state_small[name].shape[0]
            state_med[name][:C_small] = state_small[name]
            print("Prelu:",top_big_idx)

            # Copy selected big model channels (must match conv selection)
            state_med[name][C_small:total_features] = state_big[name][top_big_idx]

            continue
        # fallback: copy if shape matches
        elif param.shape == state_small[name].shape:
            state_med[name].copy_(state_small[name])

    model_medium.load_state_dict(state_med)
    print("Intermediate model created using similarity+activation scoring.")
    return model_medium

class PairedImageFolderDataset(Dataset):
    """
    Dataset for paired images: low-quality (LQ) and ground-truth (GT).

    Args:
        lq_folder (str): Path to folder containing low-quality images.
        gt_folder (str): Path to folder containing ground-truth images.
        image_size (int, optional): Resize shorter side to this size and center crop.
    """
    def __init__(self, lq_folder, gt_folder, image_size=240, scale=1):
        self.lq_paths = sorted([
            os.path.join(lq_folder, f)
            for f in os.listdir(lq_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
        ])
        self.gt_paths = sorted([
            os.path.join(gt_folder, f)
            for f in os.listdir(gt_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
        ])
        assert len(self.lq_paths) == len(self.gt_paths), "LQ and GT folders must have the same number of images."

        self.image_size = image_size
        #self.transformX = T.ToTensor()
        #self.transformY = T.ToTensor()
        self.scale = scale
        self.scaledSize = image_size*scale
        print("LQ Size:", image_size)
        print("HQ Size:", image_size*scale)
        print("Scale:", scale)
        #if image_size is not None:
        self.transformX = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.transformY = T.Compose([
            T.Resize(self.scaledSize),
            T.CenterCrop(self.scaledSize),
            T.ToTensor()
        ])
        print("Created Special Transforms")

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq_img = Image.open(self.lq_paths[idx]).convert("RGB")
        gt_img = Image.open(self.gt_paths[idx]).convert("RGB")

        return self.transformX(lq_img), self.transformY(gt_img)
    
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, image_size=None):
        """
        Args:
            folder_path (str): Path to folder containing images
            image_size (int, optional): Resize smaller side to this value (keep aspect ratio)
        """
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff','.webp'))
        ]
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),  # ensure float32
        ])
        if image_size is not None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)


def psnr(pred, target, max_val=1.0):
    """Compute PSNR for a single pair of tensors."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * log10((max_val ** 2) / mse.item())

import random
import torchvision.utils as vutils

@torch.no_grad()
def evaluate_psnr(model, dataloader, device='cuda', max_val=1.0, save_examples=True):
    """
    Evaluates average PSNR of the model on the given dataset.
    Saves 10% of example outputs to ./transferWeightEval/.
    """
    model.eval()
    total_psnr = 0.0
    count = 0

    save_dir = "./transferWeightEval"
    if save_examples:
        os.makedirs(save_dir, exist_ok=True)

        # Pick ~10% of batches to save (or at least one)
        num_batches = len(dataloader)
        num_to_save = max(1, num_batches // 10)
        save_batches = set(random.sample(range(num_batches), num_to_save))

    for batch_idx, batch in enumerate(dataloader):
        # --- Handle different batch formats ---
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        elif isinstance(batch, dict):
            x, y = batch.get('input'), batch.get('target')
        else:
            raise ValueError(
                f"Expected batch to be (input, target) or dict, got type {type(batch)}"
            )

        x, y = x.to(device), y.to(device)

        # --- Forward pass ---
        output = model(x)

        # --- Clamp and compute PSNR ---
        output = torch.clamp(output, 0.0, max_val)
        y = torch.clamp(y, 0.0, max_val)

        total_psnr += psnr(output, y)
        count += 1

         # --- Optionally save examples ---
        if save_examples and batch_idx in save_batches:
            # Make sure tensors are 4D (B,C,H,W)
            if x.dim() == 3:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                output = output.unsqueeze(0)

            # Resize x and y to match output size if needed (for visualization)
            _, _, H_out, W_out = output.shape
            if x.shape[2:] != (H_out, W_out):
                x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
            if y.shape[2:] != (H_out, W_out):
                y = F.interpolate(y, size=(H_out, W_out), mode='bilinear', align_corners=False)

            # Denormalize to [0,1] for saving
            grid = torch.cat([x, output, y], dim=0).cpu()
            grid = torch.clamp(grid, 0.0, 1.0)

            # Save a grid image: input | output | target
            filename = os.path.join(save_dir, f"example_{batch_idx:03d}.png")
            vutils.save_image(grid, filename, nrow=3)
            print(f"Saved example to {filename}")

    avg_psnr = total_psnr / max(count, 1)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    return avg_psnr




# ================= Usage =================
# folder = "/path/to/your/images"
# dataset = ImageFolderDataset(folder, image_size=128)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

def main():
    parser = argparse.ArgumentParser(description="Transfer top-K important channels from a big model to a smaller one.")
    parser.add_argument('--big_model_path', type=str, default=None, required=False, help='Path to the pretrained big model (.pth)')
    parser.add_argument('--small_model_path', type=str, default=None, required=False, help='Path to the pretrained big model (.pth)')
    parser.add_argument('--image_folder', type=str, required=True, help='Folder with images for activation importance computation')
    parser.add_argument('--image_folder_hq', type=str, required=True, help='Folder with images for evaluation')
    parser.add_argument('--save_new_model_path', type=str, default='new_model.pth', help='Where to save the smaller model weights')
    parser.add_argument('--big_features', type=int, default=64, help='Number of features in big model')
    parser.add_argument('--small_features', type=int, default=24, help='Number of features in small model')
    parser.add_argument('--frank_features', type=int, default=35, help='Number of features in frankenstein model')
    parser.add_argument('--big_convs', type=int, default=8, help='Number of conv layers in big model')
    parser.add_argument('--small_convs', type=int, default=8, help='Number of conv layers in small model')
    parser.add_argument('--frank_convs', type=int, default=8, help='Number of conv layers in frankenstein model')
    parser.add_argument('--upscale', type=int, default=1, help='Upscale Factor')
    parser.add_argument('--image_size', type=int, default=240, help='Resize images to this size for analysis')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for dataloader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    
    # --- Build dataset/dataloader ---
    print("Building image dataset...")
    dataset = ImageFolderDataset(args.image_folder, image_size=args.image_size)
    evaluation_dataset = PairedImageFolderDataset(args.image_folder, args.image_folder_hq, image_size=args.image_size, scale=args.upscale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    evaluation_dataloader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # --- Instantiate models ---
    print(f"Loading big model with {args.big_features} and {args.big_convs} layers... upscale {args.upscale}")
    bigger_model = compact(num_feat=args.big_features, upscale=args.upscale,num_conv=args.big_convs) 
    if (args.big_model_path is not None):
        state = torch.load(args.big_model_path, map_location=args.device, weights_only=True)
        if 'params_ema' in state:
            print("Using EMA parameters from checkpoint.")
            state = state['params_ema']
        elif 'params' in state:
            print("Using normal parameters from checkpoint.")
            state = state['params']
        bigger_model.load_state_dict(state)
        bigger_model.to(args.device)
        print("Done with bigger model")
        
    if (args.small_model_path is not None):
        small_model = compact(num_feat=args.small_features, upscale=args.upscale,num_conv=args.small_convs) 
        state = torch.load(args.small_model_path, map_location=args.device, weights_only=True)
        if 'params_ema' in state:
            print("Using EMA parameters from checkpoint.")
            state = state['params_ema']
        elif 'params' in state:
            print("Using normal parameters from checkpoint.")
            state = state['params']
        small_model.load_state_dict(state)
        small_model.to(args.device)
    
    
    if (args.small_model_path is not None and args.big_model_path is not None):
        print(f"Creating new model with {args.frank_features} features and {args.frank_convs} layers...")
        new_model = compact(num_feat=args.frank_features, upscale=args.upscale,num_conv=args.frank_convs)
        new_model.to(args.device)
        # --- Compute activation importance ---
        print("Computing Channel Importance for Big Model...")
        importance_big = compute_channel_importance(bigger_model, dataloader, device=args.device)
        evaluate_psnr(bigger_model, evaluation_dataloader, device='cuda')
        print("Computing Channel Importance for Small Model...")
        #importance_small = compute_channel_importance(small_model, dataloader, device=args.device)
        evaluate_psnr(small_model, evaluation_dataloader, device='cuda')
        
        print("Building Mixed Model...")
        new_model = build_frankenstein_model(new_model, small_model, bigger_model, importance_big,
                             total_features=args.frank_features,upscale=args.upscale)
        torch.save(new_model.state_dict(), args.save_new_model_path)
        evaluate_psnr(new_model, evaluation_dataloader, device='cuda',save_examples=True)
        print(f"Frankenstein model saved to {args.save_new_model_path}")
    elif args.big_model_path is not None:
        print(f"Creating new model with {args.small_features} features and {args.small_convs} layers...")
        new_model = compact(num_feat=args.small_features, upscale=args.upscale,num_conv=args.small_convs)
        new_model.to(args.device)
        # --- Compute activation importance ---
        print("Computing channel importance from activations...")
        importance = compute_channel_importance(bigger_model, dataloader, device=args.device)
        evaluate_psnr(bigger_model, evaluation_dataloader, device='cuda')
        # --- Transfer top-K channels ---
        print("Transferring top-K important channels...")
        load_topk_channels(new_model, bigger_model, importance, device=args.device)
        evaluate_psnr(new_model, evaluation_dataloader, device='cuda',save_examples=True)
        # --- Save new small model ---
        torch.save(new_model.state_dict(), args.save_new_model_path)
        print(f"Small model saved to {args.save_new_model_path}")
    elif args.small_model_path is not None:
        print(f"Creating new model with {args.big_features} features and {args.big_convs} layers... upscale {args.upscale}" )
        #Copy small model to a bigger one with identity behavior
        new_model = compact(num_feat=args.big_features, upscale=args.upscale,num_conv=args.big_convs) 
        new_model.to(args.device)
        
        new_model = init_identity_and_transfer(new_model, small_model)
        
        evaluate_psnr(new_model, evaluation_dataloader, device='cuda',save_examples=True)
        torch.save(new_model.state_dict(), args.save_new_model_path)
        print(f"Intermediate model saved to {args.save_new_model_path}")

if __name__ == "__main__":
    print("potuto")
    main()