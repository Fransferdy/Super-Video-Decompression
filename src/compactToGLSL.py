"""
Convert a PyTorch 'compact' model (compact.py) into mpv/placebo-compatible GLSL shaders.

Usage:
    from compact import compact
    from pytorch_compact_to_placebo import convert_compact_model_to_shaders

    # instantiate a compact model matching the weights file
    model = compact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=8, upscale=2)
    model.load_state_dict(torch.load("weights.pth", map_location="cpu"))

    convert_compact_model_to_shaders(model, out_prefix="compact_shader", hook="MAIN")

This script only supports the compact architecture defined in compact.py:
- sequential conv + PReLU layers stored in model.body (Conv2d and PReLU alternating),
- last conv producing num_out_ch * upscale * upscale channels,
- PixelShuffle(upscale),
- residual addition using nearest neighbor upsampling of input.

The generated files will be written to {out_prefix}.glsl (single concatenated shader text).
"""
import numpy as np
import torch
import torch.nn as nn
import os
from typing import List, Tuple

# ---------------------------
# Helpers adapted from shaderutils
# ---------------------------

def format_layer_name(name):
    """Make a safe bind name from a given layer name."""
    if not isinstance(name, str):
        name = str(name)
    if "." in name:
        splitname = name.split(".")[-1]
        return splitname
    else:
        return name

def pad_to_4(x, axis):
    """Pad numpy array x along axis to multiple of 4 by zero-padding."""
    length = x.shape[axis]
    pad = (-length) % 4
    if pad == 0:
        return x
    pad_width = [(0,0)] * x.ndim
    pad_width[axis] = (0, pad)
    return np.pad(x, pad_width)

def chunked(iterable, n=4):
    """Yield slices of size n from iterable length (0..len-1)."""
    for i in range(0, len(iterable), n):
        yield i, i + n

# ---------------------------
# Shader text generation primitives
# (adapted/simplified for linear compact model)
# ---------------------------

def conv_stage_to_shader(weights: np.ndarray,
                         bias: np.ndarray,
                         in_bind: str,
                         out_name: str,
                         prelu_params: np.ndarray = None,
                         hook: str = "MAIN",
                         desc: str = None,
                         max_bind_num: int = 16) -> str:
    """
    weights: numpy array shape (kh, kw, in_ch, out_ch)
    bias: None or (out_ch,)
    in_bind: name of the input texture bind (e.g. "MAIN" or "layer0", etc.)
    out_name: base name for output binds (will be suffixed by 0,1,... for 4-channel blocks)
    prelu_params: None or array of shape (out_ch,) containing 'a' for PReLU per-channel
    """
    shader = ""
    kh, kw, in_ch, out_ch = weights.shape

    # number of output stages (each produces up to 4 channels)
    for n, n_end in chunked(range(out_ch), 4):
        stage_index = n // 4
        current_bind = out_name + (str(stage_index) if stage_index > 0 else "")
        # header
        desc_str = desc or f"{out_name}-Conv"
        shader += f"//!DESC {desc_str}\n"
        shader += f"//!HOOK {hook}\n"
        # BIND input and SAVE output
        shader += f"//!BIND {in_bind}\n"
        shader += f"//!SAVE {current_bind}\n"
        shader += f"//!WIDTH {in_bind}.w\n"
        shader += f"//!HEIGHT {in_bind}.h\n"
        shader += f"//!COMPONENTS 4\n"
        shader += "\n"

        # Build the hook body
        shader += "vec4 hook() {\n"
        shader += "    vec4 result = vec4(0.0);\n"

        # For each location in the kernel
        for i in range(kh):
            for j in range(kw):
                # compute tex offset used in shader macro go_k(x_off, y_off)
                # same convention as shaderutils: offsets are floats (centered)
                tex_off_x = float(i - kh // 2 + (1 - kh % 2))
                tex_off_y = float(j - kw // 2 + (1 - kw % 2))

                # slice input channels and output channels for this stage
                # For input channels we need to split them into groups of 4 GLSL channels where
                # the macro g_k or go_k will supply a vec4 from texture fetch.
                # We'll iterate in groups of up to 4 input channels.
                in_start = 0
                while in_start < in_ch:
                    in_end = min(in_start + 4, in_ch)
                    # gather the submatrix weights: shape (in_block, out_block)
                    out_block = weights[i, j, in_start:in_end, n:n_end]  # shape (in_block, out_block_len)
                    # pad to 4x4 for mat4 * vec4 multiplication
                    mat = np.zeros((4,4), dtype=float)
                    mat[:out_block.shape[0], :out_block.shape[1]] = out_block
                    mat_tuple = tuple(mat.flatten())
                    # we will reference input macro name according to a single g_k macro per input block
                    input_block_index = (in_start//4)
                    # we'll create uniquely named macros per input block: g_{k} where k is input block index
                    # Compose GLSL line: result += mat4(...) * go_k(tex_off);
                    shader += f"    result += mat4{mat_tuple} * go_{input_block_index}({tex_off_x}, {tex_off_y});\n"
                    in_start += 4

        # Add bias if present for this particular stage
        if bias is not None:
            b = bias[n:n_end]
            # pad to 4
            b_padded = np.pad(b, (0, 4 - b.shape[0]), constant_values=0.0)
            shader += f"    result += vec4{tuple(b_padded.tolist())};\n"

        # apply PReLU if present (per-channel)
        if prelu_params is not None:
            a = prelu_params[n:n_end]
            a_padded = np.pad(a, (0, 4 - a.shape[0]), constant_values=0.0)
            # compute prelu componentwise: y = max(x,0) + a * min(x,0)
            # preserve original result
            shader += "    vec4 r = result;\n"
            shader += "    result = max(r, vec4(0.0));\n"
            shader += f"    result += vec4{tuple(a_padded.tolist())} * min(r, vec4(0.0));\n"
        # return
        shader += "    return result;\n"
        shader += "}\n\n"

    return shader

def depth_to_space_shader(prev_bind: str, out_name: str, in_ch: int, scale: int = 2, hook: str = "MAIN", desc: str = None) -> str:
    """
    Emit a Depth-to-Space / PixelShuffle shader compatible with shaderutils' behavior.
    prev_bind: input bind base name
    out_name: output bind base name
    in_ch: number of channels in previous tensor (should equal out_ch * scale * scale)
    scale: upscale integer (2 for 2x)
    """
    shader = ""
    desc_str = desc or (out_name + "-Depth-to-Space")
    # number of output 4-channel binds after depth-to-space
    out_ch = in_ch // (scale * scale)
    num_shaders = (out_ch + 3) // 4
    # Depth-to-space reads 4 "sub-binds" per output channel group
    for n in range(num_shaders):
        bind_names = []
        for k in range(4):
            kn = n * 4 + k
            if (kn * 4) < in_ch:
                bind_names.append(prev_bind + (str(kn) if kn > 0 else ""))
            else:
                bind_names.append(None)

        current_bind = out_name + (str(n) if n > 0 else "")

        shader += f"//!DESC {desc_str}\n"
        shader += f"//!HOOK {hook}\n"
        shader += f"//!SAVE {current_bind}\n"
        # Bind the previous layer chunks
        for b in bind_names:
            if b is not None:
                shader += f"//!BIND {b}\n"
        shader += f"//!COMPONENTS 4\n"
        shader += f"//!WIDTH {bind_names[0]}.w 2 *\n"
        shader += f"//!HEIGHT {bind_names[0]}.h 2 *\n"
        shader += "\n"

        shader += "vec4 hook() {\n"
        # emulate the shaderutils approach to sample 4 floats and pack them
        for bi, bname in enumerate(bind_names):
            if bname is not None:
                shader += f"    vec2 f{bi} = fract({bname}_pos * {bname}_size);\n"
                shader += f"    ivec2 i{bi} = ivec2(f{bi} * vec2({scale}.0));\n"
                shader += f"    float c{bi} = {bname}_tex((vec2(0.5) - f{bi}) * {bname}_pt + {bname}_pos)[i{bi}.y * {scale} + i{bi}.x];\n"
            else:
                if bi > 0:
                    shader += f"    float c{bi} = c{bi-1};\n"
                else:
                    shader += f"    float c{bi} = 0.0;\n"
        shader += "    return vec4(c0, c1, c2, c3);\n"
        shader += "}\n\n"
    return shader

# ---------------------------
# Main conversion routine for compact model
# ---------------------------

def convert_compact_model_to_shaders(model: nn.Module, out_prefix: str = "compact_shader", hook: str = "MAIN", outfile: str = None):
    """
    Convert a loaded 'compact' model instance (from compact.compact) into a single GLSL file.
    outfile: optional path. If None, writes to {out_prefix}.glsl
    """
    # Validate architecture shape: we expect model.body ModuleList and model.upsampler PixelShuffle
    if not hasattr(model, "body") or not isinstance(model.body, nn.ModuleList):
        raise ValueError("Model doesn't have expected .body ModuleList structure.")

    if not hasattr(model, "upsampler") or not isinstance(model.upsampler, nn.PixelShuffle):
        raise ValueError("Model doesn't have expected PixelShuffle upsampler in .upsampler")

    upscale = int(model.upscale) if hasattr(model, "upscale") else model.upsampler.upscale_factor
    in_channels = int(model.num_in_ch) if hasattr(model, "num_in_ch") else 3
    out_channels = int(model.num_out_ch) if hasattr(model, "num_out_ch") else 3

    # We'll process the body sequentially. Each Conv2d is followed by an activation (PReLU).
    # We'll name binds as layer0, layer1, ... where each bind holds 4-channel group chunks (like shaderutils).
    shader_parts: List[str] = []

    # Keep track of bind name for the "current" feature map
    current_bind_base = "MAIN"  # input texture
    layer_counter = 0
    # We'll also need to produce the macro definitions for input feature fetches used by conv_stage:
    # For each input texture that represents a chunk of 4 channels we must define g_k and/or go_k macros.
    # shaderutils did this by making parse_inputs of the Keras graph. For this sequential architecture,
    # we know the pattern: the input MAIN supplies the initial g_0 macros for the first conv's inputs.
    # We'll therefore create a small header that defines how many input g_k macros are available at each stage.
    # However the shader engine expects the macro definitions to be provided as comments like "g_0" via //!BIND lines.
    # The shader code below will reference go_0(x_off, y_off) and g_0; the mpv/placebo preprocessor will resolve them
    # when the bindings are provided. For that to work we must emit the relevant BIND lines for each shader stage,
    # which is already done in conv_stage_to_shader (it binds the input bind name).

    # Walk through model.body: detect sequence Conv2d -> (PReLU) repeated. Last Conv before PixelShuffle is final_conv.
    body = list(model.body)
    i = 0
    conv_index = 0
    prelu_params_per_conv = {}  # mapping conv index -> prelu weight array for that conv's output channels

    # We'll extract convs into a list to process
    conv_layers: List[Tuple[nn.Conv2d, np.ndarray, np.ndarray]] = []  # (conv_module, prelu_weights or None)
    while i < len(body):
        module = body[i]
        if isinstance(module, nn.Conv2d):
            conv_mod = module
            # see if next module is PReLU and get its parameters (default per-channel)
            prelu_weights = None
            if (i + 1) < len(body) and isinstance(body[i+1], nn.PReLU):
                prelu_mod = body[i+1]
                # prelu_mod.weight: shape (num_parameters,) where num_parameters==num_feat expected
                prelu_weights = prelu_mod.weight.detach().cpu().numpy().astype(float)
            conv_layers.append((conv_mod, prelu_weights))
            i += 1
        else:
            i += 1

    # The last conv is the one that outputs out_channels * upscale * upscale
    if len(conv_layers) == 0:
        raise ValueError("No Conv2d layers found in model.body")

    # We'll iterate conv layers and create shaders for each.
    prev_bind = "MAIN"
    for idx, (conv_mod, prelu_weights) in enumerate(conv_layers):
        # Extract weights/bias from conv_mod
        w = conv_mod.weight.detach().cpu().numpy()  # shape (out_ch, in_ch, kh, kw)
        b = conv_mod.bias.detach().cpu().numpy() if (conv_mod.bias is not None) else None
        # reorder to (kh, kw, in_ch, out_ch)
        w_reordered = np.transpose(w, (2, 3, 1, 0)).astype(float)
        kh, kw, in_ch, out_ch = w_reordered.shape

        # If this is the final conv (produces channels for pixelshuffle), we'll mark its out_name accordingly
        is_final_conv = False
        next_module = None
        # detect if the following top-level model has an upsampler -> final conv
        # We assume the very last conv in conv_layers is final conv
        if idx == (len(conv_layers) - 1):
            is_final_conv = True

        # output bind base name
        out_bind_base = f"layer{idx}"

        # emit shader for this conv
        shader_parts.append(conv_stage_to_shader(w_reordered, b, prev_bind, out_bind_base, prelu_params=prelu_weights, hook=hook, desc=f"conv_{idx}"))
        prev_bind = out_bind_base  # next layer uses this as input

    # After convs, we have the PixelShuffle (depth-to-space). The final conv produced out_channels * upscale * upscale channels.
    final_conv_out_ch = conv_layers[-1][0].out_channels
    # the depth-to-space output channels:
    d2s_out_channels = final_conv_out_ch // (upscale * upscale)
    # name of the depth-to-space output binds: "pixel" (we'll use pixel0, pixel1, ...)
    pixel_bind_base = "pixel"
    shader_parts.append(depth_to_space_shader(prev_bind, pixel_bind_base, final_conv_out_ch, scale=upscale, hook=hook, desc="pixelshuffle"))

    # Finally, the model adds the nearest upsampled base to the pixel-shuffled result.
    # We generate a small Add shader that reads the upsampled input (we'll refer to input as MAIN)
    # shaderutils used an "add" stage that binds previous layer names; we'll replicate a simple add shader:
    # For each 4-channel chunk of output, emit a shader that returns pixelN_tex(pixelN_pos) + upsampled MAIN value at this position.
    # The mpv/placebo engine will handle the coordinates because pixelshader saved width/height as multiplied in depth_to_space stage.
    # We'll create an "add" shader per 4-out-chunk.
    # Determine how many output 4-channel chunks there are:
    num_output_shaders = (d2s_out_channels + 3) // 4
    for n in range(num_output_shaders):
        pixel_bind = pixel_bind_base + (str(n) if n > 0 else "")
        out_bind = f"OUT{n}"  # naming for saved result
        shader_parts.append(f"//!DESC final-add\n")
        shader_parts.append(f"//!HOOK {hook}\n")
        shader_parts.append(f"//!BIND {pixel_bind}\n")
        shader_parts.append(f"//!BIND MAIN\n")  # bind MAIN so we can use the upsampled base texture
        shader_parts.append(f"//!SAVE {out_bind}\n")
        shader_parts.append(f"//!COMPONENTS 4\n")
        shader_parts.append(f"//!WIDTH {pixel_bind}.w\n")
        shader_parts.append(f"//!HEIGHT {pixel_bind}.h\n")
        shader_parts.append("\n")
        shader_parts.append("vec4 hook() {\n")
        # sample pixel chunk and nearest-upsampled base: we assume mpv will provide MAIN at the proper coords;
        # replicate shaderutils behavior: base = MAIN_tex(MAIN_pos) but sampled at the final resolution due to //!WIDTH/HEIGHT in depth_to_space.
        shader_parts.append(f"    vec4 a = {pixel_bind}_tex({pixel_bind}_pos);\n")
        shader_parts.append(f"    vec4 b = MAIN_tex(MAIN_pos);\n")
        shader_parts.append("    return a + b;\n")
        shader_parts.append("}\n\n")

    # Concatenate and write to file
    out_file = outfile or (out_prefix + ".glsl")
    with open(out_file, "w") as f:
        f.write("// Generated by pytorch_compact_to_placebo.py\n")
        f.write("// Model: compact (PyTorch) -> mpv/placebo GLSL shaders\n\n")
        f.write("\n".join(shader_parts))

    print(f"Wrote shaders to {out_file}")
    return out_file

# ---------------------------
# CLI convenience
# ---------------------------

if __name__ == "__main__":
    import argparse
    from compact import compact as CompactModel

    parser = argparse.ArgumentParser(description="Convert compact PyTorch model to placebo GLSL shaders")
    parser.add_argument("--weights", required=True, help="Path to PyTorch state_dict (.pth/.pt) or a script-module that can be loaded")
    parser.add_argument("--out", default="compact_shader.glsl", help="Output GLSL file")
    parser.add_argument("--num_feat", type=int, default=64, help="num_feat used when constructing compact")
    parser.add_argument("--num_conv", type=int, default=8, help="num_conv used in compact")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor (1/2/4)")
    args = parser.parse_args()

    # instantiate
    model = CompactModel(num_in_ch=3, num_out_ch=3, num_feat=args.num_feat, num_conv=args.num_conv, upscale=args.upscale)
    state = torch.load(args.weights, map_location="cpu")
    if 'params_ema' in state:
        print("Using EMA parameters from checkpoint.")
        state = state['params_ema']
    elif 'params' in state:
        print("Using normal parameters from checkpoint.")
        state = state['params']
    model.load_state_dict(state)
    model.to("cpu")
    
    '''
    # Accept either direct state_dict or a model file containing 'state_dict'
    if isinstance(state, dict) and any(k.startswith("module.") or k.startswith("body") for k in state.keys()):
        # assume it's a state_dict
        try:
            model.load_state_dict(state)
        except Exception as e:
            # maybe the saved file wraps the state dict
            if "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                raise
    else:
        # might be a scripted module
        model = state
    ''' 

    model.eval()
    convert_compact_model_to_shaders(model, out_prefix=os.path.splitext(args.out)[0], hook="MAIN", outfile=args.out)
