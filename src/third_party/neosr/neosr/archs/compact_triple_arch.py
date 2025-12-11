# type: ignore
import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

from neosr.archs.compact_arch import compact

upscale, __ = net_opt()


@ARCH_REGISTRY.register()
class compact_triple(nn.Module):
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
        num_feat1=64,
        num_feat2=64,
        num_feat3=64,
        num_conv=8,
        upscale1=1,
        upscale2=2,
        upscale3=2,
        compact_path1='',
        compact_path2='',
        compact_path3='',
        act_type="prelu",
        **kwargs,
    ):
        super(compact_triple, self).__init__()
        print("Parameters:","compact_path1:",compact_path1,"compact_path2:",compact_path2,"compact_path3:",compact_path3,
              "num_in_ch:",num_in_ch,"num_out_ch:",num_out_ch,"num_feat1:",num_feat1,"num_feat2:",num_feat2,"num_feat3:",num_feat3,
              "num_conv:",num_conv,"upscale1:",upscale1,"upscale2:",upscale2,"upscale3:",upscale3,"act_type:",act_type)
        self.compact1 = compact(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat1,
            num_conv=num_conv,
            upscale=upscale1,
            act_type=act_type
        )
        if compact_path1 != '':
            state_dict = torch.load(compact_path1, map_location=torch.device("cuda"), weights_only=True)
            try:
                if "params-ema" in state_dict:
                    param_key = "params-ema"
                elif "params" in state_dict:
                    param_key = "params"
                elif "params_ema" in state_dict:
                    param_key = "params_ema"
                state_dict = state_dict[param_key]
            except:
                pass
            self.compact1.load_state_dict(state_dict)
            
        self.compact2 = compact(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat2,
            num_conv=num_conv,
            upscale=upscale2,
            act_type=act_type
        )
        if compact_path2 != '':
            state_dict = torch.load(compact_path2, map_location=torch.device("cuda"), weights_only=True)
            try:
                if "params-ema" in state_dict:
                    param_key = "params-ema"
                elif "params" in state_dict:
                    param_key = "params"
                elif "params_ema" in state_dict:
                    param_key = "params_ema"
                state_dict = state_dict[param_key]
            except:
                pass
            self.compact2.load_state_dict(state_dict)
        self.compact3 = compact(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat3,
            num_conv=num_conv,
            upscale=upscale3,
            act_type=act_type
        )
        if compact_path3 != '':
            state_dict = torch.load(compact_path3, map_location=torch.device("cuda"), weights_only=True)
            try:
                if "params-ema" in state_dict:
                    param_key = "params-ema"
                elif "params" in state_dict:
                    param_key = "params"
                elif "params_ema" in state_dict:
                    param_key = "params_ema"
                state_dict = state_dict[param_key]
            except:
                pass
            self.compact3.load_state_dict(state_dict)

    def forward(self, x):
        return self.compact3(self.compact2(self.compact1(x)))
