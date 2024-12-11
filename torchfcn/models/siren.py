import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List

def exists(val):
    return val is not None

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.0,
        c = 6.0,
        is_first = False,
        use_bias = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out
    
def calculate_shape(layer, h_in, w_in):  
    if isinstance(layer, nn.Conv2d):
        h_out = math.floor((h_in + 2*layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0]-1) - 1)/layer.stride[0] + 1)
        w_out = math.floor((w_in + 2*layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1]-1) - 1)/layer.stride[1] + 1)
    elif isinstance(layer, nn.MaxPool2d):
        h_out = math.floor((h_in + 2*layer.padding - layer.dilation*(layer.kernel_size-1))/layer.stride + 1)
        w_out = math.floor((w_in + 2*layer.padding - layer.dilation*(layer.kernel_size-1))/layer.stride + 1)
    return h_out, w_out

class SirenFCN(nn.Module):
    def __init__(self, fcn, h_in, w_in, filters: Optional[List] =None):
        super().__init__()

        self.fcn = fcn
        self.h_in = h_in
        self.w_in = w_in
        self.filters = filters

        for k, module in dict(self.fcn.named_modules()).items():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
                h_out, w_out = calculate_shape(module, h_in, w_in)
                h_in = h_out
                w_in = w_out
            elif isinstance(module, nn.ReLU) and self.apply_filter(k):
                siren_module = Siren(w_in, w_in)
                setattr(self.fcn, k, siren_module)

    def apply_filter(self, model_layer):
        # if there are no filters always return true
        if self.filters is None:
            return True
        
        if model_layer in self.filters:
            return True
        
        return False

    def forward(self, x):
        x = self.fcn(x)
        return x
