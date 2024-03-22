import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module): #quoted by LammaRMSNorm
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class BitLinear158b(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear158b, self).__init__(in_features, out_features, bias)
        self.rmsnorm = RMSNorm(in_features)
        
    def activation_norm_quant(self, x):
        scale = 127.0 / x.abs().max(dim=-1,keepdim=True).values.clamp(min=1e-5) #gamma
        y = (x * scale).round().clamp(-128,127) / scale
        return y
    
    def weight_quant(self,w):
        scale = 1.0 / w.abs().mean().clamp(min=1e-5) #beta
        u = (w * scale).round().clamp(-1,1) / scale
        return u

    def forward(self, x):
        w = self.weight
        x_norm = self.rmsnorm(x)
        x_quant = x_norm + (self.activation_norm_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        output = F.linear(x_quant,w_quant,self.bias)
        return output
