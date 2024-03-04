import torch
import torch.nn as nn
from torch.nn import functional as F

class BitLinear158b(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear158b, self).__init__(in_features, out_features, bias)
        self.bit_scale = 8
        self.Q_b = 2 ** (self.bit_scale - 1)
        self.eps = 1e-12
        
    def quantize_activations(self, input_norm, abs_max_x):
        scaled_x = torch.clamp(
            input_norm * self.Q_b / (abs_max_x + self.eps), -self.Q_b + self.eps, self.Q_b - self.eps
        )
        return scaled_x
    
    def ternarize_weights(self,abs_mean_W):
        scaled_W = self.weight / (abs_mean_W + self.eps)
        quantize_weights = torch.sign(torch.clamp(scaled_W.round(), -1, 1))
        return quantize_weights

    def forward(self, input):
        input_norm = F.layer_norm(input, (self.in_features,))
        
        abs_max_x = input_norm.abs().max() #gamma
        quant_scaled_input = self.quantize_activations(input_norm,abs_max_x)

        abs_mean_W = self.weight.abs().mean() #beta
        ternarized_weights = self.ternarize_weights(abs_mean_W)

        matmal_weight = F.linear(quant_scaled_input, ternarized_weights, self.bias)

        beta_gamma = abs_mean_W * abs_max_x 
        output = matmal_weight * beta_gamma / self.Q_b
        return output
