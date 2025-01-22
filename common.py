import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn as nn
import math

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
        
def compute_pairwise_distances(tensor):
    """
    Computes pairwise Euclidean distances for each pair of agents at each of the 12 moments,
    resulting in a tensor of shape [N, N, 12, 2].
    """
    N = tensor.shape[0]

    # Expanding the tensor to calculate pairwise distances
    tensor_expanded = tensor.unsqueeze(1).repeat(1, N, 1, 1)
    tensor_tiled = tensor.unsqueeze(0).repeat(N, 1, 1, 1)

    # Calculating Euclidean distances
    distances = torch.norm(tensor_expanded - tensor_tiled, dim=3, keepdim=True)

    return distances

def normalize_distances(distances):
    max_distance = distances.max()
    min_distance = distances.min()
    if max_distance == min_distance:
        probabilities = torch.ones_like(distances) / distances.size(0)
    else:
        normalized_distances = (distances - min_distance) / (max_distance - min_distance)
        probabilities = 1 - normalized_distances  # Inverse probabilities
    return probabilities

def generate_adjacency_matrix_01(points):
    """
    Generates a 0-1 adjacency matrix based on the normalized probabilities.
    """
    N = points.shape[0]

    # Calculate the minimum distance at each moment
    distances = compute_pairwise_distances(points)
    min_distances = distances.min(dim=2).values.squeeze(-1)

    # Normalize distances to probabilities
    probabilities = normalize_distances(min_distances)

    # Generating adjacency matrix using binomial sampling
    adjacency_matrix = torch.bernoulli(probabilities)
    return adjacency_matrix

# def reparameterize_gaussian(mean, logvar):
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn(std.size()).to(mean)
#     return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)




class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class ConcatTransformerLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatTransformerLinear, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=8)
        #self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # x: (B*12*2)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self.encoder_layer(x) * gate + bias
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

# def lr_func(epoch):
#     if epoch <= start_epoch:
#         return 1.0
#     elif epoch <= end_epoch:
#         total = end_epoch - start_epoch
#         delta = epoch - start_epoch
#         frac = delta / total
#         return (1-frac) * 1.0 + frac * (end_lr / start_lr)
#     else:
#         return end_lr / start_lr
