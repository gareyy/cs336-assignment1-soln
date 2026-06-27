import torch
import torch.nn as nn
from math import sqrt, ceil

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # basically do a y = Wx
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a= -3 * sigma, b= 3* sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("... i, o i -> ... o", x, self.W)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embeds = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeds, mean=0, std=1, a= -3, b= 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # we are given a list of token IDs and we want a vector valued vector (a matrix lol) where each entry corresponds to a mapping between a token and an embedding vector
        return self.embeds[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_32 = x.to(torch.float32)

        rms = torch.sqrt(x_32.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        result = (x_32 / rms) * self.gain.to(torch.float32)
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, multiple_of: int = 64, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_ff = d_ff if d_ff else int((( int(ceil( (8.0 * d_model) / 3.0)) + multiple_of - 1) // multiple_of) * multiple_of)
        self.W1 = Linear(d_model, self.d_ff, device, dtype)
        self.W2 = Linear(self.d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, self.d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one = self.W1(x)
        silu_result = one * torch.sigmoid(one)
        two = torch.mul(silu_result, self.W3(x))
        return self.W2(two)
