import torch
import torch.nn as nn
from math import sqrt, ceil
from einops import rearrange

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

class RotaryPositionalEmbedding(nn.Module):
    cos: torch.Tensor # max_seq_len, half_d_k
    sin: torch.Tensor # max_seq_len, half_d_k

    def __init__(self, theta: float, d_k:int, max_seq_len: int, device= None):
        super().__init__()
        thetas = torch.einsum("i, t -> i t", torch.arange(max_seq_len), torch.pow(theta, -torch.arange(0, d_k, 2) / d_k))
        # create \theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d_k}}

        cos = torch.cos(thetas).to(device)
        sin = torch.sin(thetas).to(device)
        # precalculate cos and sin

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positons: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        tokcos = self.cos[token_positons] # get specific cos and sin results
        toksin = self.sin[token_positons]

        result = torch.empty_like(x)

        result[..., ::2] = x_even * tokcos - x_odd * toksin
        result[..., 1::2] = x_even * toksin + x_odd * tokcos
        # for each pair of tokens x1 and x2, apply rotation matrix to it and get rotated pair
        return result

def softmax(x: torch.Tensor, i: int):
    z = x - torch.amax(x, dim=i, keepdim=True)
    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=i, keepdim=True)
    return exp_z/sum_exp

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Q and K is (batch_size, ..., seq_len, d_k)
    # V is (batch_size, ..., seq_len, d_v)
    # Q is n long, K and V are m long
    # mask is nxm
    d_k = Q.shape[-1]

    #presoftmax = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    presoftmax = torch.einsum("... q d, ... k d -> ... q k", Q, K) / sqrt(d_k)

    if mask is not None:
        # to apply mask, compare each i,jth element in presoftmax with mask, if its false, replace with -infinity
        presoftmax = presoftmax + torch.where(mask, 0.0, float('-inf'))

    after_softmax = softmax(presoftmax, i=-1)

    return torch.matmul(after_softmax, V)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None, pos_encoder: RotaryPositionalEmbedding | None = None) -> None:
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.pos_encoder = pos_encoder
        self.device = device
        self.dtype = dtype

        self.postok_to_query = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.postok_to_key = Linear(self.d_model, self.num_heads * self.d_k, device, dtype)
        self.postok_to_value = Linear(self.d_model, self.num_heads * self.d_v, device, dtype)

        self.output_projector = Linear(self.num_heads * self.d_v, self.d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positons: torch.Tensor | None = None) -> torch.Tensor:
        *batch_dims, seq_len, d_model = x.size()

        Q = self.postok_to_query(x)
        K = self.postok_to_key(x)
        V = self.postok_to_value(x)

        # head splitting
        Qh = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        Kh = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        Vh = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)

        if self.pos_encoder is not None and token_positons is not None:
            # apply RoPE (if available) to query and key
            Qh = self.pos_encoder(Qh, token_positons)
            Kh = self.pos_encoder(Kh, token_positons)

        # mask for causal attention (do not do attention on future tokens)
        iota = torch.arange(seq_len, device=x.device)
        qi = rearrange(iota, "query -> query 1")
        kj = rearrange(iota, "k -> 1 k")
        mask = qi >= kj
        mask = mask.__getitem__((None,) * len(batch_dims) + (...,))

        # do attention
        outH = scaled_dot_product_attention(Qh, Kh, Vh, mask)

        # then put result through output MLP
        out = rearrange(outH, "... h seq_len d -> ... seq_len (h d)")
        return self.output_projector(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        self.attention = CausalMultiHeadSelfAttention(d_model, num_heads, pos_encoder=self.rope, device=device, dtype=dtype)

        self.token_positons = torch.arange(max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n1 = self.norm1(x)                
        mha = self.attention(n1, self.token_positons[: x.shape[-2]])
        add1 = x + mha

        n2 = self.norm2(add1)
        sg = self.swiglu(n2)
        return sg + add1

class TransformerLanguageModel(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, vocab_size: int, context_length: int, num_layers: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device, dtype)
        self.layers = nn.ModuleList(
                [
                    TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device, dtype)
                     for _ in range(num_layers)
                     ]
                )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens)
        for block in self.layers:
            x = block(x)
        return self.output_embedding(self.final_norm(x))
