import torch
import math
from torch import nn, Tensor
import triton
import triton.language as tl


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def precompute_freqs(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


@triton.jit
def fused_rmsnorm_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    B, S,
    D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
):
    pid = tl.program_id(0)
    s = pid % S
    row_off = pid * D

    h_idx = tl.arange(0, H)
    l_idx = tl.arange(0, HALF)
    head_base = h_idx[:, None] * HD
    offs1 = head_base + l_idx[None, :]
    offs2 = offs1 + HALF

    x1 = tl.load(x_ptr + row_off + offs1).to(tl.float32)
    x2 = tl.load(x_ptr + row_off + offs2).to(tl.float32)
    w1 = tl.load(w_ptr + offs1).to(tl.float32)
    w2 = tl.load(w_ptr + offs2).to(tl.float32)

    ss = tl.sum(x1 * x1) + tl.sum(x2 * x2)
    rstd = 1.0 / tl.sqrt(ss / D + eps)

    n1 = x1 * rstd * w1
    n2 = x2 * rstd * w2

    cs_off = s * HALF + l_idx
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)
    cos_b = cos[None, :]
    sin_b = sin[None, :]

    out1 = n1 * cos_b - n2 * sin_b
    out2 = n1 * sin_b + n2 * cos_b

    tl.store(out_ptr + row_off + offs1, out1.to(tl.float16))
    tl.store(out_ptr + row_off + offs2, out2.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dim=4096, num_heads=32, seq_len=2048, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

        cos, sin = precompute_freqs(self.head_dim, seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.half()

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        assert D == self.dim
        out = torch.empty_like(x)

        grid = (B * S,)

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, out,
            B, S, D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            num_warps=8,
            num_stages=3,
        )
        return out