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
    S,
    D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S

    row_off = pid * D
    offs = tl.arange(0, BLOCK_D)

    # Load full row (BLOCK_D == D, power of 2 here since D=4096)
    x = tl.load(x_ptr + row_off + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * w  # fp32, kept in registers

    # RoPE: split row into [first_half_per_head, second_half_per_head]
    # Layout: for each head h, positions [h*HD .. h*HD+HALF) are x1, [h*HD+HALF .. h*HD+HD) are x2.
    # Build first/second half views by indexing.
    half_offs = tl.arange(0, BLOCK_D // 2)  # H * HALF
    head_idx = half_offs // HALF
    lane = half_offs % HALF

    x1_off = head_idx * HD + lane
    x2_off = head_idx * HD + HALF + lane

    # Use tl.where masking to gather; since normed is a single vector indexed by offs,
    # we cannot gather from registers. Reload from x_ptr instead for partners.
    x1_raw = tl.load(x_ptr + row_off + x1_off).to(tl.float32)
    w1 = tl.load(w_ptr + x1_off).to(tl.float32)
    x1 = x1_raw * rstd * w1

    x2_raw = tl.load(x_ptr + row_off + x2_off).to(tl.float32)
    w2 = tl.load(w_ptr + x2_off).to(tl.float32)
    x2 = x2_raw * rstd * w2

    # cos/sin for this position: shape (HALF,), broadcast across heads
    cs_off = s * HALF + lane
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    tl.store(out_ptr + row_off + x1_off, out1.to(tl.float16))
    tl.store(out_ptr + row_off + x2_off, out2.to(tl.float16))


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

        BLOCK_D = D  # D=4096 is power of two
        grid = (B * S,)

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, out,
            S,
            D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=2,
        )
        return out