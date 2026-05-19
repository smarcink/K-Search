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
    X_ptr, W_ptr, COS_ptr, SIN_ptr, Y_ptr,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    stride_cs,
    S, D, HD, HALF,
    eps,
    BLOCK_D: tl.constexpr,
    HALF_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_x = X_ptr + b * stride_xb + s * stride_xs
    row_y = Y_ptr + b * stride_yb + s * stride_ys

    # --- Load x and weight once ---
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm ---
    sumsq = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(sumsq / D + eps)
    normed = x * rstd * w  # fp32, length BLOCK_D

    # --- Compute partner via reshape trick ---
    # Reshape normed (BLOCK_D,) -> (H, HD). We can build a 2D view via offsets.
    # Use tl.reshape if D == BLOCK_D, else fall back via gather.
    # Get the "partner" value for RoPE: swap halves within each head.
    in_head = offs % HD
    is_second = in_head >= HALF
    partner_offs = tl.where(is_second, offs - HALF, offs + HALF)

    # Gather partner from `normed` via shared memory: use tl.load on row_x for partner
    # but better: rebuild partner from `normed` using tl.load on a scratch? 
    # We avoid that — recompute partner from x and w loaded with partner_offs.
    # The kernel already does this; we keep loads but mark them as cached.
    x_p = tl.load(row_x + partner_offs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    w_p = tl.load(W_ptr + partner_offs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    normed_p = x_p * rstd * w_p

    # --- cos/sin gather: index in [0, HALF) ---
    i_idx = tl.where(is_second, in_head - HALF, in_head)
    cos_e = tl.load(COS_ptr + s * stride_cs + i_idx, mask=mask, other=0.0).to(tl.float32)
    sin_e = tl.load(SIN_ptr + s * stride_cs + i_idx, mask=mask, other=0.0).to(tl.float32)

    # is_second=False: out = normed*cos - partner*sin
    # is_second=True:  out = partner*sin + normed*cos
    # Combine: sign on the sin term = -1 if first half, +1 if second half
    sin_sign = tl.where(is_second, sin_e, -sin_e)
    out = normed * cos_e + normed_p * sin_sign

    tl.store(row_y + offs, out.to(tl.float16), mask=mask)


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
        HD = self.head_dim
        HALF = HD // 2

        x = x.contiguous()
        y = torch.empty_like(x)

        BLOCK_D = triton.next_power_of_2(D)
        HALF_C = triton.next_power_of_2(HALF)

        grid = (B * S,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self.rope_cos.stride(0),
            S, D, HD, HALF,
            self.eps,
            BLOCK_D=BLOCK_D,
            HALF_C=HALF_C,
            num_warps=8,
            num_stages=3,
        )
        return y


def get_inputs():
    device = get_device()
    x = torch.randn(8, 2048, 4096, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    return []