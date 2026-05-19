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
    S, D, NUM_HEADS, HEAD_DIM, HALF,
    eps,
    BLOCK_D: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    HALF_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # RMSNorm
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / D
    rstd = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w  # fp32

    # RoPE: within each head of size HEAD_DIM, rotate halves
    # offs lane index within head
    lane = offs % HEAD_DIM_C
    is_low = lane < HALF_C
    # idx into cos/sin table: lane if low else lane - HALF
    cs_idx = tl.where(is_low, lane, lane - HALF_C)

    cos_row_ptr = COS_ptr + s * stride_cs
    sin_row_ptr = SIN_ptr + s * stride_cs
    cos = tl.load(cos_row_ptr + cs_idx, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_row_ptr + cs_idx, mask=mask, other=0.0).to(tl.float32)

    # Need partner value: for low lanes, partner is normed at lane+HALF; for high lanes, partner is normed at lane-HALF
    # We need to get the partner from the same row. Use a second load via xor with HALF.
    # Easier: reload normed in shuffled order. But we can compute via index trick using tl.load with permuted offsets.
    # offs_partner = offs XOR HALF (within head)
    partner_offs = tl.where(is_low, offs + HALF_C, offs - HALF_C)
    x_partner = tl.load(x_row_ptr + partner_offs, mask=mask, other=0.0).to(tl.float32)
    w_partner = tl.load(W_ptr + partner_offs, mask=mask, other=0.0).to(tl.float32)
    normed_partner = x_partner * rstd * w_partner

    # For low lanes (output position = low): out = x1*cos - x2*sin, where x1=normed (low), x2=normed_partner (high)
    # For high lanes (output position = high): out = x1*sin + x2*cos, where x1=normed_partner (low), x2=normed (high)
    out_low = normed * cos - normed_partner * sin
    out_high = normed_partner * sin + normed * cos
    out = tl.where(is_low, out_low, out_high)

    tl.store(y_row_ptr + offs, out.to(tl.float16), mask=mask)


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
        y = torch.empty_like(x)

        cos = self.rope_cos
        sin = self.rope_sin

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * S,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, cos, sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            cos.stride(0),
            S, D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            BLOCK_D=BLOCK_D,
            HEAD_DIM_C=self.head_dim,
            HALF_C=self.head_dim // 2,
            num_warps=8,
            num_stages=2,
        )
        return y