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
    B, S, D,
    eps,
    HALF_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    # Vectorized contiguous load of entire row
    d_idx = tl.arange(0, BLOCK_D)
    x_flat = tl.load(x_row_ptr + d_idx)
    w_flat = tl.load(W_ptr + d_idx)

    x_f = x_flat.to(tl.float32)
    sumsq = tl.sum(x_f * x_f)
    mean = sumsq / D
    rstd = 1.0 / tl.sqrt(mean + eps)

    w_f = w_flat.to(tl.float32)
    normed = x_f * rstd * w_f  # (BLOCK_D,)

    # Build 2D head/half indices
    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    low_offs = head_idx * (2 * HALF_C) + half_idx
    high_offs = head_idx * (2 * HALF_C) + HALF_C + half_idx

    # Gather low/high via offsets directly from the flat normed vector
    # Use a 2D reshape and index via flatten-back trick: pull via re-load using flat index pattern
    # Since `normed` is a 1D triton tensor of size BLOCK_D, use tl.reshape to (NUM_HEADS, 2*HALF)
    normed_2d = tl.reshape(normed, (NUM_HEADS_C, 2 * HALF_C))

    # Build a (NUM_HEADS, HALF) gather via arithmetic on 2D reshape isn't slice-supported,
    # so instead recompute low/high directly from x_f and w_f using offset arithmetic.
    x_f_2d = tl.reshape(x_f, (NUM_HEADS_C, 2 * HALF_C))
    w_f_2d = tl.reshape(w_f, (NUM_HEADS_C, 2 * HALF_C))

    # Use mask-based selection: build a mask for "low" lanes (half_id < HALF_C in last dim)
    lane = tl.arange(0, 2 * HALF_C)[None, :]
    low_mask = lane < HALF_C

    # Re-load directly from global with low/high offsets for clean (NUM_HEADS, HALF) tensors
    x_low = tl.load(x_row_ptr + low_offs).to(tl.float32)
    x_high = tl.load(x_row_ptr + high_offs).to(tl.float32)
    w_low = tl.load(W_ptr + low_offs).to(tl.float32)
    w_high = tl.load(W_ptr + high_offs).to(tl.float32)

    n_low = x_low * rstd * w_low
    n_high = x_high * rstd * w_high

    cos = tl.load(COS_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]
    sin = tl.load(SIN_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]

    out_low = n_low * cos - n_high * sin
    out_high = n_low * sin + n_high * cos

    tl.store(y_row_ptr + low_offs, out_low.to(tl.float16))
    tl.store(y_row_ptr + high_offs, out_high.to(tl.float16))


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

        total = B * S
        grid = (total,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, cos, sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            cos.stride(0),
            B, S, D,
            self.eps,
            HALF_C=self.head_dim // 2,
            NUM_HEADS_C=self.num_heads,
            BLOCK_D=D,
            num_warps=8,
            num_stages=2,
        )
        return y