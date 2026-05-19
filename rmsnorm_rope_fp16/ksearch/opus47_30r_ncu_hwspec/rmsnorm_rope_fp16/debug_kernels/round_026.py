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
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    D_C: tl.constexpr = NUM_HEADS_C * HALF_C * 2
    offs = tl.arange(0, D_C)
    x = tl.load(x_row_ptr + offs)
    w = tl.load(W_ptr + offs)

    x_f = x.to(tl.float32)
    sumsq = tl.sum(x_f * x_f)
    rstd = 1.0 / tl.sqrt(sumsq / D + eps)

    n = (x_f * rstd) * w.to(tl.float32)

    # reshape to (NUM_HEADS, 2, HALF)
    n2 = tl.reshape(n, (NUM_HEADS_C, 2, HALF_C))
    n_low = tl.reshape(tl.sum(n2 * tl.reshape(tl.cat(tl.full((HALF_C,), 1.0, tl.float32), tl.full((HALF_C,), 0.0, tl.float32)), (1, 2, HALF_C)), axis=1), (NUM_HEADS_C, HALF_C))

    # Simpler approach: reload via gather using strided indices
    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    low_offs = head_idx * (2 * HALF_C) + half_idx
    high_offs = low_offs + HALF_C

    x_low = tl.load(x_row_ptr + low_offs)
    x_high = tl.load(x_row_ptr + high_offs)
    w_low = tl.load(W_ptr + low_offs)
    w_high = tl.load(W_ptr + high_offs)

    x_low_f = x_low.to(tl.float32)
    x_high_f = x_high.to(tl.float32)

    cos = tl.load(COS_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]
    sin = tl.load(SIN_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]

    n_low = x_low_f * rstd * w_low.to(tl.float32)
    n_high = x_high_f * rstd * w_high.to(tl.float32)

    out_low = n_low * cos - n_high * sin
    out_high = n_low * sin + n_high * cos

    tl.store(y_row_ptr + low_offs, out_low.to(tl.float16))
    tl.store(y_row_ptr + high_offs, out_high.to(tl.float16))


@triton.jit
def fused_rmsnorm_rope_kernel_v2(
    X_ptr, W_ptr, COS_ptr, SIN_ptr, Y_ptr,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    stride_cs,
    B, S, D,
    eps,
    HALF_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    D_C: tl.constexpr = NUM_HEADS_C * HALF_C * 2
    offs = tl.arange(0, D_C)

    # single contiguous load of the entire row
    x = tl.load(x_row_ptr + offs)
    w = tl.load(W_ptr + offs)

    x_f = x.to(tl.float32)
    sumsq = tl.sum(x_f * x_f)
    rstd = 1.0 / tl.sqrt(sumsq / D + eps)

    n = x_f * rstd * w.to(tl.float32)

    # split into low/high halves of each head via reshape
    n_r = tl.reshape(n, (NUM_HEADS_C, 2, HALF_C))
    # extract using masks
    pair_idx = tl.arange(0, 2)[None, :, None]  # (1,2,1)
    is_low = (pair_idx == 0).to(tl.float32)
    is_high = (pair_idx == 1).to(tl.float32)
    n_low = tl.sum(n_r * is_low, axis=1)   # (NUM_HEADS, HALF)
    n_high = tl.sum(n_r * is_high, axis=1)

    cos = tl.load(COS_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]
    sin = tl.load(SIN_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)[None, :]

    out_low = n_low * cos - n_high * sin
    out_high = n_low * sin + n_high * cos

    # store contiguously
    out = tl.reshape(
        tl.join(out_low, out_high),  # (NUM_HEADS, HALF, 2) ? - join stacks last dim
        (NUM_HEADS_C, 2 * HALF_C),
    )
    # Actually tl.join produces shape (..., 2), so we need to transpose-like
    # Fall back to two stores below; ignore this 'out'.

    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    low_offs = head_idx * (2 * HALF_C) + half_idx
    high_offs = low_offs + HALF_C

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
        fused_rmsnorm_rope_kernel_v2[grid](
            x, self.weight, cos, sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            cos.stride(0),
            B, S, D,
            self.eps,
            HALF_C=self.head_dim // 2,
            NUM_HEADS_C=self.num_heads,
            num_warps=8,
            num_stages=2,
        )
        return y