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
    S,
    inv_D,
    eps,
    HALF_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid - b * S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    low_offs = head_idx * (2 * HALF_C) + half_idx
    high_offs = low_offs + HALF_C

    x_low = tl.load(x_row_ptr + low_offs)
    x_high = tl.load(x_row_ptr + high_offs)
    xl_f = x_low.to(tl.float32)
    xh_f = x_high.to(tl.float32)

    sumsq = tl.sum(xl_f * xl_f) + tl.sum(xh_f * xh_f)
    rstd = tl.rsqrt(sumsq * inv_D + eps)

    w_low = tl.load(W_ptr + low_offs).to(tl.float32)
    w_high = tl.load(W_ptr + high_offs).to(tl.float32)

    n_low = xl_f * rstd * w_low
    n_high = xh_f * rstd * w_high

    cos = tl.load(COS_ptr + s * stride_cs + tl.arange(0, HALF_C))[None, :].to(tl.float32)
    sin = tl.load(SIN_ptr + s * stride_cs + tl.arange(0, HALF_C))[None, :].to(tl.float32)

    out_low = n_low * cos - n_high * sin
    out_high = n_low * sin + n_high * cos

    tl.store(y_row_ptr + low_offs, out_low.to(tl.float16))
    tl.store(y_row_ptr + high_offs, out_high.to(tl.float16))


@triton.jit
def fused_rmsnorm_rope_kernel_2row(
    X_ptr, W_ptr, COS_ptr, SIN_ptr, Y_ptr,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    stride_cs,
    S,
    inv_D,
    eps,
    HALF_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    low_offs = head_idx * (2 * HALF_C) + half_idx
    high_offs = low_offs + HALF_C
    half_range = tl.arange(0, HALF_C)

    w_low = tl.load(W_ptr + low_offs).to(tl.float32)
    w_high = tl.load(W_ptr + high_offs).to(tl.float32)

    for i in tl.static_range(0, ROWS_PER_PROG):
        row = row_start + i
        b = row // S
        s = row - b * S

        x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
        y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

        x_low = tl.load(x_row_ptr + low_offs)
        x_high = tl.load(x_row_ptr + high_offs)
        xl_f = x_low.to(tl.float32)
        xh_f = x_high.to(tl.float32)

        sumsq = tl.sum(xl_f * xl_f) + tl.sum(xh_f * xh_f)
        rstd = tl.rsqrt(sumsq * inv_D + eps)

        n_low = xl_f * rstd * w_low
        n_high = xh_f * rstd * w_high

        cos = tl.load(COS_ptr + s * stride_cs + half_range)[None, :].to(tl.float32)
        sin = tl.load(SIN_ptr + s * stride_cs + half_range)[None, :].to(tl.float32)

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
        self.register_buffer("rope_cos", cos.half())
        self.register_buffer("rope_sin", sin.half())

        self.half()

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        assert D == self.dim
        y = torch.empty_like(x)

        cos = self.rope_cos
        sin = self.rope_sin

        total_rows = B * S
        ROWS_PER_PROG = 2
        if total_rows % ROWS_PER_PROG == 0:
            grid = (total_rows // ROWS_PER_PROG,)
            fused_rmsnorm_rope_kernel_2row[grid](
                x, self.weight, cos, sin, y,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                cos.stride(0),
                S,
                1.0 / D,
                self.eps,
                HALF_C=self.head_dim // 2,
                NUM_HEADS_C=self.num_heads,
                ROWS_PER_PROG=ROWS_PER_PROG,
                num_warps=8,
                num_stages=3,
            )
        else:
            grid = (total_rows,)
            fused_rmsnorm_rope_kernel[grid](
                x, self.weight, cos, sin, y,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                cos.stride(0),
                S,
                1.0 / D,
                self.eps,
                HALF_C=self.head_dim // 2,
                NUM_HEADS_C=self.num_heads,
                num_warps=8,
                num_stages=3,
            )
        return y