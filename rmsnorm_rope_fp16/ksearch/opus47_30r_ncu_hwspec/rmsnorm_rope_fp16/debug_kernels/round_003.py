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
    S, D,
    eps,
    BLOCK_D: tl.constexpr,
    HALF_C: tl.constexpr,
    NUM_HEADS_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    x_row_ptr = X_ptr + b * stride_xb + s * stride_xs
    y_row_ptr = Y_ptr + b * stride_yb + s * stride_ys

    offs = tl.arange(0, BLOCK_D)

    # Load full row in fp16, compute rmsnorm in fp32
    x_fp16 = tl.load(x_row_ptr + offs)
    x = x_fp16.to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / D
    rstd = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(W_ptr + offs).to(tl.float32)
    normed = x * rstd * w  # fp32, length BLOCK_D == D

    # Build 2D indices: (NUM_HEADS, HALF)
    head_idx = tl.arange(0, NUM_HEADS_C)[:, None]   # (H, 1)
    half_idx = tl.arange(0, HALF_C)[None, :]        # (1, HALF)

    low_offs = head_idx * (2 * HALF_C) + half_idx                # (H, HALF)
    high_offs = head_idx * (2 * HALF_C) + HALF_C + half_idx      # (H, HALF)

    # Gather low and high halves directly from global (already cached in L1 via prior load)
    n_low_vals = tl.load(x_row_ptr + low_offs).to(tl.float32)
    n_high_vals = tl.load(x_row_ptr + high_offs).to(tl.float32)
    w_low = tl.load(W_ptr + low_offs).to(tl.float32)
    w_high = tl.load(W_ptr + high_offs).to(tl.float32)
    n_low_vals = n_low_vals * rstd * w_low
    n_high_vals = n_high_vals * rstd * w_high

    cos_row_ptr = COS_ptr + s * stride_cs
    sin_row_ptr = SIN_ptr + s * stride_cs
    cos = tl.load(cos_row_ptr + tl.arange(0, HALF_C)).to(tl.float32)[None, :]  # (1, HALF)
    sin = tl.load(sin_row_ptr + tl.arange(0, HALF_C)).to(tl.float32)[None, :]  # (1, HALF)

    out_low = n_low_vals * cos - n_high_vals * sin
    out_high = n_low_vals * sin + n_high_vals * cos

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

        BLOCK_D = D
        grid = (B * S,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, cos, sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            cos.stride(0),
            S, D,
            self.eps,
            BLOCK_D=BLOCK_D,
            HALF_C=self.head_dim // 2,
            NUM_HEADS_C=self.num_heads,
            num_warps=8,
            num_stages=2,
        )
        return y