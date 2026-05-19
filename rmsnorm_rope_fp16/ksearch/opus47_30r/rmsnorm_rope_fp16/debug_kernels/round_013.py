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
    inv_D, eps,
    BLOCK_D: tl.constexpr,
    HALF_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_x = X_ptr + b * stride_xb + s * stride_xs
    row_y = Y_ptr + b * stride_yb + s * stride_ys

    offs = tl.arange(0, BLOCK_D)
    x = tl.load(row_x + offs).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)

    sumsq = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(sumsq * inv_D + eps)
    normed = x * rstd * w  # fp32, length BLOCK_D = D

    # View normed as (num_heads, HD). Split into two halves.
    # Reshape register vector via 2D arange.
    n2d = tl.reshape(normed, (BLOCK_D // HD, HD))
    n1 = tl.reshape(n2d[:, :HALF], (HALF_TOTAL,))
    n2 = tl.reshape(n2d[:, HALF:], (HALF_TOTAL,))

    # cos/sin: shape (HALF,), broadcast across heads.
    inner = tl.arange(0, HALF)
    cos_v = tl.load(COS_ptr + s * stride_cs + inner).to(tl.float32)
    sin_v = tl.load(SIN_ptr + s * stride_cs + inner).to(tl.float32)

    # Broadcast cos/sin across heads
    cos_e = tl.reshape(tl.broadcast_to(cos_v[None, :], (BLOCK_D // HD, HALF)), (HALF_TOTAL,))
    sin_e = tl.reshape(tl.broadcast_to(sin_v[None, :], (BLOCK_D // HD, HALF)), (HALF_TOTAL,))

    out1 = n1 * cos_e - n2 * sin_e
    out2 = n1 * sin_e + n2 * cos_e

    # Store back to interleaved layout
    h_idx = tl.arange(0, HALF_TOTAL)
    head = h_idx // HALF
    inner2 = h_idx % HALF
    first_off = head * HD + inner2
    second_off = first_off + HALF

    tl.store(row_y + first_off, out1.to(tl.float16))
    tl.store(row_y + second_off, out2.to(tl.float16))


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
        HALF_TOTAL = D // 2

        x = x.contiguous()
        y = torch.empty_like(x)

        BLOCK_D = triton.next_power_of_2(D)

        grid = (B * S,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self.rope_cos.stride(0),
            S, D, HD, HALF,
            1.0 / D, self.eps,
            BLOCK_D=BLOCK_D,
            HALF_TOTAL=HALF_TOTAL,
            num_warps=8,
            num_stages=2,
        )
        return y


def get_inputs():
    device = get_device()
    x = torch.randn(8, 2048, 4096, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    return []