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
    inv_D, eps,
    BLOCK_D: tl.constexpr,
    HD_C: tl.constexpr,
    HALF_C: tl.constexpr,
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
    rstd = tl.rsqrt(sumsq * inv_D + eps)
    normed = x * (rstd * w)

    # Reshape into (NUM_HEADS, HD) to do RoPE per head using register-level slicing
    NH: tl.constexpr = BLOCK_D // HD_C
    normed_2d = tl.reshape(normed, (NH, HD_C))

    # Split into halves along head_dim
    half_offs = tl.arange(0, HALF_C)
    # Build (NH, HALF) views via reshape: take first half and second half
    # Use 2D arange index
    n_idx = tl.arange(0, NH)[:, None]
    h_idx = tl.arange(0, HALF_C)[None, :]
    # First half: index [n, h]
    x1 = tl.reshape(
        tl.load(row_x + (n_idx * HD_C + h_idx).reshape(NH * HALF_C)),
        (NH, HALF_C),
    )
    # Actually simpler: derive from normed_2d via gather-equivalent reshape isn't directly supported.
    # Recompute x1, x2 from normed by slicing using masks. Use the trick: rebuild via two separate loads.

    # Load cos/sin (HALF_C)
    cos_e = tl.load(COS_ptr + s * stride_cs + half_offs).to(tl.float32)
    sin_e = tl.load(SIN_ptr + s * stride_cs + half_offs).to(tl.float32)

    # Recompute normed for first and second halves cleanly with two loads:
    # offsets for first half across all heads:
    first_offs = (n_idx * HD_C + h_idx).reshape(NH * HALF_C)
    second_offs = (n_idx * HD_C + h_idx + HALF_C).reshape(NH * HALF_C)

    x1_raw = tl.load(row_x + first_offs).to(tl.float32)
    w1 = tl.load(W_ptr + first_offs).to(tl.float32)
    x2_raw = tl.load(row_x + second_offs).to(tl.float32)
    w2 = tl.load(W_ptr + second_offs).to(tl.float32)

    n1 = x1_raw * rstd * w1
    n2 = x2_raw * rstd * w2

    n1_2d = tl.reshape(n1, (NH, HALF_C))
    n2_2d = tl.reshape(n2, (NH, HALF_C))

    cos_b = cos_e[None, :]
    sin_b = sin_e[None, :]

    out1 = n1_2d * cos_b - n2_2d * sin_b
    out2 = n1_2d * sin_b + n2_2d * cos_b

    tl.store(row_y + first_offs, tl.reshape(out1, (NH * HALF_C,)).to(tl.float16))
    tl.store(row_y + second_offs, tl.reshape(out2, (NH * HALF_C,)).to(tl.float16))


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

        grid = (B * S,)
        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self.rope_cos.stride(0),
            S,
            1.0 / D, self.eps,
            BLOCK_D=BLOCK_D,
            HD_C=HD,
            HALF_C=HALF,
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