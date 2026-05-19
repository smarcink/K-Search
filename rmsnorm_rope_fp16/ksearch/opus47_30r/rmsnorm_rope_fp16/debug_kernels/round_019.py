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

    # Reshape into (NH, HD) and split halves via slicing (no extra global loads)
    NH: tl.constexpr = BLOCK_D // HD_C
    normed_2d = tl.reshape(normed, (NH, HD_C))

    # Slice halves using reshape into (NH, 2, HALF) then index axis 1
    normed_split = tl.reshape(normed, (NH, 2, HALF_C))
    n1_2d = tl.reshape(tl.view(normed_split[:, 0, :], (NH, HALF_C)) if False else normed_split, (NH, 2, HALF_C))
    # Use explicit gather via two reshapes:
    # Build (NH, HALF) for first and second halves
    n1_flat = tl.reshape(normed_split, (NH * 2 * HALF_C,))
    # That's just normed flat; not helpful. Use arange-based splitting instead.

    n_idx = tl.arange(0, NH)[:, None]
    h_idx = tl.arange(0, HALF_C)[None, :]
    first_offs = (n_idx * HD_C + h_idx)
    second_offs = (n_idx * HD_C + h_idx + HALF_C)

    # Gather from normed (in-register) using these offsets via tl.load on a pointer is not possible.
    # Instead: do two loads from the in-register `normed` by recomputing through reshape.
    # normed_2d shape (NH, HD_C). Slice along last dim:
    # Triton supports advanced indexing limited; use tl.reshape + split.
    a, b_half = tl.split(tl.reshape(normed_2d, (NH, 2, HALF_C)).permute(0, 2, 1))
    # a, b_half shape (NH, HALF_C)
    n1_2d = a
    n2_2d = b_half

    # Load cos/sin (HALF_C)
    half_offs = tl.arange(0, HALF_C)
    cos_e = tl.load(COS_ptr + s * stride_cs + half_offs).to(tl.float32)
    sin_e = tl.load(SIN_ptr + s * stride_cs + half_offs).to(tl.float32)

    cos_b = cos_e[None, :]
    sin_b = sin_e[None, :]

    out1 = n1_2d * cos_b - n2_2d * sin_b
    out2 = n1_2d * sin_b + n2_2d * cos_b

    # Stitch back to (NH, HD) by interleaving along last axis: [out1 | out2]
    out_2d = tl.join(out1, out2).reshape(NH, 2, HALF_C).permute(0, 2, 1).reshape(NH, HD_C)
    # That interleaves; we want concat. Use store with split offsets.
    tl.store(row_y + first_offs.reshape(NH * HALF_C), tl.reshape(out1, (NH * HALF_C,)).to(tl.float16))
    tl.store(row_y + second_offs.reshape(NH * HALF_C), tl.reshape(out2, (NH * HALF_C,)).to(tl.float16))


@triton.jit
def fused_rmsnorm_rope_kernel_v2(
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
    rw = rstd * w
    normed = x * rw  # (BLOCK_D,)

    NH: tl.constexpr = BLOCK_D // HD_C
    n_idx = tl.arange(0, NH)[:, None]
    h_idx = tl.arange(0, HALF_C)[None, :]
    first_offs = (n_idx * HD_C + h_idx).reshape(NH * HALF_C)
    second_offs = (n_idx * HD_C + h_idx + HALF_C).reshape(NH * HALF_C)

    # Reload first/second halves from normed via gather using offsets — not possible on register array.
    # Use direct loads of x, w with separate offsets (memory is L1/L2-cached).
    x1 = tl.load(row_x + first_offs).to(tl.float32)
    w1 = tl.load(W_ptr + first_offs).to(tl.float32)
    x2 = tl.load(row_x + second_offs).to(tl.float32)
    w2 = tl.load(W_ptr + second_offs).to(tl.float32)

    n1 = x1 * rstd * w1
    n2 = x2 * rstd * w2

    n1_2d = tl.reshape(n1, (NH, HALF_C))
    n2_2d = tl.reshape(n2, (NH, HALF_C))

    half_offs = tl.arange(0, HALF_C)
    cos_e = tl.load(COS_ptr + s * stride_cs + half_offs).to(tl.float32)
    sin_e = tl.load(SIN_ptr + s * stride_cs + half_offs).to(tl.float32)

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
        fused_rmsnorm_rope_kernel_v2[grid](
            x, self.weight, self.rope_cos, self.rope_sin, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self.rope_cos.stride(0),
            S,
            1.0 / D, self.eps,
            BLOCK_D=BLOCK_D,
            HD_C=HD,
            HALF_C=HALF,
            num_warps=4,
            num_stages=2,
        )
        return y


def get_inputs():
    device = get_device()
    x = torch.randn(8, 2048, 4096, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    return []