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
    HD_C: tl.constexpr,
    HALF_C: tl.constexpr,
    NHEADS_C: tl.constexpr,
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

    # Reshape register vector into (NHEADS, HD)
    n2d = tl.reshape(normed, (NHEADS_C, HD_C))

    # Split into halves via slicing
    h_range = tl.arange(0, HD_C)
    mask_first = h_range < HALF_C
    # Use where to gather halves: more reliable than slicing
    # Build (NHEADS, HALF) tensors by indexing
    # We'll use tl.reshape with explicit gather via arange products
    nh_idx = tl.arange(0, NHEADS_C)[:, None]
    half_idx = tl.arange(0, HALF_C)[None, :]
    flat_first = nh_idx * HD_C + half_idx
    flat_second = nh_idx * HD_C + (half_idx + HALF_C)

    # Flatten normed to 1D and gather
    normed_1d = tl.reshape(normed, (BLOCK_D,))
    # cos/sin per s
    cos_v = tl.load(COS_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)
    sin_v = tl.load(SIN_ptr + s * stride_cs + tl.arange(0, HALF_C)).to(tl.float32)
    cos_e = tl.broadcast_to(cos_v[None, :], (NHEADS_C, HALF_C))
    sin_e = tl.broadcast_to(sin_v[None, :], (NHEADS_C, HALF_C))

    # Use n2d slicing
    n1 = tl.reshape(tl.where(mask_first[None, :], n2d, 0.0), (NHEADS_C, HD_C))
    # Easier: use tl.split? Just compute via two passes using full HD-wide ops
    # Build rotated full-head: for each head, out[i] = n1*cos - n2*sin for i<HALF, n1*sin + n2*cos for i>=HALF
    # We need n2 aligned to first half positions and n1 aligned to second half positions.
    # Use shifts:
    # Construct n_first_half = n2d[:, :HALF], n_second_half = n2d[:, HALF:]
    # Implemented by gathering with computed indices via tl.load from shared? Not possible from registers.
    # Use a different approach: cos_full/sin_full of HD wide, with sign trick.
    # cos_full[i] = cos[i % HALF]; sin_full[i] = (i<HALF ? -sin[i] : sin[i % HALF])
    # partner[i] = n2d[:, (i + HALF) % HD]  -- a rotation by HALF along last dim.

    # Build full-width cos/sin
    full_idx = tl.arange(0, HD_C)
    inner = full_idx % HALF_C
    is_second = full_idx >= HALF_C
    cos_full_1d = tl.load(COS_ptr + s * stride_cs + inner).to(tl.float32)
    sin_full_1d = tl.load(SIN_ptr + s * stride_cs + inner).to(tl.float32)
    sin_signed = tl.where(is_second, sin_full_1d, -sin_full_1d)
    cos_full = tl.broadcast_to(cos_full_1d[None, :], (NHEADS_C, HD_C))
    sin_full = tl.broadcast_to(sin_signed[None, :], (NHEADS_C, HD_C))

    # Partner: rotate each head by HALF positions (swap halves)
    # partner_index_in_head = (i + HALF) % HD
    # Reload from global into 2D using computed indices is awkward; instead reload from X via partner offs.
    partner_offs_1d = (full_idx + HALF_C) % HD_C  # within a head
    # Global partner offs for the whole row:
    head_base = nh_idx * HD_C  # (NHEADS,1)
    partner_global = head_base + partner_offs_1d[None, :]  # (NHEADS, HD)
    partner_flat = tl.reshape(partner_global, (BLOCK_D,))

    x_p = tl.load(row_x + partner_flat).to(tl.float32)
    w_p = tl.load(W_ptr + partner_flat).to(tl.float32)
    normed_p = x_p * rstd * w_p  # already in correct order

    out = normed_1d * tl.reshape(cos_full, (BLOCK_D,)) + normed_p * tl.reshape(sin_full, (BLOCK_D,))

    tl.store(row_y + offs, out.to(tl.float16))


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
        NHEADS = D // HD

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
            HD_C=HD,
            HALF_C=HALF,
            NHEADS_C=NHEADS,
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