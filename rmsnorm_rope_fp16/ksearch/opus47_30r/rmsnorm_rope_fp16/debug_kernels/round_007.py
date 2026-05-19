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
    B, S, D, H, HD, HALF,
    eps,
    BLOCK_D: tl.constexpr,
    HALF_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    row_x = X_ptr + b * stride_xb + s * stride_xs
    row_y = Y_ptr + b * stride_yb + s * stride_ys

    # --- RMSNorm: compute sum of squares (single pass, vectorized) ---
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    mean = sumsq / D
    rstd = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w  # fp32

    # --- Load cos/sin once for this row (s) ---
    # cos/sin shape: (S, HALF). Stride per s = HALF.
    cs_offs = tl.arange(0, HALF_C)
    cs_mask = cs_offs < HALF
    cos_row = tl.load(COS_ptr + s * stride_cs + cs_offs, mask=cs_mask, other=0.0).to(tl.float32)
    sin_row = tl.load(SIN_ptr + s * stride_cs + cs_offs, mask=cs_mask, other=0.0).to(tl.float32)

    # --- Apply RoPE per head: normed reshaped to (H, HD). For each head h:
    # first half: x1*cos - x2*sin, second half: x1*sin + x2*cos
    # normed is a 1D vector of length D = H*HD. Indices for head h:
    #   x1 indices: h*HD + i,         i in [0, HALF)
    #   x2 indices: h*HD + HALF + i
    # We construct a vector of length D where:
    #   pos in [h*HD, h*HD+HALF):     out = x1*cos[i] - x2*sin[i]
    #   pos in [h*HD+HALF, h*HD+HD):  out = x1*sin[i] + x2*cos[i]
    # Build per-element cos/sin and partner indices using offs.
    head_idx = offs // HD
    in_head = offs % HD
    is_second = in_head >= HALF
    i_idx = tl.where(is_second, in_head - HALF, in_head)  # index into cos/sin row [0, HALF)

    # gather cos/sin via tl.load with offsets (broadcast from row arrays)
    # We can use tl.where to construct partner offsets and reload from normed via shared memory.
    # Since normed is in registers (one chunk of BLOCK_D), use tl.where with shifts.
    # Partner offset within row: if in second half, partner = offs - HALF; if in first half, partner = offs + HALF
    partner_offs = tl.where(is_second, offs - HALF, offs + HALF)
    partner_mask = partner_offs < D
    # Reload partner value from global (already cached in L1/L2 from first load)
    x_partner = tl.load(row_x + partner_offs, mask=partner_mask, other=0.0).to(tl.float32)
    w_partner = tl.load(W_ptr + partner_offs, mask=partner_mask, other=0.0).to(tl.float32)
    normed_partner = x_partner * rstd * w_partner

    # gather cos/sin by index i_idx; use tl.load from COS_ptr+s*stride
    cos_e = tl.load(COS_ptr + s * stride_cs + i_idx, mask=mask, other=0.0).to(tl.float32)
    sin_e = tl.load(SIN_ptr + s * stride_cs + i_idx, mask=mask, other=0.0).to(tl.float32)

    # current is x1 if first half, x2 if second half
    # if first half (is_second=False): out = x1*cos - x2*sin = normed*cos - partner*sin
    # if second half (is_second=True):  out = x1*sin + x2*cos = partner*sin + normed*cos
    out = tl.where(is_second,
                   normed_partner * sin_e + normed * cos_e,
                   normed * cos_e - normed_partner * sin_e)

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
        H = self.num_heads
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
            B, S, D, H, HD, HALF,
            self.eps,
            BLOCK_D=BLOCK_D,
            HALF_C=HALF_C,
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