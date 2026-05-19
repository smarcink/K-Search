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
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    B, S,
    D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S
    row_off = pid * D

    offs = tl.arange(0, BLOCK_D)

    # Load row and weight (D == BLOCK_D, power of 2 — no mask)
    x = tl.load(x_ptr + row_off + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x * rstd * w  # fp32 vector of length D

    # Split each head into first/second half. Build a "partner" view using
    # within-head index swap: lane' = (lane + HALF) % HD.
    lane = offs % HD
    is_first_half = lane < HALF
    freq_idx = tl.where(is_first_half, lane, lane - HALF)
    cs_off = s * HALF + freq_idx
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)

    # Partner index within full row — get partner value from the already-computed
    # `normed` vector in registers using a permutation via tl.load on a scratch?
    # Triton doesn't expose register shuffles, but we can fetch via a second
    # global load that the L1 will absorb. Use x and w partner-loads.
    partner_lane = tl.where(is_first_half, lane + HALF, lane - HALF)
    head_base = (offs // HD) * HD
    partner_offs = head_base + partner_lane

    x_p = tl.load(x_ptr + row_off + partner_offs).to(tl.float32)
    w_p = tl.load(w_ptr + partner_offs).to(tl.float32)
    normed_p = x_p * rstd * w_p

    sign = tl.where(is_first_half, -1.0, 1.0)
    out = normed * cos + sign * normed_p * sin

    tl.store(out_ptr + row_off + offs, out.to(tl.float16))


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
        out = torch.empty_like(x)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * S,)

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, out,
            B, S, D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )
        return out