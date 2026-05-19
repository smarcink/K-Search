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
    B, S, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S
    row_off = pid * D

    # We process the row in "half-pairs": for each head h and lane j in [0, HALF),
    # element a = h*HD + j (first half), element b = h*HD + HALF + j (second half).
    # BLOCK_HALF = H * HALF = D/2 elements per "side".
    offs_half = tl.arange(0, BLOCK_HALF)  # indexes (h, j) flattened: h*HALF + j
    h_idx = offs_half // HALF
    j_idx = offs_half % HALF

    a_off = h_idx * HD + j_idx          # first-half element index within row
    b_off = h_idx * HD + HALF + j_idx   # second-half element index within row

    # Load both halves
    xa = tl.load(x_ptr + row_off + a_off).to(tl.float32)
    xb = tl.load(x_ptr + row_off + b_off).to(tl.float32)
    wa = tl.load(w_ptr + a_off).to(tl.float32)
    wb = tl.load(w_ptr + b_off).to(tl.float32)

    # RMSNorm: sum of squares over the full row = sum(xa^2) + sum(xb^2)
    ss = tl.sum(xa * xa, axis=0) + tl.sum(xb * xb, axis=0)
    var = ss / D
    rstd = 1.0 / tl.sqrt(var + eps)

    na = xa * rstd * wa
    nb = xb * rstd * wb

    # Load cos/sin for this position s, shape (HALF,) — broadcast across heads
    cs_off = s * HALF + j_idx
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)

    out_a = na * cos - nb * sin
    out_b = na * sin + nb * cos

    tl.store(out_ptr + row_off + a_off, out_a.to(tl.float16))
    tl.store(out_ptr + row_off + b_off, out_b.to(tl.float16))


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

        HALF = self.head_dim // 2
        BLOCK_HALF = triton.next_power_of_2(self.num_heads * HALF)
        grid = (B * S,)

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, out,
            B, S, D, self.num_heads, self.head_dim, HALF,
            self.eps,
            BLOCK_HALF=BLOCK_HALF,
            num_warps=8,
            num_stages=2,
        )
        return out