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
    S,
    D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, HALF: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % S

    row_off = pid * D
    offs = tl.arange(0, BLOCK_D)

    # Load full row once
    x = tl.load(x_ptr + row_off + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * w  # fp32, in registers

    # Reshape normed to (H, HD) view via reshape
    normed_2d = tl.reshape(normed, (H, HD))
    # Split first/second half along HD: indices [0, HALF) and [HALF, HD)
    # Use arange to index along axis
    lane = tl.arange(0, HALF)
    # Build column index masks via slicing through reshape
    # Easier: reshape to (H, 2, HALF) and split halves
    normed_3d = tl.reshape(normed, (H, 2, HALF))
    x1 = tl.reshape(tl.sum(normed_3d * tl.where(tl.arange(0, 2)[None, :, None] == 0, 1.0, 0.0), axis=1), (H, HALF))
    x2 = tl.reshape(tl.sum(normed_3d * tl.where(tl.arange(0, 2)[None, :, None] == 1, 1.0, 0.0), axis=1), (H, HALF))

    # cos/sin: (HALF,) broadcast across H
    cs_off = s * HALF + lane
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)

    out1 = x1 * cos[None, :] - x2 * sin[None, :]
    out2 = x1 * sin[None, :] + x2 * cos[None, :]

    # Stitch back: out[(h, 0:HALF)] = out1, out[(h, HALF:HD)] = out2
    # Write directly via 2D index
    head_offs = tl.arange(0, H)[:, None] * HD
    out1_off = head_offs + lane[None, :]
    out2_off = head_offs + HALF + lane[None, :]

    tl.store(out_ptr + row_off + out1_off, out1.to(tl.float16))
    tl.store(out_ptr + row_off + out2_off, out2.to(tl.float16))


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

        BLOCK_D = D
        grid = (B * S,)

        fused_rmsnorm_rope_kernel[grid](
            x, self.weight, self.rope_cos, self.rope_sin, out,
            S,
            D, self.num_heads, self.head_dim, self.head_dim // 2,
            self.eps,
            BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=2,
        )
        return out