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

    # Load full row
    x = tl.load(x_ptr + row_off + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)

    # RMSNorm
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * w  # fp32, shape (BLOCK_D,)

    # Reshape to (H, 2, HALF)
    normed_3d = tl.reshape(normed, (H, 2, HALF))

    # Load cos/sin for this position (shape HALF), broadcast across heads
    cs_off = s * HALF + tl.arange(0, HALF)
    cos = tl.load(cos_ptr + cs_off).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off).to(tl.float32)

    # Permute to (H, HALF, 2) and split
    perm = tl.permute(normed_3d, (1, 0, 2))   # (2, H, HALF)
    perm2 = tl.permute(perm, (1, 2, 0))       # (H, HALF, 2)
    x1, x2 = tl.split(perm2)                  # each (H, HALF)

    out1 = x1 * cos[None, :] - x2 * sin[None, :]
    out2 = x1 * sin[None, :] + x2 * cos[None, :]

    joined = tl.join(out1, out2)              # (H, HALF, 2)
    stacked = tl.permute(joined, (0, 2, 1))   # (H, 2, HALF)
    out_flat = tl.reshape(stacked, (BLOCK_D,))

    tl.store(out_ptr + row_off + offs, out_flat.to(tl.float16))


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
            num_warps=4,
            num_stages=2,
        )
        return out