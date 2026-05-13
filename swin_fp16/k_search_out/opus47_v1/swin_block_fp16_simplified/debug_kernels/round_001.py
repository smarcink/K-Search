import torch
from torch import nn, Tensor
import triton
import triton.language as tl


def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_relative_position_bias(window_size, num_heads, dtype):
    Wh, Ww = window_size
    table = nn.init.trunc_normal_(
        torch.empty((2 * Wh - 1) * (2 * Ww - 1), num_heads, dtype=dtype),
        std=0.02,
    )
    coords = torch.stack(torch.meshgrid(
        torch.arange(Wh), torch.arange(Ww), indexing="ij"))
    coords = coords.flatten(1)
    rel = coords[:, :, None] - coords[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[..., 0] += Wh - 1
    rel[..., 1] += Ww - 1
    rel[..., 0] *= 2 * Ww - 1
    idx = rel.sum(-1).flatten()
    bias = table[idx].view(Wh * Ww, Wh * Ww, num_heads).permute(2, 0, 1)
    return bias.unsqueeze(0).contiguous()


@triton.jit
def fused_mlp_kernel(
    x_ptr, y_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows

    c_idx = tl.arange(0, C)
    h_idx = tl.arange(0, H)

    # Load input rows: (BLOCK_M, C)
    x_ptrs = x_ptr + rows[:, None] * C + c_idx[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln_w = tl.load(ln_w_ptr + c_idx).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + c_idx).to(tl.float32)
    y_ln = xc * rstd[:, None] * ln_w[None, :] + ln_b[None, :]

    # FC1: (BLOCK_M, C) @ (C, H) + (H,)
    fc1_w = tl.load(fc1_w_ptr + h_idx[:, None] * C + c_idx[None, :]).to(tl.float32)
    # fc1_w stored as (H, C); we want y_ln @ fc1_w.T -> (BLOCK_M, H)
    h1 = tl.dot(y_ln.to(tl.float16), tl.trans(fc1_w.to(tl.float16))).to(tl.float32)
    fc1_b = tl.load(fc1_b_ptr + h_idx).to(tl.float32)
    h1 = h1 + fc1_b[None, :]

    # GELU (exact, matches nn.GELU default)
    h1 = 0.5 * h1 * (1.0 + tl.erf(h1 * 0.7071067811865476))

    # FC2: (BLOCK_M, H) @ (H, C) + (C,)
    fc2_w = tl.load(fc2_w_ptr + c_idx[:, None] * H + h_idx[None, :]).to(tl.float32)
    # fc2_w stored as (C, H); we want h1 @ fc2_w.T -> (BLOCK_M, C)
    out = tl.dot(h1.to(tl.float16), tl.trans(fc2_w.to(tl.float16))).to(tl.float32)
    fc2_b = tl.load(fc2_b_ptr + c_idx).to(tl.float32)
    out = out + fc2_b[None, :]

    # Residual
    out = out + x

    # Store
    y_ptrs = y_ptr + rows[:, None] * C + c_idx[None, :]
    tl.store(y_ptrs, out.to(tl.float16), mask=row_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1
        assert tuple(shift_size) == (0, 0)
        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = list(window_size)
        self.hidden = hidden

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)

        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

        rpe = _make_relative_position_bias(self.window_size, num_heads, dtype=torch.float32)
        self.register_buffer("rpe_bias", rpe)

        self.half()

    def _attn(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)

        qkv = self.qkv(x).reshape(-1, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1) + self.rpe_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(-1, N, C)
        x = self.proj(x)

        x = x.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return x

    def _fused_mlp(self, y1: Tensor) -> Tensor:
        # y1: (B, H, W, C), contiguous
        y1 = y1.contiguous()
        shape = y1.shape
        C = self.dim
        Hh = self.hidden
        n_rows = y1.numel() // C
        x_flat = y1.view(n_rows, C)
        out = torch.empty_like(x_flat)

        BLOCK_M = 64
        grid = (triton.cdiv(n_rows, BLOCK_M),)

        fused_mlp_kernel[grid](
            x_flat, out,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            n_rows,
            BLOCK_M=BLOCK_M,
            C=C,
            H=Hh,
        )
        return out.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x))
        x = self._fused_mlp(x)
        return x