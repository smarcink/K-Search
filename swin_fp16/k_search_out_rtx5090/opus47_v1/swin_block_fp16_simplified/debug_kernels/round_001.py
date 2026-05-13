import torch
from torch import nn, Tensor
import triton
import triton.language as tl


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
def fused_ln_mlp_kernel(
    X_ptr,            # input tokens after attn residual, shape (M, C) fp16
    RES_ptr,          # residual to add (same as X for residual connection)
    OUT_ptr,          # output (M, C) fp16
    LN_W_ptr, LN_B_ptr,        # (C,)
    FC1_W_ptr, FC1_B_ptr,      # FC1 weight (C, H), bias (H,)
    FC2_W_ptr, FC2_B_ptr,      # FC2 weight (H, C), bias (C,)
    M,
    eps,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < M

    c_offs = tl.arange(0, C)
    h_offs = tl.arange(0, H)

    # Load X: (BLOCK_M, C)
    x_ptrs = X_ptr + row_offs[:, None] * C + c_offs[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=1) / C
    xm = x - mean[:, None]
    var = tl.sum(xm * xm, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xm * rstd[:, None]

    ln_w = tl.load(LN_W_ptr + c_offs).to(tl.float32)
    ln_b = tl.load(LN_B_ptr + c_offs).to(tl.float32)
    xn = xn * ln_w[None, :] + ln_b[None, :]
    xn_f16 = xn.to(tl.float16)

    # FC1: (BLOCK_M, C) @ (C, H) -> (BLOCK_M, H)
    fc1_w_ptrs = FC1_W_ptr + c_offs[:, None] * H + h_offs[None, :]
    fc1_w = tl.load(fc1_w_ptrs)
    hidden = tl.dot(xn_f16, fc1_w, out_dtype=tl.float32)

    fc1_b = tl.load(FC1_B_ptr + h_offs).to(tl.float32)
    hidden = hidden + fc1_b[None, :]

    # GELU (tanh approx)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    k0 = 0.7978845608028654
    k1 = 0.044715
    hidden_cubed = hidden * hidden * hidden
    inner = k0 * (hidden + k1 * hidden_cubed)
    gelu_out = 0.5 * hidden * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    gelu_f16 = gelu_out.to(tl.float16)

    # FC2: (BLOCK_M, H) @ (H, C) -> (BLOCK_M, C)
    fc2_w_ptrs = FC2_W_ptr + h_offs[:, None] * C + c_offs[None, :]
    fc2_w = tl.load(fc2_w_ptrs)
    out = tl.dot(gelu_f16, fc2_w, out_dtype=tl.float32)

    fc2_b = tl.load(FC2_B_ptr + c_offs).to(tl.float32)
    out = out + fc2_b[None, :]

    # Residual
    res = tl.load(RES_ptr + row_offs[:, None] * C + c_offs[None, :],
                  mask=row_mask[:, None], other=0.0).to(tl.float32)
    out = out + res

    tl.store(OUT_ptr + row_offs[:, None] * C + c_offs[None, :],
             out.to(tl.float16), mask=row_mask[:, None])


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

        # Pre-transpose weights for tl.dot:
        # fc1.weight is (H, C); we need (C, H)
        # fc2.weight is (C, H); we need (H, C)
        self._fc1_wt = None
        self._fc2_wt = None

    def _get_fc_weights(self):
        if self._fc1_wt is None or self._fc1_wt.device != self.fc1.weight.device:
            self._fc1_wt = self.fc1.weight.t().contiguous()
            self._fc2_wt = self.fc2.weight.t().contiguous()
        return self._fc1_wt, self._fc2_wt

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

    def _fused_ln_mlp(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C) -> flat (M, C)
        B, H, W, C = x.shape
        M = B * H * W
        x_flat = x.reshape(M, C).contiguous()
        out = torch.empty_like(x_flat)

        fc1_wt, fc2_wt = self._get_fc_weights()

        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        fused_ln_mlp_kernel[grid](
            x_flat, x_flat, out,
            self.norm2.weight, self.norm2.bias,
            fc1_wt, self.fc1.bias,
            fc2_wt, self.fc2.bias,
            M,
            float(self.norm2.eps),
            BLOCK_M=BLOCK_M,
            C=C,
            H=self.hidden,
            num_warps=4,
            num_stages=2,
        )
        return out.view(B, H, W, C)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x))
        x = self._fused_ln_mlp(x)
        return x