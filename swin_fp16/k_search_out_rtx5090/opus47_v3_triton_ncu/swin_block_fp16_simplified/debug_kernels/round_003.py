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
def _swin_block_kernel(
    X_ptr, Y_ptr,
    ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    n_windows,
    C: tl.constexpr,
    N: tl.constexpr,
    HID: tl.constexpr,
    M: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_start = pid * BLOCK_W

    offs_m = tl.arange(0, M)
    offs_c = tl.arange(0, C)
    offs_h = tl.arange(0, HID)
    offs_n = tl.arange(0, N)

    win_idx_row = offs_m // N + w_start
    tok_idx_row = offs_m % N
    row_valid = win_idx_row < n_windows

    # Load X
    x_ptrs = X_ptr + win_idx_row[:, None] * (N * C) + tok_idx_row[:, None] * C + offs_c[None, :]
    x = tl.load(x_ptrs, mask=row_valid[:, None], other=0.0).to(tl.float32)

    # LN1
    mean = tl.sum(x, axis=1) / C
    xm = x - mean[:, None]
    var = tl.sum(xm * xm, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln1_w = tl.load(ln1_w_ptr + offs_c).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + offs_c).to(tl.float32)
    x_ln1 = xm * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]
    x_ln1_f16 = x_ln1.to(tl.float16)

    # QKV: three (C,C) matmuls
    qw_ptr = qkv_w_ptr + offs_c[:, None] * (3 * C) + offs_c[None, :]
    kw_ptr = qkv_w_ptr + offs_c[:, None] * (3 * C) + (C + offs_c[None, :])
    vw_ptr = qkv_w_ptr + offs_c[:, None] * (3 * C) + (2 * C + offs_c[None, :])
    qw = tl.load(qw_ptr)
    kw = tl.load(kw_ptr)
    vw = tl.load(vw_ptr)
    qb = tl.load(qkv_b_ptr + offs_c).to(tl.float32)
    kb = tl.load(qkv_b_ptr + (C + offs_c)).to(tl.float32)
    vb = tl.load(qkv_b_ptr + (2 * C + offs_c)).to(tl.float32)

    q = tl.dot(x_ln1_f16, qw, out_dtype=tl.float32) + qb[None, :]
    k = tl.dot(x_ln1_f16, kw, out_dtype=tl.float32) + kb[None, :]
    v = tl.dot(x_ln1_f16, vw, out_dtype=tl.float32) + vb[None, :]

    scale = 1.0 / tl.sqrt(C.to(tl.float32))

    q_w = tl.reshape(q, (BLOCK_W, N, C))
    k_w = tl.reshape(k, (BLOCK_W, N, C))
    v_w = tl.reshape(v, (BLOCK_W, N, C))

    rpe = tl.load(rpe_ptr + offs_n[:, None] * N + offs_n[None, :]).to(tl.float32)

    out_attn = tl.zeros((BLOCK_W, N, C), dtype=tl.float32)
    for w in tl.static_range(0, BLOCK_W):
        qw_ = tl.reshape(q_w[w, :, :], (N, C)).to(tl.float16)
        kw_ = tl.reshape(k_w[w, :, :], (N, C)).to(tl.float16)
        vw_ = tl.reshape(v_w[w, :, :], (N, C)).to(tl.float16)
        kw_t = tl.trans(kw_)
        attn = tl.dot(qw_, kw_t, out_dtype=tl.float32) * scale + rpe
        attn_max = tl.max(attn, axis=1)
        attn = attn - attn_max[:, None]
        attn_e = tl.exp(attn)
        attn_s = tl.sum(attn_e, axis=1)
        attn_p = (attn_e / attn_s[:, None]).to(tl.float16)
        ctx = tl.dot(attn_p, vw_, out_dtype=tl.float32)
        mask_w = (tl.arange(0, BLOCK_W) == w)[:, None, None]
        out_attn = tl.where(mask_w, tl.reshape(ctx, (1, N, C)), out_attn)

    ctx_m = tl.reshape(out_attn, (M, C)).to(tl.float16)

    # proj: (M,C) @ (C,C)
    proj_w = tl.load(proj_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    proj_b = tl.load(proj_b_ptr + offs_c).to(tl.float32)
    proj_out = tl.dot(ctx_m, proj_w, out_dtype=tl.float32) + proj_b[None, :]

    x1 = x + proj_out

    # LN2
    mean2 = tl.sum(x1, axis=1) / C
    xm2 = x1 - mean2[:, None]
    var2 = tl.sum(xm2 * xm2, axis=1) / C
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    ln2_w = tl.load(ln2_w_ptr + offs_c).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + offs_c).to(tl.float32)
    x_ln2 = xm2 * rstd2[:, None] * ln2_w[None, :] + ln2_b[None, :]
    x_ln2_f16 = x_ln2.to(tl.float16)

    # fc1: (M,C) @ (C,HID)
    fc1_w = tl.load(fc1_w_ptr + offs_c[:, None] * HID + offs_h[None, :])
    fc1_b = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h_out = tl.dot(x_ln2_f16, fc1_w, out_dtype=tl.float32) + fc1_b[None, :]

    # GELU tanh
    k0 = 0.7978845608028654
    k1 = 0.044715
    h_g = 0.5 * h_out * (1.0 + tl.extra.libdevice.tanh(k0 * (h_out + k1 * h_out * h_out * h_out)))
    h_g_f16 = h_g.to(tl.float16)

    # fc2: (M,HID) @ (HID,C)
    fc2_w = tl.load(fc2_w_ptr + offs_h[:, None] * C + offs_c[None, :])
    fc2_b = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    fc2_out = tl.dot(h_g_f16, fc2_w, out_dtype=tl.float32) + fc2_b[None, :]

    y = (x1 + fc2_out).to(tl.float16)

    y_ptrs = Y_ptr + win_idx_row[:, None] * (N * C) + tok_idx_row[:, None] * C + offs_c[None, :]
    tl.store(y_ptrs, y, mask=row_valid[:, None])


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

        self._cached = False

    def _prepare_weights(self):
        if self._cached:
            return
        self._qkv_w_t = self.qkv.weight.t().contiguous()
        self._qkv_b = self.qkv.bias.contiguous()
        self._proj_w_t = self.proj.weight.t().contiguous()
        self._proj_b = self.proj.bias.contiguous()
        self._fc1_w_t = self.fc1.weight.t().contiguous()
        self._fc1_b = self.fc1.bias.contiguous()
        self._fc2_w_t = self.fc2.weight.t().contiguous()
        self._fc2_b = self.fc2.bias.contiguous()
        self._rpe_f16 = self.rpe_bias.to(torch.float16).contiguous()
        self._cached = True

    def forward(self, x: Tensor) -> Tensor:
        self._prepare_weights()
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        hidden = self.hidden

        xw = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, N, C)
        n_windows = xw.shape[0]

        out = torch.empty_like(xw)

        BLOCK_W = 8
        M = BLOCK_W * N
        grid = ((n_windows + BLOCK_W - 1) // BLOCK_W,)

        _swin_block_kernel[grid](
            xw, out,
            self.norm1.weight, self.norm1.bias,
            self._qkv_w_t, self._qkv_b,
            self._proj_w_t, self._proj_b,
            self._rpe_f16,
            self.norm2.weight, self.norm2.bias,
            self._fc1_w_t, self._fc1_b,
            self._fc2_w_t, self._fc2_b,
            n_windows,
            C=C, N=N, HID=hidden, M=M,
            BLOCK_W=BLOCK_W,
            num_warps=4,
        )

        y = out.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return y