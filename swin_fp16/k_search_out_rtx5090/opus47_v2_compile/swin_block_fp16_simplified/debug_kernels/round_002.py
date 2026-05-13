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
def swin_block_kernel(
    X_ptr, Y_ptr,
    ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    B, H, W,
    nH, nW,
    C: tl.constexpr,
    C3: tl.constexpr,           # 3*C, power of 2 if C is
    HIDDEN: tl.constexpr,
    N: tl.constexpr,            # 16
    WS: tl.constexpr,           # 4
    WINDOWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    total_windows = B * nH * nW
    base_w = pid * WINDOWS_PER_PROG

    n_idx = tl.arange(0, N)
    c_idx = tl.arange(0, C)
    h_idx = tl.arange(0, HIDDEN)
    c3_idx = tl.arange(0, C3)

    ln1_w = tl.load(ln1_w_ptr + c_idx).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_idx).to(tl.float32)
    ln2_w = tl.load(ln2_w_ptr + c_idx).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + c_idx).to(tl.float32)

    # qkv_w: (3C, C)
    qkv_w = tl.load(qkv_w_ptr + c3_idx[:, None] * C + c_idx[None, :])
    qkv_b = tl.load(qkv_b_ptr + c3_idx).to(tl.float32)

    # Split into Q, K, V via masking and slicing
    # We'll just keep the full (3C, C) weight and do one matmul, then split rows.

    proj_w = tl.load(proj_w_ptr + c_idx[:, None] * C + c_idx[None, :])
    proj_b = tl.load(proj_b_ptr + c_idx).to(tl.float32)

    rpe = tl.load(rpe_ptr + n_idx[:, None] * N + n_idx[None, :]).to(tl.float32)

    fc1_w = tl.load(fc1_w_ptr + h_idx[:, None] * C + c_idx[None, :])
    fc1_b = tl.load(fc1_b_ptr + h_idx).to(tl.float32)

    fc2_w = tl.load(fc2_w_ptr + c_idx[:, None] * HIDDEN + h_idx[None, :])
    fc2_b = tl.load(fc2_b_ptr + c_idx).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), C, tl.float32))

    qkv_w_t = tl.trans(qkv_w)     # (C, 3C)
    proj_w_t = tl.trans(proj_w)   # (C, C)
    fc1_w_t = tl.trans(fc1_w)     # (C, HIDDEN)
    fc2_w_t = tl.trans(fc2_w)     # (HIDDEN, C)

    for w_off in tl.static_range(WINDOWS_PER_PROG):
        win_id = base_w + w_off
        valid = win_id < total_windows

        b = win_id // (nH * nW)
        rem = win_id % (nH * nW)
        ih = rem // nW
        iw = rem % nW

        i_in = n_idx // WS
        j_in = n_idx % WS
        row = ih * WS + i_in
        col = iw * WS + j_in

        x_off = b * (H * W * C) + row[:, None] * (W * C) + col[:, None] * C + c_idx[None, :]

        x = tl.load(X_ptr + x_off, mask=valid, other=0.0)
        x_f = x.to(tl.float32)
        residual1 = x_f

        # LayerNorm1
        mean = tl.sum(x_f, axis=1) / C
        xc = x_f - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        inv = 1.0 / tl.sqrt(var + 1e-5)
        ln1 = xc * inv[:, None] * ln1_w[None, :] + ln1_b[None, :]
        ln1_h = ln1.to(tl.float16)

        # QKV = ln1 @ qkv_w^T  -> [N, 3C]
        qkv = tl.dot(ln1_h, qkv_w_t)
        qkv = qkv + qkv_b[None, :]

        # Split via masking: rebuild q, k, v separately
        # Use the fact that C is power of 2 and C3 = 3*C may not be; we used C3 only for loading
        # Instead get q,k,v by loading separately too. Simpler: build masks.
        # Build q: select cols [0, C)
        # We do this by loading slices from qkv using shifted indexing.
        # Easiest: load q, k, v from `qkv` tensor by re-indexing
        # qkv shape: [N, C3]. To get q: qkv[:, 0:C]. Triton supports tl.split? Use tl.reshape:
        qkv_r = tl.reshape(qkv, (N, 3, C))
        q = tl.reshape(tl.sum(tl.where(tl.arange(0, 3)[None, :, None] == 0, qkv_r, 0.0), axis=1), (N, C))
        k = tl.reshape(tl.sum(tl.where(tl.arange(0, 3)[None, :, None] == 1, qkv_r, 0.0), axis=1), (N, C))
        v = tl.reshape(tl.sum(tl.where(tl.arange(0, 3)[None, :, None] == 2, qkv_r, 0.0), axis=1), (N, C))

        q_h = (q * scale).to(tl.float16)
        k_h = k.to(tl.float16)
        v_h = v.to(tl.float16)

        attn = tl.dot(q_h, tl.trans(k_h))
        attn = attn + rpe

        amax = tl.max(attn, axis=1)
        attn = attn - amax[:, None]
        attn_e = tl.exp(attn)
        asum = tl.sum(attn_e, axis=1)
        attn_p = (attn_e / asum[:, None]).to(tl.float16)

        out = tl.dot(attn_p, v_h)
        out_h = out.to(tl.float16)

        proj_out = tl.dot(out_h, proj_w_t) + proj_b[None, :]

        x1 = proj_out + residual1
        residual2 = x1

        mean2 = tl.sum(x1, axis=1) / C
        xc2 = x1 - mean2[:, None]
        var2 = tl.sum(xc2 * xc2, axis=1) / C
        inv2 = 1.0 / tl.sqrt(var2 + 1e-5)
        ln2 = xc2 * inv2[:, None] * ln2_w[None, :] + ln2_b[None, :]
        ln2_h = ln2.to(tl.float16)

        h_act = tl.dot(ln2_h, fc1_w_t) + fc1_b[None, :]
        h_gelu = 0.5 * h_act * (1.0 + tl.erf(h_act * 0.7071067811865475))
        h_gelu_h = h_gelu.to(tl.float16)

        fc2_out = tl.dot(h_gelu_h, fc2_w_t) + fc2_b[None, :]

        y = fc2_out + residual2
        y_h = y.to(tl.float16)

        tl.store(Y_ptr + x_off, y_h, mask=valid)


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

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        assert Wh == Ww == 4
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.contiguous()
        y = torch.empty_like(x)

        total_windows = B * nH * nW
        WINDOWS_PER_PROG = 1
        grid = (triton.cdiv(total_windows, WINDOWS_PER_PROG),)

        swin_block_kernel[grid](
            x, y,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            B, H, W,
            nH, nW,
            C,
            3 * C,
            self.hidden,
            N, Wh,
            WINDOWS_PER_PROG,
            num_warps=4,
            num_stages=2,
        )
        return y