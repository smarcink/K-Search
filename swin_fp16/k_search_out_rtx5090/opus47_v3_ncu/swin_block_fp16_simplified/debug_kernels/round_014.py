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
def swin_block_kernel_wpc(
    X_ptr, OUT_ptr,
    ln1_w_ptr, ln1_b_ptr,
    q_w_ptr, q_b_ptr,
    k_w_ptr, k_b_ptr,
    v_w_ptr, v_b_ptr,
    proj_w_ptr, proj_b_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    rpe_ptr,
    B, H, W,
    nH, nW,
    C: tl.constexpr,
    Wh: tl.constexpr, Ww: tl.constexpr, N: tl.constexpr,
    N_PAD: tl.constexpr,
    HIDDEN: tl.constexpr,
    INV_C: tl.constexpr,
    INV_SQRT_C: tl.constexpr,
    LN_EPS: tl.constexpr,
    WPC: tl.constexpr,
):
    pid = tl.program_id(0)
    total_win = nH * nW
    win_start = pid * WPC

    BLOCK_M: tl.constexpr = WPC * N_PAD

    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, C)
    offs_h = tl.arange(0, HIDDEN)

    w_of_m = offs_m // N_PAD
    n_of_m = offs_m % N_PAD

    mask_m = n_of_m < N

    row_in_win = n_of_m // Ww
    col_in_win = n_of_m % Ww

    ln1_w = tl.load(ln1_w_ptr + offs_c).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + offs_c).to(tl.float32)
    q_w = tl.load(q_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    q_b = tl.load(q_b_ptr + offs_c).to(tl.float32)
    k_w = tl.load(k_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    k_b = tl.load(k_b_ptr + offs_c).to(tl.float32)
    v_w = tl.load(v_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    v_b = tl.load(v_b_ptr + offs_c).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    proj_b = tl.load(proj_b_ptr + offs_c).to(tl.float32)
    ln2_w = tl.load(ln2_w_ptr + offs_c).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + offs_c).to(tl.float32)
    fc1_w = tl.load(fc1_w_ptr + offs_c[:, None] * HIDDEN + offs_h[None, :])
    fc1_b = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    fc2_w = tl.load(fc2_w_ptr + offs_h[:, None] * C + offs_c[None, :])
    fc2_b = tl.load(fc2_b_ptr + offs_c).to(tl.float32)

    win_ids = win_start + w_of_m
    valid = win_ids < (B * total_win)
    b = win_ids // total_win
    rem = win_ids % total_win
    ih = rem // nW
    iw = rem % nW

    img_row = ih * Wh + row_in_win
    img_col = iw * Ww + col_in_win
    token_base = b * (H * W * C) + img_row * (W * C) + img_col * C
    x_ptrs = token_base[:, None] + offs_c[None, :]
    load_mask = valid[:, None] & mask_m[:, None]

    x = tl.load(X_ptr + x_ptrs, mask=load_mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) * INV_C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * INV_C
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    x_norm = xc * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]
    x_norm_h = x_norm.to(tl.float16)

    q = tl.dot(x_norm_h, q_w, out_dtype=tl.float32) + q_b[None, :]
    k = tl.dot(x_norm_h, k_w, out_dtype=tl.float32) + k_b[None, :]
    v = tl.dot(x_norm_h, v_w, out_dtype=tl.float32) + v_b[None, :]

    q_scaled = (q * INV_SQRT_C).to(tl.float16)
    k_h = k.to(tl.float16)
    v_h = v.to(tl.float16)

    k_t = tl.trans(k_h)
    attn = tl.dot(q_scaled, k_t, out_dtype=tl.float32)

    same_win = (w_of_m[:, None] == w_of_m[None, :])
    valid_k = mask_m[None, :] & same_win
    rpe_full = tl.load(rpe_ptr + n_of_m[:, None] * N + n_of_m[None, :],
                       mask=mask_m[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
    attn = attn + tl.where(same_win, rpe_full, 0.0)
    attn = tl.where(valid_k, attn, -1.0e30)

    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_exp = tl.where(valid_k, attn_exp, 0.0)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]
    attn_sm_h = attn_sm.to(tl.float16)

    attn_out = tl.dot(attn_sm_h, v_h, out_dtype=tl.float32)
    attn_out_h = attn_out.to(tl.float16)
    proj_out = tl.dot(attn_out_h, proj_w, out_dtype=tl.float32) + proj_b[None, :]

    h1 = x + proj_out

    mean2 = tl.sum(h1, axis=1) * INV_C
    h1c = h1 - mean2[:, None]
    var2 = tl.sum(h1c * h1c, axis=1) * INV_C
    rstd2 = 1.0 / tl.sqrt(var2 + LN_EPS)
    h1_norm = h1c * rstd2[:, None] * ln2_w[None, :] + ln2_b[None, :]
    h1_norm_h = h1_norm.to(tl.float16)

    fc1_out = tl.dot(h1_norm_h, fc1_w, out_dtype=tl.float32) + fc1_b[None, :]
    inv_sqrt2 = 0.70710678118654752440
    gelu_out = 0.5 * fc1_out * (1.0 + tl.erf(fc1_out * inv_sqrt2))
    gelu_out_h = gelu_out.to(tl.float16)

    fc2_out = tl.dot(gelu_out_h, fc2_w, out_dtype=tl.float32) + fc2_b[None, :]
    out = h1 + fc2_out
    out_h = out.to(tl.float16)

    tl.store(OUT_ptr + x_ptrs, out_h, mask=load_mask)


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

        self._cached = None

    def _build_cache(self):
        C = self.dim
        qkv_w = self.qkv.weight
        qkv_b = self.qkv.bias
        q_w = qkv_w[0:C, :].t().contiguous()
        k_w = qkv_w[C:2*C, :].t().contiguous()
        v_w = qkv_w[2*C:3*C, :].t().contiguous()
        q_b = qkv_b[0:C].contiguous()
        k_b = qkv_b[C:2*C].contiguous()
        v_b = qkv_b[2*C:3*C].contiguous()

        proj_w = self.proj.weight.t().contiguous()
        fc1_w = self.fc1.weight.t().contiguous()
        fc2_w = self.fc2.weight.t().contiguous()
        N = self.window_size[0] * self.window_size[1]
        rpe = self.rpe_bias.to(torch.float16).contiguous().view(N, N)

        self._cached = (q_w, q_b, k_w, k_b, v_w, v_b,
                        proj_w, self.proj.bias,
                        fc1_w, self.fc1.bias,
                        fc2_w, self.fc2.bias,
                        rpe)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        NWIN = B * nH * nW

        x = x.contiguous()
        out = torch.empty_like(x)

        if self._cached is None:
            self._build_cache()

        (q_w, q_b, k_w, k_b, v_w, v_b,
         proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, rpe) = self._cached

        N_PAD = 1
        while N_PAD < N:
            N_PAD *= 2
        if N_PAD < 16:
            N_PAD = 16

        WPC = 4
        grid_size = (NWIN + WPC - 1) // WPC
        grid = (grid_size,)
        swin_block_kernel_wpc[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            q_w, q_b, k_w, k_b, v_w, v_b,
            proj_w, proj_b,
            self.norm2.weight, self.norm2.bias,
            fc1_w, fc1_b,
            fc2_w, fc2_b,
            rpe,
            B, H, W,
            nH, nW,
            C,
            Wh, Ww, N, N_PAD,
            self.hidden,
            1.0 / C,
            1.0 / (C ** 0.5),
            1e-5,
            WPC,
            num_warps=4,
            num_stages=3,
        )
        return out