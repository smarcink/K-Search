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
    X_ptr, OUT_ptr,
    ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    rpe_ptr,
    B, H, W, C: tl.constexpr,
    nH, nW, NWIN,
    Wh: tl.constexpr, Ww: tl.constexpr, N: tl.constexpr,
    HIDDEN: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    # decode window index -> (b, ih, iw)
    b = pid // (nH * nW)
    rem = pid % (nH * nW)
    ih = rem // nW
    iw = rem % nW

    # offsets within window: row r in [0,Wh), col c in [0,Ww) -> token index r*Ww+c
    offs_n = tl.arange(0, N)  # 16
    offs_c = tl.arange(0, C)  # 32
    offs_h = tl.arange(0, HIDDEN)  # 128

    # row in image = ih*Wh + (n // Ww); col = iw*Ww + (n % Ww)
    row_in_win = offs_n // Ww
    col_in_win = offs_n % Ww
    img_row = ih * Wh + row_in_win
    img_col = iw * Ww + col_in_win

    # base offset for each of N tokens in X: b*H*W*C + img_row*W*C + img_col*C
    token_base = b * (H * W * C) + img_row * (W * C) + img_col * C  # [N]

    # Load X: shape [N, C]
    x_ptrs = token_base[:, None] + offs_c[None, :]
    x = tl.load(X_ptr + x_ptrs).to(tl.float32)  # [N, C] fp32

    # LayerNorm1
    mean = tl.sum(x, axis=1) / C  # [N]
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    ln1_w = tl.load(ln1_w_ptr + offs_c).to(tl.float32)  # [C]
    ln1_b = tl.load(ln1_b_ptr + offs_c).to(tl.float32)
    x_norm = xc * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]  # [N, C] fp32
    x_norm_h = x_norm.to(tl.float16)

    # QKV = x_norm @ qkv_w + qkv_b ; qkv_w is [C, 3C]
    qkv_w = tl.load(qkv_w_ptr + offs_c[:, None] * (3 * C) + tl.arange(0, 3 * C)[None, :])  # [C, 3C] fp16
    qkv_b = tl.load(qkv_b_ptr + tl.arange(0, 3 * C)).to(tl.float32)  # [3C]
    qkv = tl.dot(x_norm_h, qkv_w, out_dtype=tl.float32) + qkv_b[None, :]  # [N, 3C]

    # split q, k, v: each [N, C]
    q = tl.where(tl.arange(0, 3 * C)[None, :] < C, qkv, 0.0)
    # Use slicing via masks - actually simpler: rebuild via reshape
    # We'll just select columns using arange comparisons
    q_cols = tl.arange(0, C)
    k_cols = tl.arange(0, C) + C
    v_cols = tl.arange(0, C) + 2 * C

    # Reload via pointer arithmetic: extract from qkv tensor
    # Trick: use tl.reshape
    qkv_r = tl.reshape(qkv, (N, 3, C))
    q = tl.reshape(tl.sum(qkv_r * (tl.arange(0, 3)[None, :, None] == 0).to(tl.float32), axis=1), (N, C))
    k = tl.reshape(tl.sum(qkv_r * (tl.arange(0, 3)[None, :, None] == 1).to(tl.float32), axis=1), (N, C))
    v = tl.reshape(tl.sum(qkv_r * (tl.arange(0, 3)[None, :, None] == 2).to(tl.float32), axis=1), (N, C))

    # attn = (q * scale) @ k^T + rpe  -> [N, N]
    scale = 1.0 / tl.sqrt(C.to(tl.float32))
    q_scaled = (q * scale).to(tl.float16)
    k_h = k.to(tl.float16)
    # need k^T as [C, N]
    k_t = tl.trans(k_h)
    attn = tl.dot(q_scaled, k_t, out_dtype=tl.float32)  # [N, N]
    rpe = tl.load(rpe_ptr + offs_n[:, None] * N + offs_n[None, :]).to(tl.float32)
    attn = attn + rpe

    # softmax over last dim
    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]  # [N, N] fp32
    attn_sm_h = attn_sm.to(tl.float16)

    # out = attn @ v -> [N, C]
    v_h = v.to(tl.float16)
    attn_out = tl.dot(attn_sm_h, v_h, out_dtype=tl.float32)  # [N, C]

    # proj: attn_out @ proj_w + proj_b ; proj_w [C, C]
    proj_w = tl.load(proj_w_ptr + offs_c[:, None] * C + offs_c[None, :])
    proj_b = tl.load(proj_b_ptr + offs_c).to(tl.float32)
    attn_out_h = attn_out.to(tl.float16)
    proj_out = tl.dot(attn_out_h, proj_w, out_dtype=tl.float32) + proj_b[None, :]  # [N, C]

    # residual: x_in (original fp16 x) + proj_out
    x_orig = tl.load(X_ptr + x_ptrs).to(tl.float32)
    h1 = x_orig + proj_out  # [N, C] fp32

    # LN2
    mean2 = tl.sum(h1, axis=1) / C
    h1c = h1 - mean2[:, None]
    var2 = tl.sum(h1c * h1c, axis=1) / C
    rstd2 = 1.0 / tl.sqrt(var2 + LN_EPS)
    ln2_w = tl.load(ln2_w_ptr + offs_c).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + offs_c).to(tl.float32)
    h1_norm = h1c * rstd2[:, None] * ln2_w[None, :] + ln2_b[None, :]
    h1_norm_h = h1_norm.to(tl.float16)

    # fc1: [N, C] @ [C, HIDDEN] -> [N, HIDDEN]
    fc1_w = tl.load(fc1_w_ptr + offs_c[:, None] * HIDDEN + offs_h[None, :])
    fc1_b = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    fc1_out = tl.dot(h1_norm_h, fc1_w, out_dtype=tl.float32) + fc1_b[None, :]  # [N, HIDDEN]

    # GELU (tanh approx, matching nn.GELU default which is exact erf; use exact for parity)
    # nn.GELU() default is approximate='none' -> uses erf
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    gelu_out = 0.5 * fc1_out * (1.0 + tl.erf(fc1_out * inv_sqrt2))
    gelu_out_h = gelu_out.to(tl.float16)

    # fc2: [N, HIDDEN] @ [HIDDEN, C] -> [N, C]
    fc2_w = tl.load(fc2_w_ptr + offs_h[:, None] * C + offs_c[None, :])
    fc2_b = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    fc2_out = tl.dot(gelu_out_h, fc2_w, out_dtype=tl.float32) + fc2_b[None, :]  # [N, C]

    # final residual: h1 + fc2_out
    out = h1 + fc2_out
    out_h = out.to(tl.float16)

    tl.store(OUT_ptr + x_ptrs, out_h)


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
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        NWIN = B * nH * nW

        x = x.contiguous()
        out = torch.empty_like(x)

        # qkv weight in nn.Linear is [out, in] = [3C, C]; we need [C, 3C]
        qkv_w = self.qkv.weight.t().contiguous()  # [C, 3C]
        proj_w = self.proj.weight.t().contiguous()  # [C, C]
        fc1_w = self.fc1.weight.t().contiguous()  # [C, HIDDEN]
        fc2_w = self.fc2.weight.t().contiguous()  # [HIDDEN, C]

        rpe = self.rpe_bias.to(torch.float16).contiguous().view(N, N)

        grid = (NWIN,)
        swin_block_kernel[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            qkv_w, self.qkv.bias,
            proj_w, self.proj.bias,
            self.norm2.weight, self.norm2.bias,
            fc1_w, self.fc1.bias,
            fc2_w, self.fc2.bias,
            rpe,
            B, H, W, C,
            nH, nW, NWIN,
            Wh, Ww, N,
            self.hidden,
            1e-5,
            num_warps=4,
        )
        return out