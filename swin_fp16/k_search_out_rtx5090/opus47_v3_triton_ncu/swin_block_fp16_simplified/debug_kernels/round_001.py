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
    H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_start = pid * BLOCK_W
    # M = BLOCK_W * N rows of width C
    M: tl.constexpr = BLOCK_W * N

    offs_m = tl.arange(0, M)  # token index across BLOCK_W*N
    offs_c = tl.arange(0, C)
    offs_3c = tl.arange(0, 3 * C)
    offs_h = tl.arange(0, H)
    offs_n = tl.arange(0, N)

    # window indices for each row of M
    win_idx_row = offs_m // N + w_start  # [M]
    tok_idx_row = offs_m % N             # [M]
    row_valid = win_idx_row < n_windows

    # Load input X: shape (n_windows*N, C) — windows are already partitioned externally
    x_ptrs = X_ptr + win_idx_row[:, None] * (N * C) + tok_idx_row[:, None] * C + offs_c[None, :]
    x = tl.load(x_ptrs, mask=row_valid[:, None], other=0.0).to(tl.float32)

    # ---- LN1 ----
    mean = tl.sum(x, axis=1) / C
    xm = x - mean[:, None]
    var = tl.sum(xm * xm, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln1_w = tl.load(ln1_w_ptr + offs_c).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + offs_c).to(tl.float32)
    x_ln1 = xm * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]
    x_ln1_f16 = x_ln1.to(tl.float16)

    # ---- QKV linear: (M, C) @ (C, 3C) ----
    qkv_w = tl.load(qkv_w_ptr + offs_c[:, None] * (3 * C) + offs_3c[None, :])  # (C, 3C)
    qkv_b = tl.load(qkv_b_ptr + offs_3c).to(tl.float32)
    qkv = tl.dot(x_ln1_f16, qkv_w, out_dtype=tl.float32) + qkv_b[None, :]  # (M, 3C)

    # Split q, k, v
    q = tl.reshape(tl.where((offs_3c < C)[None, :], qkv, 0.0), (M, 3 * C))  # placeholder
    # Instead use slicing via masks: extract via separate loads-like ops
    # Reshape: qkv (M, 3C) -> view as (M, 3, C)
    qkv_r = tl.reshape(qkv, (BLOCK_W, N, 3, C))
    q = tl.reshape(qkv_r[:, :, 0, :], (M, C))
    k = tl.reshape(qkv_r[:, :, 1, :], (M, C))
    v = tl.reshape(qkv_r[:, :, 2, :], (M, C))

    scale = 1.0 / tl.sqrt(C.to(tl.float32))

    # ---- Attention per window: q (BLOCK_W,N,C) @ k^T (BLOCK_W,C,N) ----
    q_w = tl.reshape(q, (BLOCK_W, N, C))
    k_w = tl.reshape(k, (BLOCK_W, N, C))
    v_w = tl.reshape(v, (BLOCK_W, N, C))

    # rpe bias (1, H=1, N, N) -> (N, N)
    rpe = tl.load(rpe_ptr + offs_n[:, None] * N + offs_n[None, :]).to(tl.float32)

    # compute attn per window using batched-style flatten: treat as BLOCK_W independent NxN
    # Flatten batched matmul as block-diag-free via loop unroll using tl.dot on (BLOCK_W*N, C) x (C, N) per window — we'll do per-window using reshape trick:
    # attn_bw[w] = q_w[w] @ k_w[w].T  ; do it by computing q_flat @ k_flat^T then masking same-window pairs
    # Simpler: use tl.dot on (M, C) and (C, M_other) — but we want per-window. Use explicit loop:
    out_attn = tl.zeros((BLOCK_W, N, C), dtype=tl.float32)
    for w in tl.static_range(0, BLOCK_W):
        qw = tl.reshape(q_w[w, :, :], (N, C)).to(tl.float16)
        kw = tl.reshape(k_w[w, :, :], (N, C)).to(tl.float16)
        vw = tl.reshape(v_w[w, :, :], (N, C)).to(tl.float16)
        # attn = q @ k^T * scale + rpe
        kw_t = tl.trans(kw)
        attn = tl.dot(qw, kw_t, out_dtype=tl.float32) * scale + rpe
        # softmax along last dim
        attn_max = tl.max(attn, axis=1)
        attn = attn - attn_max[:, None]
        attn_e = tl.exp(attn)
        attn_s = tl.sum(attn_e, axis=1)
        attn_p = (attn_e / attn_s[:, None]).to(tl.float16)
        # ctx = attn_p @ v
        ctx = tl.dot(attn_p, vw, out_dtype=tl.float32)  # (N, C)
        # store into out_attn[w]
        mask_w = (tl.arange(0, BLOCK_W) == w)[:, None, None]
        out_attn = tl.where(mask_w, tl.reshape(ctx, (1, N, C)), out_attn)

    ctx_m = tl.reshape(out_attn, (M, C)).to(tl.float16)

    # ---- proj linear: (M, C) @ (C, C) ----
    proj_w = tl.load(proj_w_ptr + offs_c[:, None] * C + offs_c[None, :])  # (C, C)
    proj_b = tl.load(proj_b_ptr + offs_c).to(tl.float32)
    proj_out = tl.dot(ctx_m, proj_w, out_dtype=tl.float32) + proj_b[None, :]

    # residual: x_orig + proj_out
    x1 = x + proj_out  # fp32 (M, C)

    # ---- LN2 ----
    mean2 = tl.sum(x1, axis=1) / C
    xm2 = x1 - mean2[:, None]
    var2 = tl.sum(xm2 * xm2, axis=1) / C
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    ln2_w = tl.load(ln2_w_ptr + offs_c).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + offs_c).to(tl.float32)
    x_ln2 = xm2 * rstd2[:, None] * ln2_w[None, :] + ln2_b[None, :]
    x_ln2_f16 = x_ln2.to(tl.float16)

    # ---- fc1: (M, C) @ (C, H) ----
    offs_h_full = tl.arange(0, H)
    fc1_w = tl.load(fc1_w_ptr + offs_c[:, None] * H + offs_h_full[None, :])  # (C, H)
    fc1_b = tl.load(fc1_b_ptr + offs_h_full).to(tl.float32)
    h_out = tl.dot(x_ln2_f16, fc1_w, out_dtype=tl.float32) + fc1_b[None, :]  # (M, H)

    # GELU (tanh approx)
    k0 = 0.7978845608028654
    k1 = 0.044715
    h_g = 0.5 * h_out * (1.0 + tl.extra.libdevice.tanh(k0 * (h_out + k1 * h_out * h_out * h_out)))
    h_g_f16 = h_g.to(tl.float16)

    # ---- fc2: (M, H) @ (H, C) ----
    fc2_w = tl.load(fc2_w_ptr + offs_h_full[:, None] * C + offs_c[None, :])  # (H, C)
    fc2_b = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    fc2_out = tl.dot(h_g_f16, fc2_w, out_dtype=tl.float32) + fc2_b[None, :]

    y = (x1 + fc2_out).to(tl.float16)

    # store
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

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        hidden = self.hidden

        # window partition: (B, H, W, C) -> (B*nH*nW, N, C)
        xw = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, N, C)
        n_windows = xw.shape[0]

        out = torch.empty_like(xw)

        # Prepare weights in proper layouts (already in fp16)
        qkv_w = self.qkv.weight.t().contiguous()  # (C, 3C)
        qkv_b = self.qkv.bias.contiguous()
        proj_w = self.proj.weight.t().contiguous()  # (C, C)
        proj_b = self.proj.bias.contiguous()
        fc1_w = self.fc1.weight.t().contiguous()  # (C, H)
        fc1_b = self.fc1.bias.contiguous()
        fc2_w = self.fc2.weight.t().contiguous()  # (H, C)
        fc2_b = self.fc2.bias.contiguous()

        rpe = self.rpe_bias.to(torch.float16).contiguous()

        BLOCK_W = 8
        grid = ((n_windows + BLOCK_W - 1) // BLOCK_W,)

        _swin_block_kernel[grid](
            xw, out,
            self.norm1.weight, self.norm1.bias,
            qkv_w, qkv_b,
            proj_w, proj_b,
            rpe,
            self.norm2.weight, self.norm2.bias,
            fc1_w, fc1_b,
            fc2_w, fc2_b,
            n_windows,
            C=C, N=N, H=hidden,
            BLOCK_W=BLOCK_W,
            num_warps=4,
        )

        # window reverse
        y = out.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return y