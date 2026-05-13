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
def fused_attn_kernel(
    x_ptr, out_ptr,
    g1_ptr, b1_ptr,
    wqkv_ptr, bqkv_ptr,
    wproj_ptr, bproj_ptr,
    rpe_ptr,
    H, W, nH, nW, Wh, Ww,
    eps,
    BLOCK_N: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)

    b = pid // (nH * nW)
    w_in_b = pid % (nH * nW)
    wh_idx = w_in_b // nW
    ww_idx = w_in_b % nW

    base = b * H * W * C + wh_idx * Wh * W * C + ww_idx * Ww * C

    i_vec = tl.arange(0, BLOCK_N)
    row_in_w = i_vec // Ww
    col_in_w = i_vec % Ww
    token_off = base + row_in_w * W * C + col_in_w * C  # (N,)

    cols = tl.arange(0, C)

    x_ptrs = x_ptr + token_off[:, None] + cols[None, :]
    x_in = tl.load(x_ptrs)  # (N, C) fp16
    x_f32 = x_in.to(tl.float32)

    # LayerNorm(norm1)
    mean = tl.sum(x_f32, axis=1) / C
    xc = x_f32 - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g1_ptr + cols).to(tl.float32)
    b1v = tl.load(b1_ptr + cols).to(tl.float32)
    xn = xc * rstd[:, None] * g[None, :] + b1v[None, :]
    xn_fp16 = xn.to(tl.float16)

    # Load Q, K, V weights (each (C,C))
    w_out = tl.arange(0, C)
    w_in = tl.arange(0, C)
    wq = tl.load(wqkv_ptr + w_out[:, None] * C + w_in[None, :])
    wk = tl.load(wqkv_ptr + C * C + w_out[:, None] * C + w_in[None, :])
    wv = tl.load(wqkv_ptr + 2 * C * C + w_out[:, None] * C + w_in[None, :])

    bq = tl.load(bqkv_ptr + tl.arange(0, C)).to(tl.float32)
    bk = tl.load(bqkv_ptr + C + tl.arange(0, C)).to(tl.float32)
    bv = tl.load(bqkv_ptr + 2 * C + tl.arange(0, C)).to(tl.float32)

    wq_t = tl.trans(wq)
    wk_t = tl.trans(wk)
    wv_t = tl.trans(wv)

    Q = tl.dot(xn_fp16, wq_t, out_dtype=tl.float32) + bq[None, :]
    K = tl.dot(xn_fp16, wk_t, out_dtype=tl.float32) + bk[None, :]
    V = tl.dot(xn_fp16, wv_t, out_dtype=tl.float32) + bv[None, :]

    scale = 0.17677669529663687  # 1/sqrt(32)
    Q_s = (Q * scale).to(tl.float16)
    K_fp16 = K.to(tl.float16)
    V_fp16 = V.to(tl.float16)

    K_t = tl.trans(K_fp16)  # (C, N)
    attn = tl.dot(Q_s, K_t, out_dtype=tl.float32)  # (N, N)

    rpe = tl.load(rpe_ptr + i_vec[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]).to(tl.float32)
    attn = attn + rpe

    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = (attn_exp / attn_sum[:, None]).to(tl.float16)

    O = tl.dot(attn_sm, V_fp16, out_dtype=tl.float32)  # (N, C)

    wp = tl.load(wproj_ptr + w_out[:, None] * C + w_in[None, :])
    wp_t = tl.trans(wp)
    bp = tl.load(bproj_ptr + tl.arange(0, C)).to(tl.float32)

    O_fp16 = O.to(tl.float16)
    P = tl.dot(O_fp16, wp_t, out_dtype=tl.float32) + bp[None, :]

    # residual
    P = P + x_in.to(tl.float32)

    tl.store(out_ptr + token_off[:, None] + cols[None, :], P.to(tl.float16))


@triton.jit
def fused_ln_mlp_kernel(
    x_ptr,
    g2_ptr, b2n_ptr,
    w1_ptr, b1_ptr,
    w2_ptr, bf2_ptr,
    out_ptr,
    M, C, H,
    eps,
    BLOCK_M: tl.constexpr,
    C_BLOCK: tl.constexpr,
    H_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, C_BLOCK)
    hids = tl.arange(0, H_BLOCK)

    row_mask = rows < M
    c_mask = cols < C
    h_mask = hids < H

    x_ptrs = x_ptr + rows[:, None] * C + cols[None, :]
    x_in = tl.load(x_ptrs, mask=row_mask[:, None] & c_mask[None, :], other=0.0)
    x = x_in.to(tl.float32)

    cnt = tl.sum(c_mask.to(tl.float32))
    x_masked = tl.where(c_mask[None, :], x, 0.0)
    mean = tl.sum(x_masked, axis=1) / cnt
    xc = tl.where(c_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / cnt
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g2_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    b = tl.load(b2n_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    xn = xc * rstd[:, None] * g[None, :] + b[None, :]

    xn_fp16 = xn.to(tl.float16)

    w1_ptrs = w1_ptr + hids[:, None] * C + cols[None, :]
    w1 = tl.load(w1_ptrs, mask=h_mask[:, None] & c_mask[None, :], other=0.0)
    w1_t = tl.trans(w1)

    hidden = tl.dot(xn_fp16, w1_t, out_dtype=tl.float32)
    b1 = tl.load(b1_ptr + hids, mask=h_mask, other=0.0).to(tl.float32)
    hidden = hidden + b1[None, :]
    hidden = 0.5 * hidden * (1.0 + tl.erf(hidden * 0.7071067811865476))
    hidden = tl.where(h_mask[None, :], hidden, 0.0)
    hidden_fp16 = hidden.to(tl.float16)

    w2_ptrs = w2_ptr + cols[:, None] * H + hids[None, :]
    w2 = tl.load(w2_ptrs, mask=c_mask[:, None] & h_mask[None, :], other=0.0)
    w2_t = tl.trans(w2)

    out = tl.dot(hidden_fp16, w2_t, out_dtype=tl.float32)
    bf2 = tl.load(bf2_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    out = out + bf2[None, :]

    out = out + x_in.to(tl.float32)
    out_fp16 = out.to(tl.float16)

    out_ptrs = out_ptr + rows[:, None] * C + cols[None, :]
    tl.store(out_ptrs, out_fp16, mask=row_mask[:, None] & c_mask[None, :])


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

    def _fused_attn_residual(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        assert C == self.dim
        x = x.contiguous()
        out = torch.empty_like(x)

        num_windows = B * nH * nW
        rpe_contig = self.rpe_bias.contiguous().to(torch.float16)

        grid = (num_windows,)
        fused_attn_kernel[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            self.proj.weight, self.proj.bias,
            rpe_contig,
            H, W, nH, nW, Wh, Ww,
            float(self.norm1.eps),
            BLOCK_N=N,
            C=C,
        )
        return out

    def _fused_mlp_residual(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        C = self.dim
        Hd = self.hidden
        x_flat = x.reshape(-1, C).contiguous()
        M = x_flat.shape[0]
        out = torch.empty_like(x_flat)

        BLOCK_M = 64
        C_BLOCK = 32
        H_BLOCK = 128

        grid = (triton.cdiv(M, BLOCK_M),)
        fused_ln_mlp_kernel[grid](
            x_flat, self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            out,
            M, C, Hd,
            float(self.norm2.eps),
            BLOCK_M=BLOCK_M, C_BLOCK=C_BLOCK, H_BLOCK=H_BLOCK,
        )
        return out.view(orig_shape)

    def forward(self, x: Tensor) -> Tensor:
        x = self._fused_attn_residual(x)
        x = self._fused_mlp_residual(x)
        return x