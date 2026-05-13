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
def fused_attn_kernel(
    X_ptr, OUT_ptr,
    ln1_w_ptr, ln1_b_ptr,
    q_w_ptr, q_b_ptr,
    k_w_ptr, k_b_ptr,
    v_w_ptr, v_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    nH, nW,
    H, W,
    Wh: tl.constexpr, Ww: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    nw_total = nH * nW
    b = pid // nw_total
    win = pid % nw_total
    ih = win // nW
    iw = win % nW

    row_idx = tl.arange(0, N)
    tok_row = row_idx // Ww
    tok_col = row_idx % Ww
    h_idx = ih * Wh + tok_row
    w_idx = iw * Ww + tok_col

    c_idx = tl.arange(0, BLOCK_C)
    mask_c = c_idx < C

    base = (b * H + h_idx) * W + w_idx
    x_offs = base[:, None] * C + c_idx[None, :]
    x = tl.load(X_ptr + x_offs, mask=mask_c[None, :], other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=1, keep_dims=True) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=1, keep_dims=True) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    ln1_w = tl.load(ln1_w_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    xn = xc * rstd * ln1_w[None, :] + ln1_b[None, :]
    xn_h = xn.to(tl.float16)

    # Load Q, K, V weights separately
    w_offs = c_idx[:, None] * C + c_idx[None, :]
    w_mask = mask_c[:, None] & mask_c[None, :]
    q_w = tl.load(q_w_ptr + w_offs, mask=w_mask, other=0.0)
    k_w = tl.load(k_w_ptr + w_offs, mask=w_mask, other=0.0)
    v_w = tl.load(v_w_ptr + w_offs, mask=w_mask, other=0.0)

    q_b = tl.load(q_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    k_b = tl.load(k_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    v_b = tl.load(v_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)

    q = tl.dot(xn_h, q_w, out_dtype=tl.float32) + q_b[None, :]
    k = tl.dot(xn_h, k_w, out_dtype=tl.float32) + k_b[None, :]
    v = tl.dot(xn_h, v_w, out_dtype=tl.float32) + v_b[None, :]

    scale = 1.0 / tl.sqrt(tl.full([1], C, tl.float32))
    q = q * scale

    q_h = q.to(tl.float16)
    k_h = k.to(tl.float16)
    kt = tl.trans(k_h)
    attn = tl.dot(q_h, kt, out_dtype=tl.float32)

    n_i = tl.arange(0, N)
    n_j = tl.arange(0, N)
    rpe_offs = n_i[:, None] * N + n_j[None, :]
    rpe = tl.load(rpe_ptr + rpe_offs).to(tl.float32)
    attn = attn + rpe

    amax = tl.max(attn, axis=1, keep_dims=True)
    attn = tl.exp(attn - amax)
    asum = tl.sum(attn, axis=1, keep_dims=True)
    attn = attn / asum

    attn_h = attn.to(tl.float16)
    v_h = v.to(tl.float16)
    av = tl.dot(attn_h, v_h, out_dtype=tl.float32)

    av_h = av.to(tl.float16)
    proj_w = tl.load(proj_w_ptr + w_offs, mask=w_mask, other=0.0)
    proj_b = tl.load(proj_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    out = tl.dot(av_h, proj_w, out_dtype=tl.float32) + proj_b[None, :]

    x_orig = tl.load(X_ptr + x_offs, mask=mask_c[None, :], other=0.0).to(tl.float32)
    out = out + x_orig

    tl.store(OUT_ptr + x_offs, out.to(tl.float16), mask=mask_c[None, :])


class ModelNew(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1
        assert tuple(shift_size) == (0, 0)
        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = list(window_size)

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

    def _prepare_cache(self):
        C = self.dim
        # qkv.weight is (3C, C); rows 0:C = q, C:2C = k, 2C:3C = v
        W = self.qkv.weight  # (3C, C)
        q_w = W[0:C, :].t().contiguous()      # (C, C)
        k_w = W[C:2*C, :].t().contiguous()    # (C, C)
        v_w = W[2*C:3*C, :].t().contiguous()  # (C, C)
        B = self.qkv.bias
        q_b = B[0:C].contiguous()
        k_b = B[C:2*C].contiguous()
        v_b = B[2*C:3*C].contiguous()
        proj_w = self.proj.weight.t().contiguous()  # (C, C)
        self._q_w = q_w
        self._k_w = k_w
        self._v_w = v_w
        self._q_b = q_b
        self._k_b = k_b
        self._v_b = v_b
        self._proj_w = proj_w
        self._cached = True

    def _attn_fused(self, x: Tensor) -> Tensor:
        if not self._cached:
            self._prepare_cache()

        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        N = Wh * Ww
        nH = H // Wh
        nW = W // Ww

        out = torch.empty_like(x)
        BLOCK_C = max(16, triton.next_power_of_2(C))

        grid = (B * nH * nW,)
        fused_attn_kernel[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            self._q_w, self._q_b,
            self._k_w, self._k_b,
            self._v_w, self._v_b,
            self._proj_w, self.proj.bias,
            self.rpe_bias,
            nH, nW,
            H, W,
            Wh=Wh, Ww=Ww,
            C=C,
            N=N,
            BLOCK_C=BLOCK_C,
            eps=1e-5,
            num_warps=2,
        )
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = self._attn_fused(x)
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x