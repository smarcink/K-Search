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
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    nH, nW, Wh: tl.constexpr, Ww: tl.constexpr,
    H, W, C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_3C: tl.constexpr,
    eps: tl.constexpr,
):
    # One program per window
    pid = tl.program_id(0)
    nw_total = nH * nW
    b = pid // nw_total
    win = pid % nw_total
    ih = win // nW
    iw = win % nW

    # Token indices in window: row = i // Ww, col = i % Ww (i in [0,N))
    row_idx = tl.arange(0, N)
    tok_row = row_idx // Ww  # 0..Wh-1
    tok_col = row_idx % Ww   # 0..Ww-1
    h_idx = ih * Wh + tok_row  # [N]
    w_idx = iw * Ww + tok_col  # [N]

    c_idx = tl.arange(0, BLOCK_C)

    # Compute offsets to load X tile [N, C]
    # X is (B, H, W, C) contiguous; offset = ((b*H + h)*W + w)*C + c
    base = (b * H + h_idx) * W + w_idx  # [N]
    x_offs = base[:, None] * C + c_idx[None, :]
    mask_c = c_idx < C
    x = tl.load(X_ptr + x_offs, mask=mask_c[None, :], other=0.0).to(tl.float32)  # [N, C]

    # LayerNorm over C
    mean = tl.sum(x, axis=1, keep_dims=True) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=1, keep_dims=True) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    ln1_w = tl.load(ln1_w_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    xn = xc * rstd * ln1_w[None, :] + ln1_b[None, :]  # [N, C] fp32

    xn_h = xn.to(tl.float16)

    # QKV: [N, C] @ [C, 3C] -> [N, 3C]
    c3_idx = tl.arange(0, BLOCK_3C)
    mask_3c = c3_idx < 3 * C
    qkv_w_offs = c_idx[:, None] * (3 * C) + c3_idx[None, :]
    qkv_w = tl.load(qkv_w_ptr + qkv_w_offs, mask=mask_c[:, None] & mask_3c[None, :], other=0.0)
    qkv_b = tl.load(qkv_b_ptr + c3_idx, mask=mask_3c, other=0.0).to(tl.float32)
    qkv = tl.dot(xn_h, qkv_w, out_dtype=tl.float32) + qkv_b[None, :]  # [N, 3C]

    # Split q, k, v
    q_mask = c3_idx < C
    k_mask = (c3_idx >= C) & (c3_idx < 2 * C)
    v_mask = (c3_idx >= 2 * C) & (c3_idx < 3 * C)

    # Need [N, C] for each. Use sum trick: take subranges.
    # Simpler: rebuild q, k, v as separate [N, BLOCK_C] tiles via reshape.
    # Since BLOCK_3C = 3*BLOCK_C and we want first/second/third third.
    # Use tl.reshape: qkv [N, 3*BLOCK_C] -> [N, 3, BLOCK_C]
    qkv_r = tl.reshape(qkv, (N, 3, BLOCK_C))
    q = tl.reshape(tl.sum(qkv_r * tl.where(tl.arange(0, 3)[None, :, None] == 0, 1.0, 0.0), axis=1), (N, BLOCK_C))
    k = tl.reshape(tl.sum(qkv_r * tl.where(tl.arange(0, 3)[None, :, None] == 1, 1.0, 0.0), axis=1), (N, BLOCK_C))
    v = tl.reshape(tl.sum(qkv_r * tl.where(tl.arange(0, 3)[None, :, None] == 2, 1.0, 0.0), axis=1), (N, BLOCK_C))

    scale = 1.0 / tl.sqrt(tl.full([1], C, tl.float32))
    q = q * scale

    # attn = q @ k^T  [N, C] @ [C, N] -> [N, N]
    q_h = q.to(tl.float16)
    k_h = k.to(tl.float16)
    kt = tl.trans(k_h)  # [C, N]
    attn = tl.dot(q_h, kt, out_dtype=tl.float32)  # [N, N]

    # Add rpe bias [N, N]
    n_i = tl.arange(0, N)
    n_j = tl.arange(0, N)
    rpe_offs = n_i[:, None] * N + n_j[None, :]
    rpe = tl.load(rpe_ptr + rpe_offs).to(tl.float32)
    attn = attn + rpe

    # Softmax over last dim
    amax = tl.max(attn, axis=1, keep_dims=True)
    attn = tl.exp(attn - amax)
    asum = tl.sum(attn, axis=1, keep_dims=True)
    attn = attn / asum

    # attn @ v -> [N, C]
    attn_h = attn.to(tl.float16)
    v_h = v.to(tl.float16)
    av = tl.dot(attn_h, v_h, out_dtype=tl.float32)  # [N, C]

    # proj: [N, C] @ [C, C] -> [N, C]
    av_h = av.to(tl.float16)
    proj_w_offs = c_idx[:, None] * C + c_idx[None, :]
    proj_w = tl.load(proj_w_ptr + proj_w_offs, mask=mask_c[:, None] & mask_c[None, :], other=0.0)
    proj_b = tl.load(proj_b_ptr + c_idx, mask=mask_c, other=0.0).to(tl.float32)
    out = tl.dot(av_h, proj_w, out_dtype=tl.float32) + proj_b[None, :]  # [N, C]

    # Residual: load original x
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

    def _attn_fused(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        N = Wh * Ww
        nH = H // Wh
        nW = W // Ww

        out = torch.empty_like(x)

        # Pad C to power of 2 for tile sizes
        BLOCK_C = max(16, triton.next_power_of_2(C))
        BLOCK_3C = max(16, triton.next_power_of_2(3 * C))

        # qkv weight in nn.Linear is (out, in) = (3C, C); we want (C, 3C) so transpose
        qkv_w = self.qkv.weight.t().contiguous()  # (C, 3C)
        proj_w = self.proj.weight.t().contiguous()  # (C, C)

        grid = (B * nH * nW,)
        fused_attn_kernel[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            qkv_w, self.qkv.bias,
            proj_w, self.proj.bias,
            self.rpe_bias,
            nH, nW, Wh, Ww,
            H, W, C,
            N=N,
            BLOCK_C=BLOCK_C,
            BLOCK_3C=BLOCK_3C,
            eps=1e-5,
            num_warps=2,
        )
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = self._attn_fused(x)
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x