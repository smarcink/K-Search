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
    x_ptr, ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr, y_ptr,
    H, W, nW,
    BLOCK_WIN: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    C3: tl.constexpr,
):
    pid = tl.program_id(0)
    win_id = pid  # window index, total = B*nH*nW

    nH_nW = (H // 4) * nW
    b = win_id // nH_nW
    hw = win_id % nH_nW
    ih = hw // nW
    iw = hw % nW

    # Each window is 4x4 at position (ih*4, iw*4)
    # Row offsets within window: for n in 0..15, r = n//4, c = n%4
    n_idx = tl.arange(0, N)  # 16
    row_in_win = n_idx // 4
    col_in_win = n_idx % 4
    h_idx = ih * 4 + row_in_win
    w_idx = iw * 4 + col_in_win
    # base offset for (b, h, w, :)
    base = ((b * H + h_idx) * W + w_idx) * C  # [N]

    c_idx = tl.arange(0, C)  # 32
    # Load x: [N, C]
    x_offs = base[:, None] + c_idx[None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)

    # LayerNorm1
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln1_w = tl.load(ln1_w_ptr + c_idx).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_idx).to(tl.float32)
    xn = xc * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]  # [N, C]

    # QKV: [N, C] @ [C, 3C] + [3C]
    xn_h = xn.to(tl.float16)
    c3_idx = tl.arange(0, C3)  # 96
    qkv_w = tl.load(qkv_w_ptr + c_idx[:, None] * C3 + c3_idx[None, :])  # [C, 3C]
    qkv_b = tl.load(qkv_b_ptr + c3_idx).to(tl.float32)
    qkv = tl.dot(xn_h, qkv_w).to(tl.float32) + qkv_b[None, :]  # [N, 3C]

    # Split q, k, v
    q = tl.where(c3_idx[None, :] < C, qkv, 0.0)
    # extract q [N, C]
    # Use a different approach: index ranges
    # q: cols [0, C), k: [C, 2C), v: [2C, 3C)
    # Build masks
    is_q = c3_idx < C
    is_k = (c3_idx >= C) & (c3_idx < 2 * C)
    is_v = c3_idx >= 2 * C

    # Sum-reduce trick won't work; instead reload from qkv into 3 [N,C] tensors via masking + shifting
    # Simpler: use tl.reshape on qkv [N, C3] -> can't easily split
    # Use gather via offsets:
    q_cols = tl.arange(0, C)
    k_cols = tl.arange(0, C) + C
    v_cols = tl.arange(0, C) + 2 * C
    # Recompute via dot products on separate slices: do 3 dots
    qkv_w_q = tl.load(qkv_w_ptr + c_idx[:, None] * C3 + q_cols[None, :])
    qkv_w_k = tl.load(qkv_w_ptr + c_idx[:, None] * C3 + k_cols[None, :])
    qkv_w_v = tl.load(qkv_w_ptr + c_idx[:, None] * C3 + v_cols[None, :])
    qkv_b_q = tl.load(qkv_b_ptr + q_cols).to(tl.float32)
    qkv_b_k = tl.load(qkv_b_ptr + k_cols).to(tl.float32)
    qkv_b_v = tl.load(qkv_b_ptr + v_cols).to(tl.float32)
    q = tl.dot(xn_h, qkv_w_q).to(tl.float32) + qkv_b_q[None, :]  # [N, C]
    k = tl.dot(xn_h, qkv_w_k).to(tl.float32) + qkv_b_k[None, :]
    v = tl.dot(xn_h, qkv_w_v).to(tl.float32) + qkv_b_v[None, :]

    # Attention: q @ k^T * scale
    scale = 1.0 / tl.sqrt(float(C))
    q_h = (q * scale).to(tl.float16)
    k_t_h = tl.trans(k.to(tl.float16))  # [C, N]
    attn = tl.dot(q_h, k_t_h).to(tl.float32)  # [N, N]

    # Add rpe bias
    rpe = tl.load(rpe_ptr + n_idx[:, None] * N + n_idx[None, :]).to(tl.float32)
    attn = attn + rpe

    # Softmax
    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]

    # attn @ v
    out = tl.dot(attn_sm.to(tl.float16), v.to(tl.float16)).to(tl.float32)  # [N, C]

    # Proj
    proj_w = tl.load(proj_w_ptr + c_idx[:, None] * C + c_idx[None, :])  # [C, C]
    proj_b = tl.load(proj_b_ptr + c_idx).to(tl.float32)
    out_h = out.to(tl.float16)
    out = tl.dot(out_h, proj_w).to(tl.float32) + proj_b[None, :]  # [N, C]

    # Residual: out + x_original
    y = out + x
    tl.store(y_ptr + x_offs, y.to(tl.float16))


@triton.jit
def fused_mlp_kernel(
    y1_ptr, ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    out_ptr,
    M,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    HID: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    mask = row_idx < M

    c_idx = tl.arange(0, C)
    h_idx = tl.arange(0, HID)

    # Load y1: [BLOCK_M, C]
    offs = row_idx[:, None] * C + c_idx[None, :]
    y1 = tl.load(y1_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)

    # LN2
    mean = tl.sum(y1, axis=1) / C
    xc = y1 - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln2_w = tl.load(ln2_w_ptr + c_idx).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + c_idx).to(tl.float32)
    xn = xc * rstd[:, None] * ln2_w[None, :] + ln2_b[None, :]

    # FC1: [BLOCK_M, C] @ [C, HID] + [HID]
    fc1_w = tl.load(fc1_w_ptr + c_idx[:, None] * HID + h_idx[None, :])
    fc1_b = tl.load(fc1_b_ptr + h_idx).to(tl.float32)
    h = tl.dot(xn.to(tl.float16), fc1_w).to(tl.float32) + fc1_b[None, :]

    # GELU (exact, matches nn.GELU default)
    h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))

    # FC2: [BLOCK_M, HID] @ [HID, C] + [C]
    fc2_w = tl.load(fc2_w_ptr + h_idx[:, None] * C + c_idx[None, :])
    fc2_b = tl.load(fc2_b_ptr + c_idx).to(tl.float32)
    out = tl.dot(h.to(tl.float16), fc2_w).to(tl.float32) + fc2_b[None, :]

    # Residual
    out = out + y1
    tl.store(out_ptr + offs, out.to(tl.float16), mask=mask[:, None])


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

        # Attention path
        y1 = torch.empty_like(x)
        num_windows = B * nH * nW

        qkv_w_t = self.qkv.weight.t().contiguous()  # [C, 3C]
        proj_w_t = self.proj.weight.t().contiguous()  # [C, C]

        fused_attn_kernel[(num_windows,)](
            x, self.norm1.weight, self.norm1.bias,
            qkv_w_t, self.qkv.bias,
            proj_w_t, self.proj.bias,
            self.rpe_bias, y1,
            H, W, nW,
            BLOCK_WIN=1,
            N=N, C=C, C3=3 * C,
        )

        # MLP path - fused
        M = B * H * W
        BLOCK_M = 64
        y1_flat = y1.view(M, C)
        out = torch.empty_like(y1_flat)

        fc1_w_t = self.fc1.weight.t().contiguous()  # [C, HID]
        fc2_w_t = self.fc2.weight.t().contiguous()  # [HID, C]

        grid = (triton.cdiv(M, BLOCK_M),)
        fused_mlp_kernel[grid](
            y1_flat, self.norm2.weight, self.norm2.bias,
            fc1_w_t, self.fc1.bias,
            fc2_w_t, self.fc2.bias,
            out,
            M,
            BLOCK_M=BLOCK_M,
            C=C, HID=self.hidden,
        )

        return out.view(B, H, W, C)