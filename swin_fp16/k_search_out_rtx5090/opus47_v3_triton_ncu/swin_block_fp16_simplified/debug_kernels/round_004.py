import torch
from torch import nn, Tensor
import triton
import triton.language as tl
import math


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
def attn_block_kernel(
    X_ptr, Y_ptr,
    ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    num_windows,
    C: tl.constexpr, N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_start = pid * BLOCK_W

    c_range = tl.arange(0, C)
    n_range = tl.arange(0, N)

    ln1_w = tl.load(ln1_w_ptr + c_range)
    ln1_b = tl.load(ln1_b_ptr + c_range)

    q_w = tl.load(qkv_w_ptr + (c_range[:, None]) * C + c_range[None, :])
    k_w = tl.load(qkv_w_ptr + (c_range[:, None] + C) * C + c_range[None, :])
    v_w = tl.load(qkv_w_ptr + (c_range[:, None] + 2 * C) * C + c_range[None, :])
    q_b = tl.load(qkv_b_ptr + c_range)
    k_b = tl.load(qkv_b_ptr + c_range + C)
    v_b = tl.load(qkv_b_ptr + c_range + 2 * C)

    q_wt = tl.trans(q_w).to(tl.float16)
    k_wt = tl.trans(k_w).to(tl.float16)
    v_wt = tl.trans(v_w).to(tl.float16)

    proj_w = tl.load(proj_w_ptr + c_range[:, None] * C + c_range[None, :])
    proj_b = tl.load(proj_b_ptr + c_range)
    proj_w_t = tl.trans(proj_w).to(tl.float16)

    rpe = tl.load(rpe_ptr + n_range[:, None] * N + n_range[None, :])

    scale = 1.0 / tl.sqrt(tl.full([], C, tl.float32))

    for wi in tl.static_range(0, BLOCK_W):
        w_idx = w_start + wi
        mask = w_idx < num_windows

        x_off = w_idx * N * C + n_range[:, None] * C + c_range[None, :]
        x = tl.load(X_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        x_res = x

        mean = tl.sum(x, axis=1) / C
        xm = x - mean[:, None]
        var = tl.sum(xm * xm, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        x_n = xm * rstd[:, None] * ln1_w[None, :] + ln1_b[None, :]

        x_n_f16 = x_n.to(tl.float16)

        q = tl.dot(x_n_f16, q_wt, out_dtype=tl.float32) + q_b[None, :]
        k = tl.dot(x_n_f16, k_wt, out_dtype=tl.float32) + k_b[None, :]
        v = tl.dot(x_n_f16, v_wt, out_dtype=tl.float32) + v_b[None, :]

        q_s = (q * scale).to(tl.float16)
        k_t = tl.trans(k).to(tl.float16)
        attn = tl.dot(q_s, k_t, out_dtype=tl.float32) + rpe

        attn_max = tl.max(attn, axis=1)
        attn_e = tl.exp(attn - attn_max[:, None])
        attn_s = tl.sum(attn_e, axis=1)
        attn_p = (attn_e / attn_s[:, None]).to(tl.float16)

        v_f16 = v.to(tl.float16)
        out = tl.dot(attn_p, v_f16, out_dtype=tl.float32)

        out_f16 = out.to(tl.float16)
        y = tl.dot(out_f16, proj_w_t, out_dtype=tl.float32) + proj_b[None, :]

        y = y + x_res

        tl.store(Y_ptr + x_off, y.to(tl.float16), mask=mask)


@triton.jit
def mlp_block_kernel(
    X_ptr, Y_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    num_rows,
    C: tl.constexpr, H: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    r_start = pid * BLOCK_R

    c_range = tl.arange(0, C)
    h_range = tl.arange(0, H)
    r_range = tl.arange(0, BLOCK_R)

    ln2_w = tl.load(ln2_w_ptr + c_range)
    ln2_b = tl.load(ln2_b_ptr + c_range)
    fc1_wt = tl.load(fc1_w_ptr + h_range[None, :] * C + c_range[:, None]).to(tl.float16)
    fc1_b = tl.load(fc1_b_ptr + h_range)
    fc2_wt = tl.load(fc2_w_ptr + c_range[None, :] * H + h_range[:, None]).to(tl.float16)
    fc2_b = tl.load(fc2_b_ptr + c_range)

    rows = r_start + r_range
    mask = rows < num_rows

    x_off = rows[:, None] * C + c_range[None, :]
    x = tl.load(X_ptr + x_off, mask=mask[:, None], other=0.0).to(tl.float32)
    x_res = x

    mean = tl.sum(x, axis=1) / C
    xm = x - mean[:, None]
    var = tl.sum(xm * xm, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x_n = xm * rstd[:, None] * ln2_w[None, :] + ln2_b[None, :]

    x_n_f16 = x_n.to(tl.float16)
    h = tl.dot(x_n_f16, fc1_wt, out_dtype=tl.float32) + fc1_b[None, :]

    h_g = 0.5 * h * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (h + 0.044715 * h * h * h)))

    h_g_f16 = h_g.to(tl.float16)
    y = tl.dot(h_g_f16, fc2_wt, out_dtype=tl.float32) + fc2_b[None, :]

    y = y + x_res

    tl.store(Y_ptr + x_off, y.to(tl.float16), mask=mask[:, None])


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


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

        x_win = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, N, C)
        num_windows = x_win.shape[0]

        out_a = torch.empty_like(x_win)
        BLOCK_W = 4
        grid_a = (triton.cdiv(num_windows, BLOCK_W),)
        attn_block_kernel[grid_a](
            x_win, out_a,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            num_windows,
            C=C, N=N, BLOCK_W=BLOCK_W,
            num_warps=2,
        )

        out_a = out_a.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        total_rows = B * H * W
        out_b = torch.empty_like(out_a)
        out_a_flat = out_a.view(total_rows, C)
        out_b_flat = out_b.view(total_rows, C)
        BLOCK_R = 32
        grid_b = (triton.cdiv(total_rows, BLOCK_R),)
        mlp_block_kernel[grid_b](
            out_a_flat, out_b_flat,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            total_rows,
            C=C, H=self.hidden, BLOCK_R=BLOCK_R,
            num_warps=2,
        )

        return out_b