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
def _ln_qkv_window_kernel(
    x_ptr, gamma_ptr, beta_ptr, w_ptr, b_ptr, qkv_ptr,
    W, nW, total_tokens,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < total_tokens
    safe_m = tl.where(mask_m, offs_m, 0)
    mask_n = offs_n < 96
    safe_n = tl.where(mask_n, offs_n, 0)

    x_vals = tl.load(
        x_ptr + safe_m[:, None] * 32 + offs_c[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x_vals, axis=1) * 0.03125
    xc = x_vals - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * 0.03125
    inv = tl.rsqrt(var + 1.0e-5)

    gamma = tl.load(gamma_ptr + offs_c).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c).to(tl.float32)
    norm = (xc * inv[:, None]) * gamma[None, :] + beta[None, :]
    norm_h = norm.to(tl.float16)

    ww = tl.load(
        w_ptr + safe_n[None, :] * 32 + offs_c[:, None],
        mask=mask_n[None, :],
        other=0.0,
    )
    acc = tl.dot(norm_h, ww)
    bias = tl.load(b_ptr + safe_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    h = safe_m // W
    widx = safe_m - h * W
    win = (h // 4) * nW + (widx // 4)
    in_win = (h - (h // 4) * 4) * 4 + (widx - (widx // 4) * 4)

    out_offs = (win[:, None] * 16 + in_win[:, None]) * 96 + safe_n[None, :]
    tl.store(
        qkv_ptr + out_offs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _window_attn_proj_resid_kernel(
    x_ptr, qkv_ptr, rpe_ptr, proj_w_ptr, proj_b_ptr, x1_ptr,
    W, nW,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    win = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    base = win * 16 * 96
    q = tl.load(qkv_ptr + base + offs_n[:, None] * 96 + offs_c[None, :])
    k = tl.load(qkv_ptr + base + offs_n[:, None] * 96 + 32 + offs_c[None, :])
    v = tl.load(qkv_ptr + base + offs_n[:, None] * 96 + 64 + offs_c[None, :])

    qs = (q.to(tl.float32) * 0.1767766952966369).to(tl.float16)
    scores = tl.dot(qs, tl.trans(k))
    rpe = tl.load(rpe_ptr + offs_n[:, None] * 16 + offs_n[None, :]).to(tl.float32)
    scores = scores + rpe

    m = tl.max(scores, axis=1)
    e = tl.exp(scores - m[:, None])
    d = tl.sum(e, axis=1)
    p = (e / d[:, None]).to(tl.float16)

    ctx = tl.dot(p, v)
    ctx_h = ctx.to(tl.float16)

    pw = tl.load(proj_w_ptr + offs_c[None, :] * 32 + offs_c[:, None])
    out = tl.dot(ctx_h, pw)
    pb = tl.load(proj_b_ptr + offs_c).to(tl.float32)
    out = out + pb[None, :]

    wh = win // nW
    ww = win - wh * nW
    lh = offs_n // 4
    lw = offs_n - lh * 4
    tok = ((wh * 4 + lh) * W + (ww * 4 + lw)) * 32

    resid = tl.load(x_ptr + tok[:, None] + offs_c[None, :]).to(tl.float32)
    tl.store(x1_ptr + tok[:, None] + offs_c[None, :], out + resid)


@triton.jit
def _ln_fc1_kernel(
    x_ptr, gamma_ptr, beta_ptr, w_ptr, b_ptr, hidden_ptr,
    total_tokens,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < total_tokens
    safe_m = tl.where(mask_m, offs_m, 0)

    x_vals = tl.load(
        x_ptr + safe_m[:, None] * 32 + offs_c[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    mean = tl.sum(x_vals, axis=1) * 0.03125
    xc = x_vals - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * 0.03125
    inv = tl.rsqrt(var + 1.0e-5)

    gamma = tl.load(gamma_ptr + offs_c).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c).to(tl.float32)
    norm = (xc * inv[:, None]) * gamma[None, :] + beta[None, :]
    norm_h = norm.to(tl.float16)

    ww = tl.load(w_ptr + offs_n[None, :] * 32 + offs_c[:, None])
    acc = tl.dot(norm_h, ww)
    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    acc = acc + bias[None, :]

    tl.store(
        hidden_ptr + safe_m[:, None] * 128 + offs_n[None, :],
        acc,
        mask=mask_m[:, None],
    )


@triton.jit
def _gelu_fc2_resid_kernel(
    hidden_ptr, x1_ptr, w_ptr, b_ptr, out_ptr,
    total_tokens,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < total_tokens
    safe_m = tl.where(mask_m, offs_m, 0)

    h = tl.load(
        hidden_ptr + safe_m[:, None] * 128 + offs_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    gelu = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    gelu_h = gelu.to(tl.float16)

    ww = tl.load(w_ptr + offs_n[None, :] * 128 + offs_k[:, None])
    acc = tl.dot(gelu_h, ww)
    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    acc = acc + bias[None, :]

    resid = tl.load(
        x1_ptr + safe_m[:, None] * 32 + offs_n[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    acc = acc + resid

    tl.store(
        out_ptr + safe_m[:, None] * 32 + offs_n[None, :],
        acc,
        mask=mask_m[:, None],
    )


class ModelNew(nn.Module):
    """Simplified Swin block optimized with Triton kernels for Intel XPU."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1, "simplified path: single-head only"
        assert tuple(shift_size) == (0, 0), "simplified path: no shift"
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

    def _attn_fallback(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        xw = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)
        qkv = self.qkv(xw).reshape(-1, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * (C ** -0.5)) @ k.transpose(-2, -1) + self.rpe_bias[0, 0]
        attn = attn.softmax(dim=-1)
        xw = (attn @ v).reshape(-1, N, C)
        xw = self.proj(xw)
        xw = xw.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return xw

    def forward(self, x: Tensor) -> Tensor:
        if (
            x.device.type != "xpu"
            or x.dim() != 4
            or x.shape[0] != 1
            or x.shape[-1] != 32
            or self.dim != 32
            or self.window_size[0] != 4
            or self.window_size[1] != 4
            or self.fc1.weight.shape[0] != 128
        ):
            x = x + self._attn_fallback(self.norm1(x))
            x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
            return x

        if not x.is_contiguous():
            x = x.contiguous()

        B, H, W, C = x.shape
        total = H * W
        nW = W // 4
        num_windows = (H // 4) * nW

        qkv_buf = torch.empty((num_windows, 16, 96), device=x.device, dtype=x.dtype)
        x1 = torch.empty_like(x)
        hidden = torch.empty((total, 128), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)

        grid_tokens = (triton.cdiv(total, 16),)

        _ln_qkv_window_kernel[grid_tokens](
            x, self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias, qkv_buf,
            W, nW, total,
            BLOCK_M=16, BLOCK_C=32, BLOCK_N=128,
            num_warps=4,
        )

        _window_attn_proj_resid_kernel[(num_windows,)](
            x, qkv_buf, self.rpe_bias,
            self.proj.weight, self.proj.bias, x1,
            W, nW,
            BLOCK_N=16, BLOCK_C=32,
            num_warps=4,
        )

        _ln_fc1_kernel[grid_tokens](
            x1, self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias, hidden,
            total,
            BLOCK_M=16, BLOCK_C=32, BLOCK_N=128,
            num_warps=4,
        )

        _gelu_fc2_resid_kernel[grid_tokens](
            hidden, x1,
            self.fc2.weight, self.fc2.bias, out,
            total,
            BLOCK_M=16, BLOCK_K=128, BLOCK_N=32,
            num_warps=4,
        )

        return out


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16)]


def get_init_inputs():
    return [32, 1, [4, 4], [0, 0], 4]


def state_dict_remap(src_state):
    rename = {
        "attn.qkv.weight":  "qkv.weight",
        "attn.qkv.bias":    "qkv.bias",
        "attn.proj.weight": "proj.weight",
        "attn.proj.bias":   "proj.bias",
        "mlp.0.weight":     "fc1.weight",
        "mlp.0.bias":       "fc1.bias",
        "mlp.3.weight":     "fc2.weight",
        "mlp.3.bias":       "fc2.bias",
    }
    drop_prefix = (
        "attn.relative_position_bias_table",
        "attn.relative_position_index",
        "mlp.4.",
    )
    out = {}
    for k, v in src_state.items():
        if any(k.startswith(p) for p in drop_prefix):
            continue
        out[rename.get(k, k)] = v

    table = src_state.get("attn.relative_position_bias_table")
    index = src_state.get("attn.relative_position_index")
    if table is not None and index is not None:
        num_heads = table.shape[-1]
        N = int(index.numel() ** 0.5)
        bias = table[index.long()].view(N, N, num_heads).permute(2, 0, 1)
        out["rpe_bias"] = bias.unsqueeze(0).contiguous().to(table.dtype)
    return out