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
def _swin_attn_stage_kernel(
    x_ptr, y_ptr,
    ln_w_ptr, ln_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    wh = pid // 320
    ww = pid - wh * 320

    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    local_h = offs_n // 4
    local_w = offs_n - local_h * 4

    token_base = ((wh * 4 + local_h) * 1280 + (ww * 4 + local_w)) * 32
    x_offsets = token_base[:, None] + offs_c[None, :]

    x = tl.load(x_ptr + x_offsets).to(tl.float32)

    mean = tl.sum(x, axis=1) * 0.03125
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * 0.03125
    inv = tl.rsqrt(var + 1.0e-5)

    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    xn = (xc * inv[:, None]) * ln_w[None, :] + ln_b[None, :]
    xn_h = xn.to(tl.float16)

    koffs = tl.arange(0, BLOCK_C)
    ooffs = tl.arange(0, BLOCK_C)

    wq = tl.load(qkv_w_ptr + ooffs[None, :] * 32 + koffs[:, None])
    wk = tl.load(qkv_w_ptr + (32 + ooffs[None, :]) * 32 + koffs[:, None])
    wv = tl.load(qkv_w_ptr + (64 + ooffs[None, :]) * 32 + koffs[:, None])

    bq = tl.load(qkv_b_ptr + ooffs).to(tl.float32)
    bk = tl.load(qkv_b_ptr + 32 + ooffs).to(tl.float32)
    bv = tl.load(qkv_b_ptr + 64 + ooffs).to(tl.float32)

    q = (tl.dot(xn_h, wq) + bq[None, :]).to(tl.float16)
    k = (tl.dot(xn_h, wk) + bk[None, :]).to(tl.float16)
    v = (tl.dot(xn_h, wv) + bv[None, :]).to(tl.float16)

    qs = (q.to(tl.float32) * 0.1767766952966369).to(tl.float16)
    scores = tl.dot(qs, tl.trans(k)).to(tl.float32)

    rpe_offsets = offs_n[:, None] * 16 + offs_n[None, :]
    rpe = tl.load(rpe_ptr + rpe_offsets).to(tl.float32)
    scores = scores + rpe

    m = tl.max(scores, axis=1)
    e = tl.exp(scores - m[:, None])
    denom = tl.sum(e, axis=1)
    attn = e / denom[:, None]

    av = tl.dot(attn.to(tl.float16), v)

    wp = tl.load(proj_w_ptr + ooffs[None, :] * 32 + koffs[:, None])
    bp = tl.load(proj_b_ptr + ooffs).to(tl.float32)
    p = tl.dot(av.to(tl.float16), wp) + bp[None, :]

    out = p.to(tl.float16) + x.to(tl.float16)
    tl.store(y_ptr + x_offsets, out)


@triton.jit
def _swin_mlp_fc1_gelu_kernel(
    y_ptr, tmp_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    y_offsets = offs_m[:, None] * 32 + offs_c[None, :]
    y = tl.load(y_ptr + y_offsets).to(tl.float32)

    mean = tl.sum(y, axis=1) * 0.03125
    yc = y - mean[:, None]
    var = tl.sum(yc * yc, axis=1) * 0.03125
    inv = tl.rsqrt(var + 1.0e-5)

    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    yn = (yc * inv[:, None]) * ln_w[None, :] + ln_b[None, :]
    yn_h = yn.to(tl.float16)

    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * 32 + offs_c[:, None])
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)

    h = tl.dot(yn_h, w1) + b1[None, :]
    h_half = h.to(tl.float16)
    hf = h_half.to(tl.float32)
    gelu = 0.5 * hf * (1.0 + tl.erf(hf * 0.7071067811865476))

    tmp_offsets = offs_m[:, None] * 128 + offs_h[None, :]
    tl.store(tmp_ptr + tmp_offsets, gelu.to(tl.float16))


@triton.jit
def _swin_mlp_fc2_kernel(
    y_ptr, tmp_ptr, out_ptr,
    fc2_w_ptr, fc2_b_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)

    h0 = tl.load(tmp_ptr + offs_m[:, None] * 128 + offs_h[None, :])
    w0 = tl.load(fc2_w_ptr + offs_c[None, :] * 128 + offs_h[:, None])
    acc += tl.dot(h0, w0)

    h1 = tl.load(tmp_ptr + offs_m[:, None] * 128 + (32 + offs_h)[None, :])
    w1 = tl.load(fc2_w_ptr + offs_c[None, :] * 128 + (32 + offs_h)[:, None])
    acc += tl.dot(h1, w1)

    h2 = tl.load(tmp_ptr + offs_m[:, None] * 128 + (64 + offs_h)[None, :])
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * 128 + (64 + offs_h)[:, None])
    acc += tl.dot(h2, w2)

    h3 = tl.load(tmp_ptr + offs_m[:, None] * 128 + (96 + offs_h)[None, :])
    w3 = tl.load(fc2_w_ptr + offs_c[None, :] * 128 + (96 + offs_h)[:, None])
    acc += tl.dot(h3, w3)

    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    z = acc + b2[None, :]

    y = tl.load(y_ptr + offs_m[:, None] * 32 + offs_c[None, :])
    out = z.to(tl.float16) + y

    tl.store(out_ptr + offs_m[:, None] * 32 + offs_c[None, :], out)


class ModelNew(nn.Module):
    """Specialized Triton implementation for the fixed simplified Swin block."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1, "simplified path: single-head only"
        assert tuple(shift_size) == (0, 0), "simplified path: no shift"
        assert dim == 32, "specialized Triton path expects dim=32"
        assert tuple(window_size) == (4, 4), "specialized Triton path expects 4x4 windows"

        hidden = int(dim * mlp_ratio)
        assert hidden == 128, "specialized Triton path expects hidden=128"

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
        attn = (q * (C ** -0.5)) @ k.transpose(-2, -1) + self.rpe_bias
        attn = attn.softmax(dim=-1)
        xw = (attn @ v).reshape(-1, N, C)
        xw = self.proj(xw)
        xw = xw.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return xw

    def _forward_fallback(self, x: Tensor) -> Tensor:
        x = x + self._attn_fallback(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x

    def forward(self, x: Tensor) -> Tensor:
        if (
            x.is_xpu
            and x.dtype == torch.float16
            and x.dim() == 4
            and x.shape[0] == 1
            and x.shape[1] == 720
            and x.shape[2] == 1280
            and x.shape[3] == 32
            and x.is_contiguous()
        ):
            y = torch.empty_like(x)
            tmp = torch.empty((921600, 128), device=x.device, dtype=torch.float16)
            out = torch.empty_like(x)

            _swin_attn_stage_kernel[(57600,)](
                x, y,
                self.norm1.weight, self.norm1.bias,
                self.qkv.weight, self.qkv.bias,
                self.proj.weight, self.proj.bias,
                self.rpe_bias,
                BLOCK_N=16,
                BLOCK_C=32,
                num_warps=4,
            )

            _swin_mlp_fc1_gelu_kernel[(57600, 4)](
                y, tmp,
                self.norm2.weight, self.norm2.bias,
                self.fc1.weight, self.fc1.bias,
                BLOCK_M=16,
                BLOCK_C=32,
                BLOCK_H=32,
                num_warps=4,
            )

            _swin_mlp_fc2_kernel[(57600,)](
                y, tmp, out,
                self.fc2.weight, self.fc2.bias,
                BLOCK_M=16,
                BLOCK_C=32,
                BLOCK_H=32,
                num_warps=4,
            )

            return out

        return self._forward_fallback(x)


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