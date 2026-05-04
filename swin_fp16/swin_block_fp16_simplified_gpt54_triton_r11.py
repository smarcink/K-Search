import math
import torch
from torch import nn, Tensor

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


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


class Model(nn.Module):
    """Simplified Swin block: LN -> W-MSA -> residual -> LN -> MLP -> residual."""

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

    def _attn(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)

        qkv = self.qkv(x).reshape(-1, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1) + self.rpe_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(-1, N, C)
        x = self.proj(x)

        x = x.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


if _TRITON_AVAILABLE:
    @triton.jit
    def _erf_approx(x):
        ax = tl.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
        y = 1.0 - p * tl.exp(-ax * ax)
        return tl.where(x >= 0, y, -y)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1, num_stages=1),
            triton.Config({}, num_warps=2, num_stages=1),
            triton.Config({}, num_warps=2, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=1),
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=1),
            triton.Config({}, num_warps=8, num_stages=2),
        ],
        key=["H", "W"],
    )
    @triton.jit
    def _block_kernel(
        x_ptr,
        y_ptr,
        ln1_w_ptr,
        ln1_b_ptr,
        qkv_wt_ptr,
        qkv_b_ptr,
        proj_wt_ptr,
        proj_b_ptr,
        rpe_ptr,
        ln2_w_ptr,
        ln2_b_ptr,
        fc1_wt_ptr,
        fc1_b_ptr,
        fc2_wt_ptr,
        fc2_b_ptr,
        B,
        H,
        W,
        stride_xb,
        stride_xh,
        stride_xw,
        stride_xc,
        stride_yb,
        stride_yh,
        stride_yw,
        stride_yc,
        stride_qkvt0,
        stride_qkvt1,
        stride_projt0,
        stride_projt1,
        stride_r0,
        stride_r1,
        stride_fc1t0,
        stride_fc1t1,
        stride_fc2t0,
        stride_fc2t1,
        EPS: tl.constexpr,
        C: tl.constexpr,
        HIDDEN: tl.constexpr,
        CHUNK: tl.constexpr,
        N: tl.constexpr,
        WS: tl.constexpr,
    ):
        pid = tl.program_id(0)

        nH = H // WS
        nW = W // WS
        wins_per_b = nH * nW
        b = pid // wins_per_b
        rem = pid - b * wins_per_b
        wh = rem // nW
        ww = rem - wh * nW

        tok = tl.arange(0, N)
        c = tl.arange(0, C)
        th = tok // WS
        tw = tok % WS

        x_offs = (
            b * stride_xb
            + (wh * WS + th)[:, None] * stride_xh
            + (ww * WS + tw)[:, None] * stride_xw
            + c[None, :] * stride_xc
        )
        x_h = tl.load(x_ptr + x_offs).to(tl.float16)
        x_f32 = x_h.to(tl.float32)

        ln1_mean = tl.sum(x_f32, axis=1) / C
        ln1_xc = x_f32 - ln1_mean[:, None]
        ln1_var = tl.sum(ln1_xc * ln1_xc, axis=1) / C
        ln1_rstd = tl.rsqrt(ln1_var + EPS)

        ln1_w = tl.load(ln1_w_ptr + c).to(tl.float32)
        ln1_b = tl.load(ln1_b_ptr + c).to(tl.float32)
        ln1 = ln1_xc * ln1_rstd[:, None]
        ln1 = ln1 * ln1_w[None, :] + ln1_b[None, :]
        ln1_h = ln1.to(tl.float16)

        oc = tl.arange(0, C)

        wq = tl.load(qkv_wt_ptr + c[:, None] * stride_qkvt0 + oc[None, :] * stride_qkvt1).to(tl.float16)
        wk = tl.load(qkv_wt_ptr + c[:, None] * stride_qkvt0 + (oc + C)[None, :] * stride_qkvt1).to(tl.float16)
        bq = tl.load(qkv_b_ptr + oc).to(tl.float32)
        bk = tl.load(qkv_b_ptr + oc + C).to(tl.float32)

        q = tl.dot(ln1_h, wq) + bq[None, :]
        k = tl.dot(ln1_h, wk) + bk[None, :]

        scale = 0.1767766952966369
        qh = (q * scale).to(tl.float16)
        kh = k.to(tl.float16)

        attn = tl.dot(qh, tl.trans(kh))
        rpe = tl.load(rpe_ptr + tok[:, None] * stride_r0 + tok[None, :] * stride_r1).to(tl.float32)
        attn = attn + rpe
        row_max = tl.max(attn, axis=1)
        attn_exp = tl.exp(attn - row_max[:, None])
        row_sum = tl.sum(attn_exp, axis=1)
        attn_prob = attn_exp / row_sum[:, None]

        wv = tl.load(qkv_wt_ptr + c[:, None] * stride_qkvt0 + (oc + 2 * C)[None, :] * stride_qkvt1).to(tl.float16)
        bv = tl.load(qkv_b_ptr + oc + 2 * C).to(tl.float32)
        v = tl.dot(ln1_h, wv) + bv[None, :]
        vh = v.to(tl.float16)

        ctx = tl.dot(attn_prob.to(tl.float16), vh)

        wp = tl.load(proj_wt_ptr + c[:, None] * stride_projt0 + oc[None, :] * stride_projt1).to(tl.float16)
        bp = tl.load(proj_b_ptr + oc).to(tl.float32)
        y1 = tl.dot(ctx.to(tl.float16), wp) + bp[None, :] + x_f32
        y1_h = y1.to(tl.float16)
        y1_f32 = y1_h.to(tl.float32)

        ln2_mean = tl.sum(y1_f32, axis=1) / C
        ln2_xc = y1_f32 - ln2_mean[:, None]
        ln2_var = tl.sum(ln2_xc * ln2_xc, axis=1) / C
        ln2_rstd = tl.rsqrt(ln2_var + EPS)

        ln2_w = tl.load(ln2_w_ptr + c).to(tl.float32)
        ln2_b = tl.load(ln2_b_ptr + c).to(tl.float32)
        ln2 = ln2_xc * ln2_rstd[:, None]
        ln2 = ln2 * ln2_w[None, :] + ln2_b[None, :]
        ln2_h = ln2.to(tl.float16)

        out = tl.zeros((N, C), dtype=tl.float32)
        for h0 in range(0, HIDDEN, CHUNK):
            h = tl.arange(0, CHUNK)

            w1 = tl.load(fc1_wt_ptr + c[:, None] * stride_fc1t0 + (h0 + h)[None, :] * stride_fc1t1).to(tl.float16)
            b1 = tl.load(fc1_b_ptr + h0 + h).to(tl.float32)
            up = tl.dot(ln2_h, w1) + b1[None, :]

            erf_in = up * 0.7071067811865476
            g = 0.5 * up * (1.0 + _erf_approx(erf_in))

            w2 = tl.load(fc2_wt_ptr + (h0 + h)[:, None] * stride_fc2t0 + c[None, :] * stride_fc2t1).to(tl.float16)
            out += tl.dot(g.to(tl.float16), w2)

        b2 = tl.load(fc2_b_ptr + c).to(tl.float32)
        out = out + b2[None, :] + y1_f32

        y_offs = (
            b * stride_yb
            + (wh * WS + th)[:, None] * stride_yh
            + (ww * WS + tw)[:, None] * stride_yw
            + c[None, :] * stride_yc
        )
        tl.store(y_ptr + y_offs, out.to(tl.float16))


class ModelNew(Model):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **kwargs):
        super().__init__(dim, num_heads, window_size, shift_size, mlp_ratio=mlp_ratio, **kwargs)
        self._packed_key = None
        self._qkv_w_t = None
        self._proj_w_t = None
        self._fc1_w_t = None
        self._fc2_w_t = None
        self._rpe_2d = None

    def _can_use_triton(self, x: Tensor) -> bool:
        if not _TRITON_AVAILABLE:
            return False
        if not x.is_cuda:
            return False
        if x.dtype != torch.float16:
            return False
        if self.dim != 32:
            return False
        if self.fc1.out_features != 128:
            return False
        if tuple(self.window_size) != (4, 4):
            return False
        if x.ndim != 4:
            return False
        if x.shape[-1] != 32:
            return False
        if x.shape[1] % 4 != 0 or x.shape[2] % 4 != 0:
            return False
        return True

    def _refresh_packed(self):
        key = (
            self.qkv.weight.data_ptr(), self.qkv.weight._version,
            self.proj.weight.data_ptr(), self.proj.weight._version,
            self.fc1.weight.data_ptr(), self.fc1.weight._version,
            self.fc2.weight.data_ptr(), self.fc2.weight._version,
            self.rpe_bias.data_ptr(), self.rpe_bias._version,
            self.qkv.weight.device, self.qkv.weight.dtype,
        )
        if key != self._packed_key:
            with torch.no_grad():
                self._qkv_w_t = self.qkv.weight.t().contiguous()
                self._proj_w_t = self.proj.weight.t().contiguous()
                self._fc1_w_t = self.fc1.weight.t().contiguous()
                self._fc2_w_t = self.fc2.weight.t().contiguous()
                self._rpe_2d = self.rpe_bias[0, 0].contiguous()
            self._packed_key = key

    def forward(self, x: Tensor) -> Tensor:
        if not self._can_use_triton(x):
            return super().forward(x)

        x = x.contiguous()
        self._refresh_packed()

        B, H, W, _ = x.shape
        y = torch.empty_like(x)
        grid = (B * (H // 4) * (W // 4),)

        _block_kernel[grid](
            x,
            y,
            self.norm1.weight,
            self.norm1.bias,
            self._qkv_w_t,
            self.qkv.bias,
            self._proj_w_t,
            self.proj.bias,
            self._rpe_2d,
            self.norm2.weight,
            self.norm2.bias,
            self._fc1_w_t,
            self.fc1.bias,
            self._fc2_w_t,
            self.fc2.bias,
            B,
            H,
            W,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            self._qkv_w_t.stride(0),
            self._qkv_w_t.stride(1),
            self._proj_w_t.stride(0),
            self._proj_w_t.stride(1),
            self._rpe_2d.stride(0),
            self._rpe_2d.stride(1),
            self._fc1_w_t.stride(0),
            self._fc1_w_t.stride(1),
            self._fc2_w_t.stride(0),
            self._fc2_w_t.stride(1),
            EPS=1e-5,
            C=32,
            HIDDEN=128,
            CHUNK=32,
            N=16,
            WS=4,
        )
        return y


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16, device="cuda")]


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

# Override: make compare_swin_fp16.py pick up the optimized class.
Model = ModelNew
