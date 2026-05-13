import torch
from torch import nn, Tensor

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


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


if triton is not None:
    @triton.jit
    def _ln_qkv_kernel(
        x_ptr, qkv_tmp_ptr,
        ln_w_ptr, ln_b_ptr,
        qkv_w_ptr, qkv_b_ptr,
        H, W,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_n = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)

        nW = W // 4
        windows_per_b = (H // 4) * nW
        b = pid // windows_per_b
        rem = pid - b * windows_per_b
        wh_id = rem // nW
        ww_id = rem - wh_id * nW

        dh = offs_n // 4
        dw = offs_n - dh * 4
        hs = wh_id * 4 + dh
        ws = ww_id * 4 + dw

        token_base = ((b * H + hs) * W + ws) * 32
        x = tl.load(x_ptr + token_base[:, None] + offs_c[None, :]).to(tl.float32)

        mean = tl.sum(x, axis=1) * 0.03125
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) * 0.03125
        inv = tl.rsqrt(var + 1.0e-5)

        ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
        y = xc * inv[:, None] * ln_w[None, :] + ln_b[None, :]

        wq = tl.load(qkv_w_ptr + offs_c[:, None] + (offs_c[None, :] + 0) * 32)
        wk = tl.load(qkv_w_ptr + offs_c[:, None] + (offs_c[None, :] + 32) * 32)
        wv = tl.load(qkv_w_ptr + offs_c[:, None] + (offs_c[None, :] + 64) * 32)

        bq = tl.load(qkv_b_ptr + offs_c + 0).to(tl.float32)
        bk = tl.load(qkv_b_ptr + offs_c + 32).to(tl.float32)
        bv = tl.load(qkv_b_ptr + offs_c + 64).to(tl.float32)

        yh = y.to(tl.float16)
        q = tl.dot(yh, wq).to(tl.float32) + bq[None, :]
        k = tl.dot(yh, wk).to(tl.float32) + bk[None, :]
        v = tl.dot(yh, wv).to(tl.float32) + bv[None, :]

        base = pid * 1536 + offs_n[:, None] * 96
        tl.store(qkv_tmp_ptr + base + offs_c[None, :] + 0, q.to(tl.float16))
        tl.store(qkv_tmp_ptr + base + offs_c[None, :] + 32, k.to(tl.float16))
        tl.store(qkv_tmp_ptr + base + offs_c[None, :] + 64, v.to(tl.float16))


    @triton.jit
    def _attn_proj_res_kernel(
        x_ptr, qkv_tmp_ptr, tmp_ptr,
        proj_w_ptr, proj_b_ptr,
        rpe_ptr,
        H, W,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_n = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)

        nW = W // 4
        windows_per_b = (H // 4) * nW
        b = pid // windows_per_b
        rem = pid - b * windows_per_b
        wh_id = rem // nW
        ww_id = rem - wh_id * nW

        dh = offs_n // 4
        dw = offs_n - dh * 4
        hs = wh_id * 4 + dh
        ws = ww_id * 4 + dw

        base = pid * 1536 + offs_n[:, None] * 96
        q = tl.load(qkv_tmp_ptr + base + offs_c[None, :] + 0)
        k = tl.load(qkv_tmp_ptr + base + offs_c[None, :] + 32)
        v = tl.load(qkv_tmp_ptr + base + offs_c[None, :] + 64)

        logits = tl.dot(q, tl.trans(k)).to(tl.float32) * 0.1767766952966369
        rpe = tl.load(rpe_ptr + offs_n[:, None] * 16 + offs_n[None, :]).to(tl.float32)
        logits = logits + rpe

        m = tl.max(logits, axis=1)
        p = tl.exp(logits - m[:, None])
        d = tl.sum(p, axis=1)
        p = p / d[:, None]

        ctx = tl.dot(p.to(tl.float16), v).to(tl.float16)

        wp = tl.load(proj_w_ptr + offs_c[:, None] + offs_c[None, :] * 32)
        bp = tl.load(proj_b_ptr + offs_c).to(tl.float32)
        attn_out = tl.dot(ctx, wp).to(tl.float32) + bp[None, :]

        token_base = ((b * H + hs) * W + ws) * 32
        x = tl.load(x_ptr + token_base[:, None] + offs_c[None, :]).to(tl.float32)
        out = x + attn_out
        tl.store(tmp_ptr + token_base[:, None] + offs_c[None, :], out.to(tl.float16))


    @triton.jit
    def _ln_fc1_gelu_kernel(
        tmp_ptr, hidden_ptr,
        ln_w_ptr, ln_b_ptr,
        fc1_w_ptr, fc1_b_ptr,
        H, W,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h_chunk = tl.program_id(1)

        offs_n = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)
        offs_h = h_chunk * BLOCK_H + tl.arange(0, BLOCK_H)

        nW = W // 4
        windows_per_b = (H // 4) * nW
        b = pid // windows_per_b
        rem = pid - b * windows_per_b
        wh_id = rem // nW
        ww_id = rem - wh_id * nW

        dh = offs_n // 4
        dw = offs_n - dh * 4
        hs = wh_id * 4 + dh
        ws = ww_id * 4 + dw

        token_base = ((b * H + hs) * W + ws) * 32
        x = tl.load(tmp_ptr + token_base[:, None] + offs_c[None, :]).to(tl.float32)

        mean = tl.sum(x, axis=1) * 0.03125
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) * 0.03125
        inv = tl.rsqrt(var + 1.0e-5)

        ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
        y = xc * inv[:, None] * ln_w[None, :] + ln_b[None, :]

        w1 = tl.load(fc1_w_ptr + offs_c[:, None] + offs_h[None, :] * 32)
        b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
        h = tl.dot(y.to(tl.float16), w1).to(tl.float32) + b1[None, :]

        h3 = h * h * h
        gelu = 0.5 * h * (1.0 + tl.tanh(0.7978845608028654 * (h + 0.044715 * h3)))

        tl.store(hidden_ptr + pid * 2048 + offs_n[:, None] * 128 + offs_h[None, :], gelu.to(tl.float16))


    @triton.jit
    def _fc2_res_kernel(
        tmp_ptr, hidden_ptr, out_ptr,
        fc2_w_ptr, fc2_b_ptr,
        H, W,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_n = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)
        offs_h = tl.arange(0, BLOCK_H)

        nW = W // 4
        windows_per_b = (H // 4) * nW
        b = pid // windows_per_b
        rem = pid - b * windows_per_b
        wh_id = rem // nW
        ww_id = rem - wh_id * nW

        dh = offs_n // 4
        dw = offs_n - dh * 4
        hs = wh_id * 4 + dh
        ws = ww_id * 4 + dw

        h = tl.load(hidden_ptr + pid * 2048 + offs_n[:, None] * 128 + offs_h[None, :])
        w2 = tl.load(fc2_w_ptr + offs_h[:, None] + offs_c[None, :] * 128)
        b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)

        mlp = tl.dot(h, w2).to(tl.float32) + b2[None, :]

        token_base = ((b * H + hs) * W + ws) * 32
        x = tl.load(tmp_ptr + token_base[:, None] + offs_c[None, :]).to(tl.float32)

        out = x + mlp
        tl.store(out_ptr + token_base[:, None] + offs_c[None, :], out.to(tl.float16))


class ModelNew(nn.Module):
    """Simplified Swin block optimized with staged window-local Triton kernels."""

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

    def _forward_fallback(self, x: Tensor) -> Tensor:
        x = x + self._attn_fallback(self.norm1(x))
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x

    def forward(self, x: Tensor) -> Tensor:
        if (
            triton is None
            or x.device.type != "xpu"
            or x.dtype != torch.float16
            or self.dim != 32
            or tuple(self.window_size) != (4, 4)
            or self.fc1.out_features != 128
            or x.shape[-1] != 32
            or x.shape[1] % 4 != 0
            or x.shape[2] % 4 != 0
            or not x.is_contiguous()
        ):
            return self._forward_fallback(x)

        B, H, W, C = x.shape
        num_windows = B * (H // 4) * (W // 4)

        qkv_tmp = torch.empty((num_windows, 16, 96), device=x.device, dtype=torch.float16)
        tmp = torch.empty_like(x)
        hidden = torch.empty((num_windows, 16, 128), device=x.device, dtype=torch.float16)
        out = torch.empty_like(x)

        grid = (num_windows,)

        _ln_qkv_kernel[grid](
            x, qkv_tmp,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            H, W,
            BLOCK_N=16,
            BLOCK_C=32,
            num_warps=4,
            num_stages=3,
        )

        _attn_proj_res_kernel[grid](
            x, qkv_tmp, tmp,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            H, W,
            BLOCK_N=16,
            BLOCK_C=32,
            num_warps=4,
            num_stages=3,
        )

        _ln_fc1_gelu_kernel[(num_windows, 2)](
            tmp, hidden,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            H, W,
            BLOCK_N=16,
            BLOCK_C=32,
            BLOCK_H=64,
            num_warps=4,
            num_stages=3,
        )

        _fc2_res_kernel[grid](
            tmp, hidden, out,
            self.fc2.weight, self.fc2.bias,
            H, W,
            BLOCK_N=16,
            BLOCK_C=32,
            BLOCK_H=128,
            num_warps=4,
            num_stages=3,
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


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).to(device)
    inputs = [t.to(device) for t in get_inputs()]
    print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

    with torch.no_grad():
        out = model(*inputs)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")

    with torch.no_grad():
        out2 = model(*inputs)
    max_diff = (out - out2).abs().max().item()
    print(f"Determinism check (same input twice): max_diff={max_diff:.2e}")
    print("OK")