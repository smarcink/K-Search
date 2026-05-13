import torch
from torch import nn, Tensor

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


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


if _HAS_TRITON:
    @triton.jit
    def _ln_linear_window_kernel(
        x_ptr, gamma_ptr, beta_ptr, w_ptr, bias_ptr, out_ptr,
        H, W, total_rows, nW, windows_per_b, OUT_FEATURES,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c = tl.arange(0, BLOCK_C)

        win = rows // 16
        pos = rows - win * 16
        b = win // windows_per_b
        rem = win - b * windows_per_b
        wh = rem // nW
        ww = rem - wh * nW
        ph = pos // 4
        pw = pos - ph * 4
        ih = wh * 4 + ph
        iw = ww * 4 + pw
        orig = (b * H + ih) * W + iw

        mask_r = rows < total_rows
        vals = tl.load(x_ptr + orig[:, None] * 32 + c[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)

        mean = tl.sum(vals, axis=1) * 0.03125
        diff = vals - mean[:, None]
        var = tl.sum(diff * diff, axis=1) * 0.03125
        rstd = tl.rsqrt(var + 1.0e-5)

        g = tl.load(gamma_ptr + c).to(tl.float32)
        be = tl.load(beta_ptr + c).to(tl.float32)
        y = diff * rstd[:, None]
        y = y * g[None, :] + be[None, :]
        yh = y.to(tl.float16)

        wt = tl.load(
            w_ptr + cols[None, :] * 32 + c[:, None],
            mask=cols[None, :] < OUT_FEATURES,
            other=0.0,
        )
        acc = tl.dot(yh, wt)
        bs = tl.load(bias_ptr + cols, mask=cols < OUT_FEATURES, other=0.0).to(tl.float32)
        acc = acc + bs[None, :]

        tl.store(
            out_ptr + rows[:, None] * OUT_FEATURES + cols[None, :],
            acc.to(tl.float16),
            mask=mask_r[:, None] & (cols[None, :] < OUT_FEATURES),
        )


    @triton.jit
    def _ln_fc1_gelu_kernel(
        x_ptr, gamma_ptr, beta_ptr, w_ptr, bias_ptr, out_ptr,
        total_rows, HIDDEN,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c = tl.arange(0, BLOCK_C)

        mask_r = rows < total_rows
        vals = tl.load(x_ptr + rows[:, None] * 32 + c[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)

        mean = tl.sum(vals, axis=1) * 0.03125
        diff = vals - mean[:, None]
        var = tl.sum(diff * diff, axis=1) * 0.03125
        rstd = tl.rsqrt(var + 1.0e-5)

        g = tl.load(gamma_ptr + c).to(tl.float32)
        be = tl.load(beta_ptr + c).to(tl.float32)
        y = diff * rstd[:, None]
        y = y * g[None, :] + be[None, :]
        yh = y.to(tl.float16)

        wt = tl.load(
            w_ptr + cols[None, :] * 32 + c[:, None],
            mask=cols[None, :] < HIDDEN,
            other=0.0,
        )
        acc = tl.dot(yh, wt)
        bs = tl.load(bias_ptr + cols, mask=cols < HIDDEN, other=0.0).to(tl.float32)
        acc = acc + bs[None, :]

        pre = acc.to(tl.float16).to(tl.float32)
        gelu = pre * 0.5 * (1.0 + tl.erf(pre * 0.7071067811865476))

        tl.store(
            out_ptr + rows[:, None] * HIDDEN + cols[None, :],
            gelu.to(tl.float16),
            mask=mask_r[:, None] & (cols[None, :] < HIDDEN),
        )


    @triton.jit
    def _fc2_residual_kernel(
        h_ptr, w_ptr, bias_ptr, residual_ptr, out_ptr,
        total_rows,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_r = rows < total_rows
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in tl.static_range(0, 128, 32):
            h = tl.load(
                h_ptr + rows[:, None] * 128 + (k0 + offs_k)[None, :],
                mask=mask_r[:, None],
                other=0.0,
            )
            wt = tl.load(w_ptr + cols[None, :] * 128 + (k0 + offs_k)[:, None])
            acc += tl.dot(h, wt)

        bs = tl.load(bias_ptr + cols).to(tl.float32)
        acc = acc + bs[None, :]

        y = acc.to(tl.float16).to(tl.float32)
        res = tl.load(residual_ptr + rows[:, None] * 32 + cols[None, :], mask=mask_r[:, None], other=0.0).to(tl.float32)
        out = res + y

        tl.store(out_ptr + rows[:, None] * 32 + cols[None, :], out.to(tl.float16), mask=mask_r[:, None])


class ModelNew(nn.Module):
    """Simplified Swin block with fixed-shape Triton fast paths."""

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

    def _can_use_triton(self, x: Tensor) -> bool:
        return (
            _HAS_TRITON
            and x.device.type == "xpu"
            and x.dtype == torch.float16
            and self.dim == 32
            and self.window_size[0] == 4
            and self.window_size[1] == 4
            and x.is_contiguous()
        )

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

    def _attn(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        if not self._can_use_triton(x):
            return self._attn_fallback(x)

        total = B * H * W
        qkv_flat = torch.empty((total, 3 * C), device=x.device, dtype=x.dtype)

        BM = 32
        BN = 32
        grid = (triton.cdiv(total, BM), triton.cdiv(3 * C, BN))
        _ln_linear_window_kernel[grid](
            x, self.norm1.weight, self.norm1.bias, self.qkv.weight, self.qkv.bias, qkv_flat,
            H, W, total, nW, nH * nW, 3 * C,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_C=32,
            num_warps=4,
        )

        qkv = qkv_flat.view(-1, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1) + self.rpe_bias
        attn = attn.softmax(dim=-1)

        y = (attn @ v).reshape(-1, N, C)
        y = self.proj(y)

        y = y.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return y

    def _mlp_fallback(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))

    def _mlp_triton(self, x: Tensor) -> Tensor:
        if not self._can_use_triton(x) or self.fc1.out_features != 128:
            return self._mlp_fallback(x)

        B, H, W, C = x.shape
        total = B * H * W
        hidden = torch.empty((B, H, W, 128), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)

        BM = 32
        BN = 32
        grid1 = (triton.cdiv(total, BM), triton.cdiv(128, BN))
        _ln_fc1_gelu_kernel[grid1](
            x, self.norm2.weight, self.norm2.bias, self.fc1.weight, self.fc1.bias, hidden,
            total, 128,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_C=32,
            num_warps=4,
        )

        grid2 = (triton.cdiv(total, BM),)
        _fc2_residual_kernel[grid2](
            hidden, self.fc2.weight, self.fc2.bias, x, out,
            total,
            BLOCK_M=BM, BLOCK_N=32, BLOCK_K=32,
            num_warps=4,
        )
        return out

    def forward(self, x: Tensor) -> Tensor:
        if x.is_contiguous():
            y = self._attn(x)
        else:
            x = x.contiguous()
            y = self._attn(x)
        x = x + y
        x = self._mlp_triton(x)
        return x


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