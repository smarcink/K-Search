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
    def _swin_ln_qkv_kernel(
        x_ptr,
        qkv_out_ptr,
        ln_w_ptr,
        ln_b_ptr,
        qkv_w_ptr,
        qkv_b_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        pid = tl.program_id(0)
    
        win_per_b = NH * NW
        b = pid // win_per_b
        rem = pid - b * win_per_b
        wh = rem // NW
        ww = rem - wh * NW

        offs_n = tl.arange(0, 16)
        offs_c = tl.arange(0, 32)

        th = offs_n // 4
        tw = offs_n - th * 4

        x_offsets = (
            ((b * H + wh * 4 + th[:, None]) * W + ww * 4 + tw[:, None]) * 32
            + offs_c[None, :]
        )

        x_h = tl.load(x_ptr + x_offsets)
        x_f = x_h.to(tl.float32)

        mean = tl.sum(x_f, axis=1) * 0.03125
        xc = x_f - mean[:, None]
        var = tl.sum(xc * xc, axis=1) * 0.03125
        rstd = tl.rsqrt(var + 1.0e-5)

        ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
        xn = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
        xn_h = xn.to(tl.float16)

        offs_k = tl.arange(0, 32)
        offs_o = tl.arange(0, 32)

        wq = tl.load(qkv_w_ptr + offs_o[None, :] * 32 + offs_k[:, None])
        wk = tl.load(qkv_w_ptr + (offs_o[None, :] + 32) * 32 + offs_k[:, None])
        wv = tl.load(qkv_w_ptr + (offs_o[None, :] + 64) * 32 + offs_k[:, None])

        bq = tl.load(qkv_b_ptr + offs_o).to(tl.float32)
        bk = tl.load(qkv_b_ptr + offs_o + 32).to(tl.float32)
        bv = tl.load(qkv_b_ptr + offs_o + 64).to(tl.float32)

        q = (tl.dot(xn_h, wq) + bq[None, :]).to(tl.float16)
        k = (tl.dot(xn_h, wk) + bk[None, :]).to(tl.float16)
        v = (tl.dot(xn_h, wv) + bv[None, :]).to(tl.float16)

        base = pid * 1536 + offs_n[:, None] * 96 + offs_o[None, :]
        tl.store(qkv_out_ptr + base, q)
        tl.store(qkv_out_ptr + base + 32, k)
        tl.store(qkv_out_ptr + base + 64, v)


    @triton.jit
    def _swin_attn_proj_res_kernel(
        x_ptr,
        qkv_ptr,
        out_ptr,
        proj_w_ptr,
        proj_b_ptr,
        rpe_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        pid = tl.program_id(0)

        win_per_b = NH * NW
        b = pid // win_per_b
        rem = pid - b * win_per_b
        wh = rem // NW
        ww = rem - wh * NW

        offs_n = tl.arange(0, 16)
        offs_c = tl.arange(0, 32)

        qkv_base = pid * 1536 + offs_n[:, None] * 96 + offs_c[None, :]
        q = tl.load(qkv_ptr + qkv_base)
        k = tl.load(qkv_ptr + qkv_base + 32)
        v = tl.load(qkv_ptr + qkv_base + 64)

        q_scaled = (q.to(tl.float32) * 0.17677669529663687).to(tl.float16)
        scores = tl.dot(q_scaled, tl.trans(k))
        rpe = tl.load(rpe_ptr + offs_n[:, None] * 16 + offs_n[None, :]).to(tl.float32)
        scores = scores.to(tl.float32) + rpe

        scores = scores - tl.max(scores, axis=1)[:, None]
        exp_scores = tl.exp(scores)
        denom = tl.sum(exp_scores, axis=1)
        attn = (exp_scores / denom[:, None]).to(tl.float16)

        ctx = tl.dot(attn, v).to(tl.float16)

        offs_k = tl.arange(0, 32)
        offs_o = tl.arange(0, 32)
        wp = tl.load(proj_w_ptr + offs_o[None, :] * 32 + offs_k[:, None])
        bp = tl.load(proj_b_ptr + offs_o).to(tl.float32)
        proj = (tl.dot(ctx, wp) + bp[None, :]).to(tl.float16)

        th = offs_n // 4
        tw = offs_n - th * 4
        x_offsets = (
            ((b * H + wh * 4 + th[:, None]) * W + ww * 4 + tw[:, None]) * 32
            + offs_c[None, :]
        )

        x_h = tl.load(x_ptr + x_offsets)
        tl.store(out_ptr + x_offsets, x_h + proj)


class ModelNew(nn.Module):
    """Simplified Swin block with selectable Triton kernel isolation.

    triton_mode controls which Triton kernels run:
      "both"          – both kernels (original behaviour)
      "ln_qkv_only"   – Triton kernel 1 (LN+QKV), then PyTorch for attn+proj+residual
      "attn_proj_only" – PyTorch for LN+QKV, then Triton kernel 2 (attn+proj+residual)
      "none"          – pure PyTorch fallback (no Triton at all)
    """

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

        # "both" | "ln_qkv_only" | "attn_proj_only" | "none"
        self.triton_mode = "both"

    # ---- helpers to check if triton path is feasible ----
    def _triton_eligible(self, x: Tensor) -> bool:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        return (
            _HAS_TRITON
            and x.device.type == "xpu"
            and x.dtype == torch.float16
            and C == 32
            and Wh == 4
            and Ww == 4
            and H % 4 == 0
            and W % 4 == 0
            and x.is_contiguous()
        )

    # ---- pure PyTorch attention (no residual) ----
    def _attn(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)

        qkv = self.qkv(x).reshape(-1, N, 3, C)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]

        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn + self.rpe_bias
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = self.proj(x)

        x = x.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return x

    # ---- mode: both triton kernels (original) ----
    def _attn_residual_both(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        nH = H // 4
        nW = W // 4
        total_windows = B * nH * nW
        out = torch.empty_like(x)
        qkv_tmp = torch.empty((total_windows, 16, 96), device=x.device, dtype=x.dtype)
        grid = (total_windows,)

        _swin_ln_qkv_kernel[grid](
            x, qkv_tmp,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            H, W, nH, nW,
            num_warps=1,
        )
        _swin_attn_proj_res_kernel[grid](
            x, qkv_tmp, out,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            H, W, nH, nW,
            num_warps=1,
        )
        return out

    # ---- mode: only kernel 1 (LN+QKV in Triton), rest in PyTorch ----
    def _attn_residual_ln_qkv_only(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        total_windows = B * nH * nW

        # --- Triton kernel 1: LN + QKV projection ---
        qkv_tmp = torch.empty((total_windows, 16, 96), device=x.device, dtype=x.dtype)
        grid = (total_windows,)
        _swin_ln_qkv_kernel[grid](
            x, qkv_tmp,
            self.norm1.weight, self.norm1.bias,
            self.qkv.weight, self.qkv.bias,
            H, W, nH, nW,
            num_warps=1,
        )

        # --- PyTorch: unpack Q/K/V from qkv_tmp and do attention + proj + residual ---
        # qkv_tmp layout: [total_windows, 16, 96] with Q at [:,:,:32], K at [:,:,32:64], V at [:,:,64:96]
        q = qkv_tmp[:, :, :32]   # (total_windows, 16, 32)
        k = qkv_tmp[:, :, 32:64]
        v = qkv_tmp[:, :, 64:96]

        scale = C ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)  # (total_windows, 16, 16)
        rpe = self.rpe_bias.view(1, N, N)
        attn = attn + rpe
        attn = attn.softmax(dim=-1)
        ctx = attn @ v  # (total_windows, 16, 32)
        proj_out = self.proj(ctx)  # (total_windows, 16, 32)

        # Scatter back to (B, H, W, C) and add residual
        out = x.clone()
        proj_out = proj_out.view(B, nH, nW, Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        out = out + proj_out
        return out

    # ---- mode: only kernel 2 (attn+proj+residual in Triton), LN+QKV in PyTorch ----
    def _attn_residual_attn_proj_only(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        total_windows = B * nH * nW

        # --- PyTorch: LN + QKV, pack into qkv_tmp format ---
        x_normed = self.norm1(x)
        # Window-partition: (B, H, W, C) -> (total_windows, N, C)
        x_win = x_normed.view(B, nH, Wh, nW, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, N, C)
        qkv = self.qkv(x_win).reshape(-1, N, 3, C)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]

        # Pack into qkv_tmp with same layout as kernel 1: [total_windows, 16, 96]
        qkv_tmp = torch.empty((total_windows, N, 96), device=x.device, dtype=x.dtype)
        qkv_tmp[:, :, :32] = q
        qkv_tmp[:, :, 32:64] = k
        qkv_tmp[:, :, 64:96] = v

        # --- Triton kernel 2: attention + projection + residual ---
        out = torch.empty_like(x)
        grid = (total_windows,)
        _swin_attn_proj_res_kernel[grid](
            x, qkv_tmp, out,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            H, W, nH, nW,
            num_warps=1,
        )
        return out

    def _attn_residual_triton(self, x: Tensor) -> Tensor:
        if not self._triton_eligible(x) or self.triton_mode == "none":
            return x + self._attn(self.norm1(x))

        if self.triton_mode == "both":
            return self._attn_residual_both(x)
        elif self.triton_mode == "ln_qkv_only":
            return self._attn_residual_ln_qkv_only(x)
        elif self.triton_mode == "attn_proj_only":
            return self._attn_residual_attn_proj_only(x)
        else:
            raise ValueError(f"Unknown triton_mode: {self.triton_mode!r}")

    def forward(self, x: Tensor) -> Tensor:
        x = self._attn_residual_triton(x)
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
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
    import sys

    device = get_device()
    print(f"Device: {device}")
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).to(device)
    inputs = [t.to(device) for t in get_inputs()]
    print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

    modes = sys.argv[1:] if len(sys.argv) > 1 else ["none", "ln_qkv_only", "attn_proj_only", "both"]
    for mode in modes:
        print(f"\n=== triton_mode={mode!r} ===")
        model.triton_mode = mode
        try:
            with torch.no_grad():
                out = model(*inputs)
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
            print(f"  Output abs-mean: {out.abs().mean().item():.4e}")
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
