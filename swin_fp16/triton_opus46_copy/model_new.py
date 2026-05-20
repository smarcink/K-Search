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
def _swin_fused_kernel(
    X_ptr, Out_ptr,
    ln1_w_ptr, ln1_b_ptr,
    q_w_ptr, q_b_ptr,
    k_w_ptr, k_b_ptr,
    v_w_ptr, v_b_ptr,
    proj_w_ptr, proj_b_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    rpe_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C: tl.constexpr, Wh: tl.constexpr, Ww: tl.constexpr,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    nH: tl.constexpr, nW: tl.constexpr,
):
    pid = tl.program_id(0)

    b_idx = pid // (nH * nW)
    rem = pid % (nH * nW)
    wh_idx = rem // nW
    ww_idx = rem % nW

    tok_offs = tl.arange(0, N)
    c_offs = tl.arange(0, C)

    base = b_idx * H * W * C
    row_in_window = tok_offs // Ww
    col_in_window = tok_offs % Ww
    h_coords = wh_idx * Wh + row_in_window
    w_coords = ww_idx * Ww + col_in_window

    x_offsets = base + h_coords * W * C + w_coords * C
    x = tl.load(X_ptr + x_offsets[:, None] + c_offs[None, :])
    residual1 = x

    # LayerNorm1
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=1) / C
    x_centered = x_f32 - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    ln1_w = tl.load(ln1_w_ptr + c_offs).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_offs).to(tl.float32)
    x_ln1 = (x_centered * inv_std[:, None] * ln1_w[None, :] + ln1_b[None, :]).to(tl.float16)

    # Q, K, V projections
    q_w = tl.load(q_w_ptr + c_offs[:, None] * C + c_offs[None, :])
    q_b = tl.load(q_b_ptr + c_offs)
    q = tl.dot(x_ln1, tl.trans(q_w)) + q_b[None, :]

    k_w = tl.load(k_w_ptr + c_offs[:, None] * C + c_offs[None, :])
    k_b = tl.load(k_b_ptr + c_offs)
    k = tl.dot(x_ln1, tl.trans(k_w)) + k_b[None, :]

    v_w = tl.load(v_w_ptr + c_offs[:, None] * C + c_offs[None, :])
    v_b = tl.load(v_b_ptr + c_offs)
    v = tl.dot(x_ln1, tl.trans(v_w)) + v_b[None, :]
    v = v.to(tl.float16)

    # Attention: scale Q, compute QK^T in fp32
    scale = tl.rsqrt(tl.cast(C, tl.float32))
    q_scaled = (q * scale).to(tl.float16)
    k = k.to(tl.float16)
    attn = tl.dot(q_scaled, tl.trans(k))
    rpe = tl.load(rpe_ptr + tok_offs[:, None] * N + tok_offs[None, :])
    attn = attn + rpe.to(tl.float32)

    attn_max = tl.max(attn, axis=1)
    attn = tl.exp(attn - attn_max[:, None])
    attn_sum = tl.sum(attn, axis=1)
    attn_norm = (attn / attn_sum[:, None]).to(tl.float16)

    attn_out = tl.dot(attn_norm, v).to(tl.float16)

    # Output projection
    p_w = tl.load(proj_w_ptr + c_offs[:, None] * C + c_offs[None, :])
    p_b = tl.load(proj_b_ptr + c_offs)
    proj_out = (tl.dot(attn_out, tl.trans(p_w)) + p_b[None, :]).to(tl.float16)

    # Residual 1
    x2 = residual1 + proj_out
    residual2 = x2

    # LayerNorm2
    x2_f32 = x2.to(tl.float32)
    mean2 = tl.sum(x2_f32, axis=1) / C
    x2_centered = x2_f32 - mean2[:, None]
    var2 = tl.sum(x2_centered * x2_centered, axis=1) / C
    inv_std2 = 1.0 / tl.sqrt(var2 + 1e-5)
    ln2_w = tl.load(ln2_w_ptr + c_offs).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + c_offs).to(tl.float32)
    x2_ln = (x2_centered * inv_std2[:, None] * ln2_w[None, :] + ln2_b[None, :]).to(tl.float16)

    # MLP fc1
    h_offs = tl.arange(0, HIDDEN)
    fc1_w = tl.load(fc1_w_ptr + h_offs[:, None] * C + c_offs[None, :])
    fc1_b = tl.load(fc1_b_ptr + h_offs)
    hidden_out = tl.dot(x2_ln, tl.trans(fc1_w), allow_tf32=True) + fc1_b[None, :]

    # GELU
    h_f32 = hidden_out.to(tl.float32)
    cube = h_f32 * h_f32 * h_f32
    inner = 0.7978845608028654 * (h_f32 + 0.044715 * cube)
    exp2x = tl.exp(2.0 * inner)
    tanh_val = (exp2x - 1.0) / (exp2x + 1.0)
    gelu_out = (0.5 * h_f32 * (1.0 + tanh_val)).to(tl.float16)

    # MLP fc2
    fc2_w = tl.load(fc2_w_ptr + c_offs[:, None] * HIDDEN + h_offs[None, :])
    fc2_b = tl.load(fc2_b_ptr + c_offs)
    mlp_out = (tl.dot(gelu_out, tl.trans(fc2_w), allow_tf32=True) + fc2_b[None, :]).to(tl.float16)

    out = residual2 + mlp_out
    tl.store(Out_ptr + x_offsets[:, None] + c_offs[None, :], out)


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
        # Pre-split QKV
        self._split_cached = False

    def _cache_splits(self):
        C = self.dim
        w = self.qkv.weight
        b = self.qkv.bias
        self.register_buffer("_q_w", w[:C, :].contiguous())
        self.register_buffer("_k_w", w[C:2*C, :].contiguous())
        self.register_buffer("_v_w", w[2*C:, :].contiguous())
        self.register_buffer("_q_b", b[:C].contiguous())
        self.register_buffer("_k_b", b[C:2*C].contiguous())
        self.register_buffer("_v_b", b[2*C:].contiguous())
        N = self.window_size[0] * self.window_size[1]
        self.register_buffer("_rpe_flat", self.rpe_bias.reshape(N, N).contiguous())
        self._split_cached = True

    def forward(self, x: Tensor) -> Tensor:
        if not self._split_cached:
            self._cache_splits()
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww
        num_windows = B * nH * nW
        out = torch.empty_like(x)
        grid = (num_windows,)
        _swin_fused_kernel[grid](
            x, out,
            self.norm1.weight, self.norm1.bias,
            self._q_w, self._q_b, self._k_w, self._k_b, self._v_w, self._v_b,
            self.proj.weight, self.proj.bias,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self._rpe_flat,
            B, H, W, C, Wh, Ww,
            self.hidden, N, nH, nW,
            num_warps=2,
            num_stages=1,
        )
        return out

# --- Adapters for compare_swin_fp16.py / drop-in use ---

# Alias so the comparison harness picks up this module as a Model.
Model = ModelNew


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16)]


def get_init_inputs():
    return [32, 1, [4, 4], [0, 0], 4]


def state_dict_remap(src_state):
    """Translate the reference checkpoint (swin_block_fp16.py) into this layout.

    Same mapping as swin_block_fp16_simplified.py: rename the wrapped
    ShiftedWindowAttention paths and the Sequential MLP indices, and gather
    the (relative_position_bias_table, relative_position_index) pair into
    the precomputed rpe_bias buffer.
    """
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
    def get_device():
        if torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = get_device()
    print(f"Device: {device}")
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).to(device)
    inputs = [t.to(device) for t in get_inputs()]
    print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

    with torch.no_grad():
        out = model(*inputs)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")

    # Compare against reference
    from swin_block_fp16_simplified import Model as RefModel
    ref = RefModel(*init_inputs).to(device)
    # Copy weights from triton model to reference
    ref.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        ref_out = ref(*inputs)
    max_diff = (out - ref_out).abs().max().item()
    print(f"Max diff vs reference: {max_diff:.4e}")
    print("PASS" if max_diff < 1e-2 else "FAIL")
