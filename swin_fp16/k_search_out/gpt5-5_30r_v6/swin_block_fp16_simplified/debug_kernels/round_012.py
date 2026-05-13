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
def _ln_fc1_kernel(
    x_ptr, gamma_ptr, beta_ptr, w_ptr, b_ptr, h_ptr,
    M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :],
                mask=mask_m[:, None], other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + 1.0e-5)

    gamma = tl.load(gamma_ptr + offs_c).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c).to(tl.float32)
    xn = (xc * rstd[:, None]) * gamma[None, :] + beta[None, :]
    xn = xn.to(tl.float16)

    w = tl.load(w_ptr + offs_n[None, :] * BLOCK_C + offs_c[:, None],
                mask=offs_n[None, :] < 128, other=0.0)
    acc = tl.dot(xn, w)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < 128, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    tl.store(h_ptr + offs_m[:, None] * 128 + offs_n[None, :],
             acc.to(tl.float16),
             mask=mask_m[:, None] & (offs_n[None, :] < 128))


@triton.jit
def _gelu_fc2_res_kernel(
    h_ptr, x_ptr, w_ptr, b_ptr, out_ptr,
    M,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    h = tl.load(h_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :],
                mask=mask_m[:, None], other=0.0).to(tl.float32)

    gelu = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    gelu = gelu.to(tl.float16)

    w = tl.load(w_ptr + offs_n[None, :] * BLOCK_K + offs_k[:, None])
    acc = tl.dot(gelu, w)
    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    acc = acc + bias[None, :]

    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                  mask=mask_m[:, None], other=0.0).to(tl.float32)
    y = acc + res

    tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
             y.to(tl.float16),
             mask=mask_m[:, None])


class ModelNew(nn.Module):
    """Simplified Swin block with Triton-fused MLP path."""

    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **_):
        super().__init__()
        assert num_heads == 1, "simplified path: single-head only"
        assert tuple(shift_size) == (0, 0), "simplified path: no shift"
        hidden = int(dim * mlp_ratio)
        self.dim = dim
        self.hidden = hidden
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

    def _mlp_triton(self, x: Tensor) -> Tensor:
        if (
            x.device.type != "xpu"
            or self.dim != 32
            or self.hidden != 128
            or x.dtype != torch.float16
            or not x.is_contiguous()
        ):
            return x + self.fc2(self.act(self.fc1(self.norm2(x))))

        M = x.numel() // 32
        h = torch.empty((M, 128), device=x.device, dtype=torch.float16)
        out = torch.empty_like(x)

        grid_fc1 = (triton.cdiv(M, 16), 2)
        _ln_fc1_kernel[grid_fc1](
            x, self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias, h,
            M,
            BLOCK_M=16, BLOCK_N=64, BLOCK_C=32,
            num_warps=4,
        )

        grid_fc2 = (triton.cdiv(M, 16),)
        _gelu_fc2_res_kernel[grid_fc2](
            h, x, self.fc2.weight, self.fc2.bias, out,
            M,
            BLOCK_M=16, BLOCK_K=128, BLOCK_N=32,
            num_warps=4,
        )
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attn(self.norm1(x))
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