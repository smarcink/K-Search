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
def swin_block_kernel(
    X_ptr, Y_ptr,
    ln1_w_ptr, ln1_b_ptr,
    q_w_ptr, q_b_ptr,
    k_w_ptr, k_b_ptr,
    v_w_ptr, v_b_ptr,
    proj_w_ptr, proj_b_ptr,
    rpe_ptr,
    ln2_w_ptr, ln2_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    B, H, W,
    nH, nW,
    C: tl.constexpr,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    WS: tl.constexpr,
    WINDOWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    total_windows = B * nH * nW
    base_w = pid * WINDOWS_PER_PROG

    NTOT: tl.constexpr = N * WINDOWS_PER_PROG

    c_idx = tl.arange(0, C)
    h_idx = tl.arange(0, HIDDEN)
    nt_idx = tl.arange(0, NTOT)
    w_local = nt_idx // N
    tok_local = nt_idx % N

    ln1_w = tl.load(ln1_w_ptr + c_idx).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + c_idx).to(tl.float32)
    ln2_w = tl.load(ln2_w_ptr + c_idx).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + c_idx).to(tl.float32)

    q_b = tl.load(q_b_ptr + c_idx).to(tl.float32)
    k_b = tl.load(k_b_ptr + c_idx).to(tl.float32)
    v_b = tl.load(v_b_ptr + c_idx).to(tl.float32)
    proj_b = tl.load(proj_b_ptr + c_idx).to(tl.float32)
    fc1_b = tl.load(fc1_b_ptr + h_idx).to(tl.float32)
    fc2_b = tl.load(fc2_b_ptr + c_idx).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.full((), C, tl.float32))

    win_ids = base_w + w_local
    valid_w = win_ids < total_windows
    safe_win = tl.where(valid_w, win_ids, 0)
    b = safe_win // (nH * nW)
    rem = safe_win % (nH * nW)
    ih = rem // nW
    iw = rem % nW

    i_in = tok_local // WS
    j_in = tok_local % WS
    row = ih * WS + i_in
    col = iw * WS + j_in

    x_off = b[:, None] * (H * W * C) + row[:, None] * (W * C) + col[:, None] * C + c_idx[None, :]
    mask2d = valid_w[:, None]

    x = tl.load(X_ptr + x_off, mask=mask2d, other=0.0)
    x_f = x.to(tl.float32)
    residual1 = x_f

    mean = tl.sum(x_f, axis=1) / C
    xc = x_f - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    inv = 1.0 / tl.sqrt(var + 1e-5)
    ln1 = xc * inv[:, None] * ln1_w[None, :] + ln1_b[None, :]
    ln1_h = ln1.to(tl.float16)

    q = tl.full((NTOT, C), 0.0, tl.float32) + q_b[None, :]
    k = tl.full((NTOT, C), 0.0, tl.float32) + k_b[None, :]
    v = tl.full((NTOT, C), 0.0, tl.float32) + v_b[None, :]
    for ic in tl.range(0, C):
        ln_val = tl.sum(tl.where(c_idx[None, :] == ic, ln1_h, 0.0), axis=1).to(tl.float32)
        q_w = tl.load(q_w_ptr + c_idx * C + ic).to(tl.float32)
        k_w = tl.load(k_w_ptr + c_idx * C + ic).to(tl.float32)
        v_w = tl.load(v_w_ptr + c_idx * C + ic).to(tl.float32)
        q += ln_val[:, None] * q_w[None, :]
        k += ln_val[:, None] * k_w[None, :]
        v += ln_val[:, None] * v_w[None, :]

    q_h = (q * scale).to(tl.float16)
    k_h = k.to(tl.float16)
    v_h = v.to(tl.float16)

    attn = tl.full((NTOT, NTOT), 0.0, tl.float32)
    for ic2 in tl.range(0, C):
        q_val = tl.sum(tl.where(c_idx[None, :] == ic2, q_h, 0.0), axis=1).to(tl.float32)
        k_val = tl.sum(tl.where(c_idx[None, :] == ic2, k_h, 0.0), axis=1).to(tl.float32)
        attn += q_val[:, None] * k_val[None, :]

    same_win = w_local[:, None] == w_local[None, :]
    rpe_bcast = tl.load(rpe_ptr + tok_local[:, None] * N + (nt_idx[None, :] % N)).to(tl.float32)
    attn = attn + tl.where(same_win, rpe_bcast, 0.0)
    attn = tl.where(same_win, attn, -float("inf"))

    amax = tl.max(attn, axis=1)
    attn = attn - amax[:, None]
    attn_e = tl.exp(attn)
    asum = tl.sum(attn_e, axis=1)
    attn_p = (attn_e / asum[:, None]).to(tl.float16)

    out = tl.full((NTOT, C), 0.0, tl.float32)
    for jt in tl.range(0, NTOT):
        attn_val = tl.sum(tl.where(nt_idx[None, :] == jt, attn_p, 0.0), axis=1).to(tl.float32)
        v_val = tl.sum(tl.where(nt_idx[:, None] == jt, v_h, 0.0), axis=0).to(tl.float32)
        out += attn_val[:, None] * v_val[None, :]
    out_h = out.to(tl.float16)

    proj_out = tl.full((NTOT, C), 0.0, tl.float32) + proj_b[None, :]
    for ic3 in tl.range(0, C):
        out_val = tl.sum(tl.where(c_idx[None, :] == ic3, out_h, 0.0), axis=1).to(tl.float32)
        proj_w = tl.load(proj_w_ptr + c_idx * C + ic3).to(tl.float32)
        proj_out += out_val[:, None] * proj_w[None, :]

    x1 = proj_out + residual1
    residual2 = x1

    mean2 = tl.sum(x1, axis=1) / C
    xc2 = x1 - mean2[:, None]
    var2 = tl.sum(xc2 * xc2, axis=1) / C
    inv2 = 1.0 / tl.sqrt(var2 + 1e-5)
    ln2 = xc2 * inv2[:, None] * ln2_w[None, :] + ln2_b[None, :]
    ln2_h = ln2.to(tl.float16)

    h_act = tl.full((NTOT, HIDDEN), 0.0, tl.float32) + fc1_b[None, :]
    for ic4 in tl.range(0, C):
        ln2_val = tl.sum(tl.where(c_idx[None, :] == ic4, ln2_h, 0.0), axis=1).to(tl.float32)
        fc1_w = tl.load(fc1_w_ptr + h_idx * C + ic4).to(tl.float32)
        h_act += ln2_val[:, None] * fc1_w[None, :]

    h_gelu = 0.5 * h_act * (1.0 + tl.erf(h_act * 0.7071067811865475))
    h_gelu_h = h_gelu.to(tl.float16)

    fc2_out = tl.full((NTOT, C), 0.0, tl.float32) + fc2_b[None, :]
    for ihid in tl.range(0, HIDDEN):
        hidden_val = tl.sum(tl.where(h_idx[None, :] == ihid, h_gelu_h, 0.0), axis=1).to(tl.float32)
        fc2_w = tl.load(fc2_w_ptr + c_idx * HIDDEN + ihid).to(tl.float32)
        fc2_out += hidden_val[:, None] * fc2_w[None, :]

    y = fc2_out + residual2
    y_h = y.to(tl.float16)

    tl.store(Y_ptr + x_off, y_h, mask=mask2d)


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

        self._cached_qkv = None
        self._cached_device = None
        self._best_cfg = (1, 1, 2)

    def _get_qkv_split(self):
        C = self.dim
        dev = self.qkv.weight.device
        if self._cached_qkv is None or self._cached_device != dev:
            qkv_w = self.qkv.weight
            qkv_b = self.qkv.bias
            q_w = qkv_w[0:C].contiguous()
            k_w = qkv_w[C:2*C].contiguous()
            v_w = qkv_w[2*C:3*C].contiguous()
            q_b = qkv_b[0:C].contiguous()
            k_b = qkv_b[C:2*C].contiguous()
            v_b = qkv_b[2*C:3*C].contiguous()
            self._cached_qkv = (q_w, k_w, v_w, q_b, k_b, v_b)
            self._cached_device = dev
        return self._cached_qkv

    def _autotune(self, x, y, q_w, k_w, v_w, q_b, k_b, v_b, B, H, W, nH, nW, N):
        total_windows = B * nH * nW
        C = self.dim
        Wh = self.window_size[0]
        candidates = []
        for wpp in [1, 2]:
            if total_windows % wpp != 0:
                continue
            for nw in [1, 2, 4]:
                for ns in [2, 3]:
                    candidates.append((wpp, nw, ns))

        best_time = float("inf")
        best_cfg = (1, 1, 2)
        torch.cuda.synchronize()
        for (wpp, nw, ns) in candidates:
            grid = (triton.cdiv(total_windows, wpp),)
            try:
                for _ in range(2):
                    swin_block_kernel[grid](
                        x, y,
                        self.norm1.weight, self.norm1.bias,
                        q_w, q_b, k_w, k_b, v_w, v_b,
                        self.proj.weight, self.proj.bias,
                        self.rpe_bias,
                        self.norm2.weight, self.norm2.bias,
                        self.fc1.weight, self.fc1.bias,
                        self.fc2.weight, self.fc2.bias,
                        B, H, W, nH, nW,
                        C, self.hidden, N, Wh, wpp,
                        num_warps=nw, num_stages=ns,
                    )
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    swin_block_kernel[grid](
                        x, y,
                        self.norm1.weight, self.norm1.bias,
                        q_w, q_b, k_w, k_b, v_w, v_b,
                        self.proj.weight, self.proj.bias,
                        self.rpe_bias,
                        self.norm2.weight, self.norm2.bias,
                        self.fc1.weight, self.fc1.bias,
                        self.fc2.weight, self.fc2.bias,
                        B, H, W, nH, nW,
                        C, self.hidden, N, Wh, wpp,
                        num_warps=nw, num_stages=ns,
                    )
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end)
                if t < best_time:
                    best_time = t
                    best_cfg = (wpp, nw, ns)
            except Exception:
                continue
        return best_cfg

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        Wh, Ww = self.window_size
        assert Wh == Ww == 4
        nH, nW = H // Wh, W // Ww
        N = Wh * Ww

        x = x.contiguous()
        y = torch.empty_like(x)

        q_w, k_w, v_w, q_b, k_b, v_b = self._get_qkv_split()

        total_windows = B * nH * nW

        if self._best_cfg is None:
            self._best_cfg = self._autotune(
                x, y, q_w, k_w, v_w, q_b, k_b, v_b,
                B, H, W, nH, nW, N
            )

        WINDOWS_PER_PROG, NW, NS = self._best_cfg
        if total_windows % WINDOWS_PER_PROG != 0:
            WINDOWS_PER_PROG = 1

        grid = (triton.cdiv(total_windows, WINDOWS_PER_PROG),)

        swin_block_kernel[grid](
            x, y,
            self.norm1.weight, self.norm1.bias,
            q_w, q_b,
            k_w, k_b,
            v_w, v_b,
            self.proj.weight, self.proj.bias,
            self.rpe_bias,
            self.norm2.weight, self.norm2.bias,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            B, H, W,
            nH, nW,
            C,
            self.hidden,
            N, Wh,
            WINDOWS_PER_PROG,
            num_warps=NW,
            num_stages=NS,
        )
        return y
