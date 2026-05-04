import math
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional
import triton
import triton.language as tl


@triton.jit
def fused_ln_window_attn_kernel(
    X_ptr, LNW_ptr, LNB_ptr,
    QKVW_ptr, QKVB_ptr,
    PROJW_ptr, PROJB_ptr,
    RPB_ptr, OUT_ptr,
    B, H, W,
    nW_h, nW_w,
    WS0: tl.constexpr, WS1: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr,
    LN_EPS: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * nW_h * nW_w
    if pid >= total:
        return

    win_w = pid % nW_w
    tmp = pid // nW_w
    win_h = tmp % nW_h
    b = tmp // nW_h

    col_idx = tl.arange(0, C)
    tok_idx = tl.arange(0, N)

    h_in_win = tok_idx // WS1
    w_in_win = tok_idx % WS1
    h_abs = win_h * WS0 + h_in_win
    w_abs = win_w * WS1 + w_in_win

    x_base = b * (H * W * C)
    x_row_off = h_abs * (W * C) + w_abs * C
    x_ptrs = X_ptr + x_base + x_row_off[:, None] + col_idx[None, :]
    x_raw = tl.load(x_ptrs).to(tl.float32)

    mean = tl.sum(x_raw, axis=1) / C
    xc = x_raw - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    lnw = tl.load(LNW_ptr + col_idx).to(tl.float32)
    lnb = tl.load(LNB_ptr + col_idx).to(tl.float32)
    x = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]

    k_idx = tl.arange(0, C)

    qw = tl.load(QKVW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    qb = tl.load(QKVB_ptr + col_idx).to(tl.float32)
    q = tl.dot(x, tl.trans(qw)) + qb[None, :]

    kw = tl.load(QKVW_ptr + (col_idx[:, None] + C) * C + k_idx[None, :]).to(tl.float32)
    kb = tl.load(QKVB_ptr + col_idx + C).to(tl.float32)
    k = tl.dot(x, tl.trans(kw)) + kb[None, :]

    vw = tl.load(QKVW_ptr + (col_idx[:, None] + 2 * C) * C + k_idx[None, :]).to(tl.float32)
    vb = tl.load(QKVB_ptr + col_idx + 2 * C).to(tl.float32)
    v = tl.dot(x, tl.trans(vw)) + vb[None, :]

    q_scaled = q * SCALE
    attn = tl.dot(q_scaled.to(tl.float16), tl.trans(k).to(tl.float16)).to(tl.float32)

    rpb_ptrs = RPB_ptr + tok_idx[:, None] * N + tl.arange(0, N)[None, :]
    rpb = tl.load(rpb_ptrs).to(tl.float32)
    attn = attn + rpb

    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]

    out_h = tl.dot(attn_sm.to(tl.float16), v.to(tl.float16)).to(tl.float32)

    pw = tl.load(PROJW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    pb = tl.load(PROJB_ptr + col_idx).to(tl.float32)
    out = tl.dot(out_h.to(tl.float16), tl.trans(pw).to(tl.float16)).to(tl.float32) + pb[None, :]

    out_ptrs = OUT_ptr + x_base + x_row_off[:, None] + col_idx[None, :]
    tl.store(out_ptrs, out.to(tl.float16))


@triton.jit
def fused_ln_window_attn_residual_kernel(
    X_ptr, LNW_ptr, LNB_ptr,
    QKVW_ptr, QKVB_ptr,
    PROJW_ptr, PROJB_ptr,
    RPB_ptr, OUT_ptr,
    B, H, W,
    nW_h, nW_w,
    WS0: tl.constexpr, WS1: tl.constexpr,
    N: tl.constexpr, C: tl.constexpr,
    LN_EPS: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * nW_h * nW_w
    if pid >= total:
        return

    win_w = pid % nW_w
    tmp = pid // nW_w
    win_h = tmp % nW_h
    b = tmp // nW_h

    col_idx = tl.arange(0, C)
    tok_idx = tl.arange(0, N)

    h_in_win = tok_idx // WS1
    w_in_win = tok_idx % WS1
    h_abs = win_h * WS0 + h_in_win
    w_abs = win_w * WS1 + w_in_win

    x_base = b * (H * W * C)
    x_row_off = h_abs * (W * C) + w_abs * C
    x_ptrs = X_ptr + x_base + x_row_off[:, None] + col_idx[None, :]
    x_raw_fp16 = tl.load(x_ptrs)
    x_raw = x_raw_fp16.to(tl.float32)

    mean = tl.sum(x_raw, axis=1) / C
    xc = x_raw - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    lnw = tl.load(LNW_ptr + col_idx).to(tl.float32)
    lnb = tl.load(LNB_ptr + col_idx).to(tl.float32)
    x = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]

    k_idx = tl.arange(0, C)

    qw = tl.load(QKVW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    qb = tl.load(QKVB_ptr + col_idx).to(tl.float32)
    q = tl.dot(x, tl.trans(qw)) + qb[None, :]

    kw = tl.load(QKVW_ptr + (col_idx[:, None] + C) * C + k_idx[None, :]).to(tl.float32)
    kb = tl.load(QKVB_ptr + col_idx + C).to(tl.float32)
    k = tl.dot(x, tl.trans(kw)) + kb[None, :]

    vw = tl.load(QKVW_ptr + (col_idx[:, None] + 2 * C) * C + k_idx[None, :]).to(tl.float32)
    vb = tl.load(QKVB_ptr + col_idx + 2 * C).to(tl.float32)
    v = tl.dot(x, tl.trans(vw)) + vb[None, :]

    q_scaled = q * SCALE
    attn = tl.dot(q_scaled.to(tl.float16), tl.trans(k).to(tl.float16)).to(tl.float32)

    rpb_ptrs = RPB_ptr + tok_idx[:, None] * N + tl.arange(0, N)[None, :]
    rpb = tl.load(rpb_ptrs).to(tl.float32)
    attn = attn + rpb

    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]

    out_h = tl.dot(attn_sm.to(tl.float16), v.to(tl.float16)).to(tl.float32)

    pw = tl.load(PROJW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    pb = tl.load(PROJB_ptr + col_idx).to(tl.float32)
    out = tl.dot(out_h.to(tl.float16), tl.trans(pw).to(tl.float16)).to(tl.float32) + pb[None, :]
    out = out + x_raw

    out_ptrs = OUT_ptr + x_base + x_row_off[:, None] + col_idx[None, :]
    tl.store(out_ptrs, out.to(tl.float16))


@triton.jit
def fused_window_attn_kernel(
    X_ptr, QKVW_ptr, QKVB_ptr, PROJW_ptr, PROJB_ptr, RPB_ptr, OUT_ptr,
    B_total,
    N: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B_total:
        return
    row_idx = tl.arange(0, N)
    col_idx = tl.arange(0, C)
    x_offset = pid * N * C
    x_ptrs = X_ptr + x_offset + row_idx[:, None] * C + col_idx[None, :]
    x = tl.load(x_ptrs).to(tl.float32)

    k_idx = tl.arange(0, C)
    qw = tl.load(QKVW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    qb = tl.load(QKVB_ptr + col_idx).to(tl.float32)
    q = tl.dot(x, tl.trans(qw)) + qb[None, :]

    kw = tl.load(QKVW_ptr + (col_idx[:, None] + C) * C + k_idx[None, :]).to(tl.float32)
    kb = tl.load(QKVB_ptr + col_idx + C).to(tl.float32)
    k = tl.dot(x, tl.trans(kw)) + kb[None, :]

    vw = tl.load(QKVW_ptr + (col_idx[:, None] + 2 * C) * C + k_idx[None, :]).to(tl.float32)
    vb = tl.load(QKVB_ptr + col_idx + 2 * C).to(tl.float32)
    v = tl.dot(x, tl.trans(vw)) + vb[None, :]

    q_scaled = q * SCALE
    attn = tl.dot(q_scaled.to(tl.float16), tl.trans(k).to(tl.float16)).to(tl.float32)
    rpb = tl.load(RPB_ptr + row_idx[:, None] * N + tl.arange(0, N)[None, :]).to(tl.float32)
    attn = attn + rpb
    attn_max = tl.max(attn, axis=1)
    attn = attn - attn_max[:, None]
    attn_exp = tl.exp(attn)
    attn_sum = tl.sum(attn_exp, axis=1)
    attn_sm = attn_exp / attn_sum[:, None]
    out_h = tl.dot(attn_sm.to(tl.float16), v.to(tl.float16)).to(tl.float32)

    pw = tl.load(PROJW_ptr + col_idx[:, None] * C + k_idx[None, :]).to(tl.float32)
    pb = tl.load(PROJB_ptr + col_idx).to(tl.float32)
    out = tl.dot(out_h.to(tl.float16), tl.trans(pw).to(tl.float16)).to(tl.float32) + pb[None, :]
    out_ptrs = OUT_ptr + x_offset + row_idx[:, None] * C + col_idx[None, :]
    tl.store(out_ptrs, out.to(tl.float16))


@triton.jit
def fused_ln_mlp_residual_kernel(
    X_ptr, LNW_ptr, LNB_ptr,
    W1_ptr, B1_ptr, W2_ptr, B2_ptr,
    OUT_ptr,
    M, C: tl.constexpr, H: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    col_idx = tl.arange(0, C)
    x_ptrs = X_ptr + rows[:, None] * C + col_idx[None, :]
    x_raw_fp16 = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
    x_raw = x_raw_fp16.to(tl.float32)

    mean = tl.sum(x_raw, axis=1) / C
    xc = x_raw - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    lnw = tl.load(LNW_ptr + col_idx).to(tl.float32)
    lnb = tl.load(LNB_ptr + col_idx).to(tl.float32)
    xn = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]

    h_idx = tl.arange(0, H)
    w1 = tl.load(W1_ptr + h_idx[:, None] * C + col_idx[None, :])
    b1 = tl.load(B1_ptr + h_idx).to(tl.float32)
    y = tl.dot(xn.to(tl.float16), tl.trans(w1)).to(tl.float32) + b1[None, :]

    inv_sqrt2 = 0.70710678118654752440
    y_gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    w2 = tl.load(W2_ptr + col_idx[:, None] * H + h_idx[None, :])
    b2 = tl.load(B2_ptr + col_idx).to(tl.float32)
    z = tl.dot(y_gelu.to(tl.float16), tl.trans(w2)).to(tl.float32) + b2[None, :]

    z = z + x_raw

    out_ptrs = OUT_ptr + rows[:, None] * C + col_idx[None, :]
    tl.store(out_ptrs, z.to(tl.float16), mask=row_mask[:, None])


def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input
    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: list
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: list,
        shift_size: list,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size
        )

    def forward_fused_with_ln_residual(self, x: Tensor, ln_weight: Tensor, ln_bias: Tensor, ln_eps: float) -> Optional[Tensor]:
        B, H, W, C = x.shape
        ws0, ws1 = self.window_size
        if (self.shift_size[0] != 0 or self.shift_size[1] != 0):
            return None
        if (H % ws0 != 0) or (W % ws1 != 0):
            return None
        if self.num_heads != 1 or C != 32 or (ws0 * ws1) != 16:
            return None

        rpb = self.get_relative_position_bias().squeeze(0).contiguous()
        nW_h = H // ws0
        nW_w = W // ws1
        N = ws0 * ws1
        scale = (C // self.num_heads) ** -0.5

        out = torch.empty_like(x)
        grid = (B * nW_h * nW_w,)
        fused_ln_window_attn_residual_kernel[grid](
            x, ln_weight, ln_bias,
            self.qkv.weight, self.qkv.bias,
            self.proj.weight, self.proj.bias,
            rpb, out,
            B, H, W, nW_h, nW_w,
            WS0=ws0, WS1=ws1, N=N, C=C,
            LN_EPS=ln_eps, SCALE=scale,
            num_warps=2, num_stages=2,
        )
        return out

    def forward_fused_with_ln(self, x: Tensor, ln_weight: Tensor, ln_bias: Tensor, ln_eps: float) -> Optional[Tensor]:
        B, H, W, C = x.shape
        ws0, ws1 = self.window_size
        if (self.shift_size[0] != 0 or self.shift_size[1] != 0):
            return None
        if (H % ws0 != 0) or (W % ws1 != 0):
            return None
        if self.num_heads != 1 or C != 32 or (ws0 * ws1) != 16:
            return None

        rpb = self.get_relative_position_bias().squeeze(0).contiguous()
        nW_h = H // ws0
        nW_w = W // ws1
        N = ws0 * ws1
        scale = (C // self.num_heads) ** -0.5

        out = torch.empty_like(x)
        grid = (B * nW_h * nW_w,)
        fused_ln_window_attn_kernel[grid](
            x, ln_weight, ln_bias,
            self.qkv.weight, self.qkv.bias,
            self.proj.weight, self.proj.bias,
            rpb, out,
            B, H, W, nW_h, nW_w,
            WS0=ws0, WS1=ws1, N=N, C=C,
            LN_EPS=ln_eps, SCALE=scale,
            num_warps=2, num_stages=2,
        )
        return out

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        B, H, W, C = x.shape
        ws0, ws1 = self.window_size
        pad_r = (ws1 - W % ws1) % ws1
        pad_b = (ws0 - H % ws0) % ws0
        if pad_r or pad_b:
            xp = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        else:
            xp = x
        _, pad_H, pad_W, _ = xp.shape
        shift_size = list(self.shift_size)
        if ws0 >= pad_H:
            shift_size[0] = 0
        if ws1 >= pad_W:
            shift_size[1] = 0
        do_shift = sum(shift_size) > 0
        if do_shift:
            xp = torch.roll(xp, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        nW = (pad_H // ws0) * (pad_W // ws1)
        N = ws0 * ws1
        xp = xp.view(B, pad_H // ws0, ws0, pad_W // ws1, ws1, C)
        xp = xp.permute(0, 1, 3, 2, 4, 5).reshape(B * nW, N, C).contiguous()
        scale = (C // self.num_heads) ** -0.5
        can_fuse = (not do_shift) and (self.num_heads == 1) and (C == 32) and (N == 16)
        if can_fuse:
            rpb = relative_position_bias.squeeze(0).contiguous()
            x_out = torch.empty_like(xp)
            grid = (xp.shape[0],)
            fused_window_attn_kernel[grid](
                xp, self.qkv.weight, self.qkv.bias, self.proj.weight, self.proj.bias,
                rpb, x_out, xp.shape[0], N=N, C=C, SCALE=scale,
                num_warps=2, num_stages=2,
            )
        else:
            qkv = F.linear(xp, self.qkv.weight, self.qkv.bias)
            qkv = qkv.reshape(xp.size(0), xp.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * scale
            attn = q.matmul(k.transpose(-2, -1))
            attn = attn + relative_position_bias
            attn = F.softmax(attn, dim=-1)
            x_out = attn.matmul(v).transpose(1, 2).reshape(xp.size(0), xp.size(1), C)
            x_out = F.linear(x_out, self.proj.weight, self.proj.bias)
        x_out = x_out.view(B, pad_H // ws0, pad_W // ws1, ws0, ws1, C)
        x_out = x_out.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        if do_shift:
            x_out = torch.roll(x_out, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        if pad_r or pad_b:
            x_out = x_out[:, :H, :W, :].contiguous()
        return x_out


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        super().__init__(*layers)


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list,
        shift_size: list,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.half()
        self._dim = dim
        self._hidden = int(dim * mlp_ratio)

    def _fused_ln_mlp_residual_single(self, x: Tensor) -> Optional[Tensor]:
        if self.mlp[0].__class__.__name__ != 'Linear':
            return None
        fc1 = self.mlp[0]
        fc2 = self.mlp[3]
        B, H, W, C = x.shape
        if C != self._dim:
            return None
        M = B * H * W
        Hdim = self._hidden
        if Hdim != 4 * C:
            return None
        if C != 32 or Hdim != 128:
            return None

        x_flat = x.reshape(M, C)
        out = torch.empty((M, C), dtype=torch.float16, device=x.device)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        fused_ln_mlp_residual_kernel[grid](
            x_flat, self.norm2.weight, self.norm2.bias,
            fc1.weight, fc1.bias, fc2.weight, fc2.bias,
            out,
            M, C=C, H=Hdim, LN_EPS=float(self.norm2.eps),
            BLOCK_M=BLOCK_M, num_warps=4, num_stages=2,
        )
        return out.view(B, H, W, C)

    def forward(self, x: Tensor):
        fused_out = None
        sd_is_identity = (not self.training) or (self.stochastic_depth.p == 0.0)
        if isinstance(self.attn, ShiftedWindowAttention) and sd_is_identity:
            fused_out = self.attn.forward_fused_with_ln_residual(
                x, self.norm1.weight, self.norm1.bias, float(self.norm1.eps)
            )
        if fused_out is not None:
            x = fused_out
        else:
            attn_out = None
            if isinstance(self.attn, ShiftedWindowAttention):
                attn_out = self.attn.forward_fused_with_ln(
                    x, self.norm1.weight, self.norm1.bias, float(self.norm1.eps)
                )
            if attn_out is not None:
                x = x + self.stochastic_depth(attn_out)
            else:
                x = x + self.stochastic_depth(self.attn(self.norm1(x)))

        mlp_full = None
        if sd_is_identity and (not self.training or (self.mlp[-1].p == 0.0 if hasattr(self.mlp[-1], 'p') else True)):
            try:
                mlp_full = self._fused_ln_mlp_residual_single(x)
            except Exception:
                mlp_full = None
        if mlp_full is not None:
            x = mlp_full
        else:
            x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


# --- Adapters for compare_swin_fp16.py ---

Model = ModelNew


def get_inputs():
    return [torch.randn(1, 720, 1280, 32, dtype=torch.float16)]


def get_init_inputs():
    return [32, 1, [4, 4], [0, 0], 4]
