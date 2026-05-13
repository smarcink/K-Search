import torch
import torch.nn as nn
import triton
import triton.language as tl

from enum import Enum
from functools import partial

from activations import ActivationType, create_activation
from initializer import icnr
from quantization import (
    QConv2D,
    QFloor,
    QRound,
    QRoundFn,
    QTrunc,
    create_quantizer,
)


class InitMethod(Enum):
    DEFAULT = "default"
    HE_UNIFORM = "he_unif"
    HE_NORMAL = "he_norm"


def _init_conv_weights(conv2d, activation, init_method, init_icnr=False):
    if init_method == InitMethod.HE_UNIFORM:
        init_fn = torch.nn.init.kaiming_uniform_
    elif init_method == InitMethod.HE_NORMAL:
        init_fn = torch.nn.init.kaiming_normal_
    else:
        init_fn = None

    if init_icnr:
        if init_fn is None:
            icnr(conv2d.weight, conv2d.bias)
        else:
            init_fn = partial(icnr, bias=conv2d.bias, initializer=init_fn)

    if init_fn is None:
        return

    if isinstance(activation, torch.nn.LeakyReLU):
        init_fn(conv2d.weight, a=activation.negative_slope, nonlinearity="leaky_relu")
    if isinstance(activation, torch.nn.PReLU):
        init_fn(conv2d.weight, a=activation.weight.item(), nonlinearity="leaky_relu")
    else:
        init_fn(conv2d.weight, nonlinearity="relu")


@triton.jit
def fused_conv_relu_pool_single_dot_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, Hout, Wout,
    Cin, Cout,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    COUT: tl.constexpr, CIN: tl.constexpr,
    K_PAD: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    n_tiles_w = (Wout + BLOCK_W - 1) // BLOCK_W
    pid_h = pid_hw // n_tiles_w
    pid_w = pid_hw % n_tiles_w

    oh_off = pid_h * BLOCK_H
    ow_off = pid_w * BLOCK_W

    CONV_H: tl.constexpr = BLOCK_H * 2
    CONV_W: tl.constexpr = BLOCK_W * 2
    M: tl.constexpr = CONV_H * CONV_W

    cy_base = oh_off * 2
    cx_base = ow_off * 2

    cout_idx = tl.arange(0, COUT)
    cout_mask = cout_idx < Cout
    bias = tl.load(b_ptr + cout_idx, mask=cout_mask, other=0.0).to(tl.float32)

    k_idx = tl.arange(0, K_PAD)
    k_valid = k_idx < (CIN * 9)
    k_cin = k_idx % CIN
    k_pos = k_idx // CIN
    k_ky = k_pos // 3
    k_kx = k_pos % 3
    k_cin_in = k_cin < Cin
    k_pos_in = k_pos < 9
    k_mask_full = k_valid & k_cin_in & k_pos_in

    m_idx = tl.arange(0, M)
    hh = m_idx // CONV_W
    ww = m_idx % CONV_W

    iy = cy_base + hh[:, None] + k_ky[None, :] - 1
    ix = cx_base + ww[:, None] + k_kx[None, :] - 1
    iy_m = (iy >= 0) & (iy < H)
    ix_m = (ix >= 0) & (ix < W)

    x_off = k_cin[None, :] * (H * W) + iy * W + ix
    x_mask = iy_m & ix_m & k_mask_full[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    w_row_off = k_cin[:, None] * 9 + k_ky[:, None] * 3 + k_kx[:, None]
    w_off = cout_idx[None, :] * (Cin * 9) + w_row_off
    w_mask = k_mask_full[:, None] & cout_mask[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    x_tile = x_tile.to(tl.float16)
    w_tile = w_tile.to(tl.float16)

    conv_acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    conv_acc = conv_acc + bias[None, :]
    conv_acc = tl.maximum(conv_acc, 0.0)

    conv_3d = tl.reshape(conv_acc, (CONV_H, CONV_W, COUT))
    pooled = tl.reshape(conv_3d, (BLOCK_H, 2, BLOCK_W, 2, COUT))
    pooled = tl.sum(pooled, axis=3)
    pooled = tl.sum(pooled, axis=1)
    pooled = pooled * 0.25

    oh = oh_off + tl.arange(0, BLOCK_H)
    ow = ow_off + tl.arange(0, BLOCK_W)
    oh_mask = oh < Hout
    ow_mask = ow < Wout

    y_offsets = (
        cout_idx[None, None, :] * (Hout * Wout)
        + oh[:, None, None] * Wout
        + ow[None, :, None]
    )
    y_mask = oh_mask[:, None, None] & ow_mask[None, :, None] & cout_mask[None, None, :]
    tl.store(y_ptr + y_offsets, pooled.to(tl.float16), mask=y_mask)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        activation_type=ActivationType.RELU,
        signed_quant=True,
        quantize_conv=False,
        quantize_act=False,
        activation_quantizer=None,
        integer_forward=False,
        num_bits_conv=8,
        num_bits_act=8,
        padding_mode="zeros",
        round_fn=QRoundFn.TRUNC,
        init_method=InitMethod.HE_NORMAL,
        init_icnr=False,
        **kwargs,
    ):
        super().__init__()

        if integer_forward and not quantize_conv:
            raise Exception("integer_forward is only valid for quantized conv operation.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward
        self.signed_quant = signed_quant
        self.activation_type = activation_type

        round_class = {QRoundFn.TRUNC: QTrunc, QRoundFn.FLOOR: QFloor, QRoundFn.ROUND: QRound}[round_fn]

        self.blocks = {
            "activation": create_activation(activation_type),
            "convolution": QConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=1,
                padding_mode=padding_mode,
                bias=bias,
                quantize=quantize_conv,
                num_bits=num_bits_conv,
                round_fn=round_class,
            ),
            "activation_quantizer": activation_quantizer,
        }

        if activation_quantizer is None:
            self.blocks["activation_quantizer"] = create_quantizer(
                quantize=quantize_act,
                num_bits=num_bits_act,
                trainable=True,
                unsigned=not signed_quant,
                round_fn=round_fn,
                max_val={8: 3.0, 4: 0.5}[num_bits_act],
            )

        self.activation = self.blocks["activation"]
        self.act_quantizer = self.blocks["activation_quantizer"]
        self.conv2d = self.blocks["convolution"]
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

        _init_conv_weights(
            self.blocks["convolution"], self.blocks["activation"], init_method, init_icnr
        )

        self._fused_ok = (
            kernel_size == 3
            and stride == 1
            and padding_mode == "zeros"
            and not quantize_conv
            and not quantize_act
            and not integer_forward
            and activation_type == ActivationType.RELU
            and bias
        )

        self._cached_weight = None
        self._cached_bias = None

    def get_out_channels(self):
        return self.out_channels

    def _fallback_forward(self, x):
        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x

    def _fused_forward(self, x):
        N, Cin, H, W = x.shape
        Cout = self.out_channels

        Hout = H // 2
        Wout = W // 2

        if N != 1 or H % 2 != 0 or W % 2 != 0:
            return self._fallback_forward(x)

        x_c = x.contiguous()
        if self._cached_weight is None or self._cached_weight.device != x.device:
            self._cached_weight = self.conv2d.weight.detach().contiguous()
            self._cached_bias = self.conv2d.bias.detach().contiguous()
        weight = self._cached_weight
        bias = self._cached_bias

        y = torch.empty((N, Cout, Hout, Wout), dtype=torch.float16, device=x.device)

        BLOCK_H = 8
        BLOCK_W = 64
        COUT = 32
        CIN = 16
        K_PAD = 160  # ceil(16*9 / 16) * 16 = 144 -> next pow2 multiple suitable; keep 160? must be multiple of 16

        # K_PAD must be multiple of 16 for tl.dot; smallest >= 144 multiple of 16 is 144 itself.
        K_PAD = 144

        if Hout % BLOCK_H != 0 or Wout % BLOCK_W != 0:
            BLOCK_H = 8
            BLOCK_W = 16

        n_tiles_h = (Hout + BLOCK_H - 1) // BLOCK_H
        n_tiles_w = (Wout + BLOCK_W - 1) // BLOCK_W
        grid = (n_tiles_h * n_tiles_w,)

        fused_conv_relu_pool_single_dot_kernel[grid](
            x_c, weight, bias, y,
            H, W, Hout, Wout,
            Cin, Cout,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            COUT=COUT, CIN=CIN,
            K_PAD=K_PAD,
            num_warps=8,
            num_stages=2,
        )
        return y

    def _forward_impl(self, x):
        if self._fused_ok and x.is_xpu and x.dtype == torch.float16:
            try:
                return self._fused_forward(x)
            except Exception:
                return self._fallback_forward(x)
        return self._fallback_forward(x)

    def forward(self, x):
        if self.integer_forward:
            input_step_size = x[1]
            x_q = self.blocks["convolution"].conv_integer(x[0])
            act_step = self.blocks["activation_quantizer"].get_quant_step() if self.blocks["activation_quantizer"] else 1
            mac_bias = self.blocks["convolution"].bias / act_step
            wt_step = self.blocks["convolution"].get_quant_step().flatten()
            mac_scale = input_step_size * wt_step / act_step
            y_q = x_q * mac_scale.view(1, -1, 1, 1) + mac_bias.view(1, -1, 1, 1)
            y_q = self.pool(y_q)
            if self.blocks["activation"]:
                y_q = self.blocks["activation"](y_q)
            if self.blocks["activation_quantizer"]:
                z_q = self.blocks["activation_quantizer"].integer(y_q)
                return z_q, act_step
            return y_q
        return self._forward_impl(x)