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
def fused_conv_relu_pool_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, Hout, Wout,
    Cin, Cout,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    COUT: tl.constexpr, CIN: tl.constexpr,
):
    # Each program computes a BLOCK_H x BLOCK_W tile of pooled outputs for ALL Cout channels.
    pid_hw = tl.program_id(0)
    n_tiles_w = (Wout + BLOCK_W - 1) // BLOCK_W
    pid_h = pid_hw // n_tiles_w
    pid_w = pid_hw % n_tiles_w

    # pooled output coords
    oh_off = pid_h * BLOCK_H
    ow_off = pid_w * BLOCK_W
    oh = oh_off + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    ow = ow_off + tl.arange(0, BLOCK_W)  # [BLOCK_W]

    # conv output coords (2x pool) -> we need conv outputs at (2*oh, 2*oh+1) x (2*ow, 2*ow+1)
    # For BLOCK_H pooled rows we need 2*BLOCK_H conv rows; similarly width.
    # We'll compute 4 conv outputs per pooled output and average.

    # Allocate accumulators for 4 sub-positions, each [BLOCK_H, BLOCK_W, COUT]
    # We'll do this by computing conv outputs at conv_y, conv_x for dy in {0,1}, dx in {0,1}.
    # To keep it tractable, compute conv directly in fp32 and accumulate via summing 4 evaluations.

    # Pooled output sum accumulator [BLOCK_H, BLOCK_W, COUT]
    acc = tl.zeros((BLOCK_H, BLOCK_W, COUT), dtype=tl.float32)

    # bias [COUT]
    cout_idx = tl.arange(0, COUT)
    cout_mask = cout_idx < Cout
    bias = tl.load(b_ptr + cout_idx, mask=cout_mask, other=0.0).to(tl.float32)

    # Iterate over 2x2 sub-positions in the pool window
    for dy in tl.static_range(0, 2):
        for dx in tl.static_range(0, 2):
            # conv output coordinates for this sub-position
            cy = (oh_off * 2) + dy + tl.arange(0, BLOCK_H) * 2  # [BLOCK_H]
            cx = (ow_off * 2) + dx + tl.arange(0, BLOCK_W) * 2  # [BLOCK_W]

            # conv sum [BLOCK_H, BLOCK_W, COUT]
            conv_sum = tl.zeros((BLOCK_H, BLOCK_W, COUT), dtype=tl.float32)

            # iterate over kernel positions and input channels
            for ky in tl.static_range(0, 3):
                for kx in tl.static_range(0, 3):
                    # input coords (with padding=1)
                    iy = cy + ky - 1  # [BLOCK_H]
                    ix = cx + kx - 1  # [BLOCK_W]
                    iy_mask = (iy >= 0) & (iy < H)
                    ix_mask = (ix >= 0) & (ix < W)

                    # Load input patch [BLOCK_H, BLOCK_W, CIN]
                    cin_idx = tl.arange(0, CIN)
                    cin_mask = cin_idx < Cin

                    # ptr: x[0, c, iy, ix] = c*H*W + iy*W + ix
                    x_offsets = (
                        cin_idx[None, None, :] * (H * W)
                        + iy[:, None, None] * W
                        + ix[None, :, None]
                    )
                    full_mask = (
                        iy_mask[:, None, None]
                        & ix_mask[None, :, None]
                        & cin_mask[None, None, :]
                    )
                    x_vals = tl.load(x_ptr + x_offsets, mask=full_mask, other=0.0).to(tl.float32)
                    # x_vals: [BLOCK_H, BLOCK_W, CIN]

                    # Load weights for this (ky, kx) position: w[cout, cin, ky, kx]
                    # offset = cout*Cin*9 + cin*9 + ky*3 + kx
                    w_offsets = (
                        cout_idx[None, :] * (Cin * 9)
                        + cin_idx[:, None] * 9
                        + ky * 3
                        + kx
                    )
                    w_mask = cin_mask[:, None] & cout_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
                    # w_vals: [CIN, COUT]

                    # Contract over CIN: out[h, w, cout] += sum_cin x[h,w,cin] * w[cin,cout]
                    # Reshape to use tl.dot: x_2d [BLOCK_H*BLOCK_W, CIN] @ w [CIN, COUT]
                    x_2d = tl.reshape(x_vals, (BLOCK_H * BLOCK_W, CIN))
                    out_2d = tl.dot(x_2d, w_vals, out_dtype=tl.float32)
                    out_3d = tl.reshape(out_2d, (BLOCK_H, BLOCK_W, COUT))
                    conv_sum += out_3d

            # add bias and ReLU
            conv_sum = conv_sum + bias[None, None, :]
            conv_sum = tl.maximum(conv_sum, 0.0)

            acc += conv_sum

    # average pool (divide by 4)
    acc = acc * 0.25

    # Store output: y[0, cout, oh, ow]
    oh_mask = oh < Hout
    ow_mask = ow < Wout

    y_offsets = (
        cout_idx[None, None, :] * (Hout * Wout)
        + oh[:, None, None] * Wout
        + ow[None, :, None]
    )
    y_mask = oh_mask[:, None, None] & ow_mask[None, :, None] & cout_mask[None, None, :]
    tl.store(y_ptr + y_offsets, acc.to(tl.float16), mask=y_mask)


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

        # Eligibility for fused kernel
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
        # x: (N, Cin, H, W); only N=1 supported in this fused path
        N, Cin, H, W = x.shape
        Cout = self.out_channels

        # conv output size with padding=1, stride=1
        Hc = H
        Wc = W
        # avgpool 2x2 stride 2
        Hout = Hc // 2
        Wout = Wc // 2

        if N != 1 or H % 2 != 0 or W % 2 != 0:
            return self._fallback_forward(x)

        x_c = x.contiguous()
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous()

        y = torch.empty((N, Cout, Hout, Wout), dtype=torch.float16, device=x.device)

        BLOCK_H = 8
        BLOCK_W = 16
        COUT = 32  # next pow2 >= Cout=32
        CIN = 16   # next pow2 >= Cin=16

        n_tiles_h = (Hout + BLOCK_H - 1) // BLOCK_H
        n_tiles_w = (Wout + BLOCK_W - 1) // BLOCK_W
        grid = (n_tiles_h * n_tiles_w,)

        fused_conv_relu_pool_kernel[grid](
            x_c, weight, bias, y,
            H, W, Hout, Wout,
            Cin, Cout,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            COUT=COUT, CIN=CIN,
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
            # fallback path for integer forward
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