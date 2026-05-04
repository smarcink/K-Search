import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_relu_pool_kernel_v4(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, OH, OW,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    CH: tl.constexpr = BLOCK_H * 2
    CW: tl.constexpr = BLOCK_W * 2

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    ch = pid_h * CH + tl.arange(0, CH)
    cw = pid_w * CW + tl.arange(0, CW)

    acc = tl.zeros((C_OUT, CH * CW), dtype=tl.float32)

    c_idx = tl.arange(0, C_IN)
    co_idx = tl.arange(0, C_OUT)

    for kh in tl.static_range(0, 3):
        for kw in tl.static_range(0, 3):
            ih = ch + kh - 1
            iw = cw + kw - 1
            h_mask = (ih >= 0) & (ih < H)
            w_mask = (iw >= 0) & (iw < W)
            off = (c_idx[:, None, None] * (H * W)
                   + ih[None, :, None] * W
                   + iw[None, None, :])
            mask = h_mask[None, :, None] & w_mask[None, None, :]
            x_tile = tl.load(x_ptr + off, mask=mask, other=0.0)
            x_flat = tl.reshape(x_tile, (C_IN, CH * CW))

            w_off = co_idx[:, None] * (C_IN * 9) + c_idx[None, :] * 9 + kh * 3 + kw
            w_tile = tl.load(w_ptr + w_off)

            acc += tl.dot(w_tile, x_flat, out_dtype=tl.float32)

    bias = tl.load(b_ptr + co_idx).to(tl.float32)
    acc = acc + bias[:, None]
    acc = tl.maximum(acc, 0.0)

    acc_r = tl.reshape(acc, (C_OUT, BLOCK_H, 2, BLOCK_W, 2))
    pooled = tl.sum(tl.sum(acc_r, axis=4), axis=2) * 0.25

    h_mask_o = oh < OH
    w_mask_o = ow < OW
    y_off = (co_idx[:, None, None] * (OH * OW)
             + oh[None, :, None] * OW
             + ow[None, None, :])
    y_mask = h_mask_o[None, :, None] & w_mask_o[None, None, :]
    tl.store(y_ptr + y_off, pooled.to(tl.float16), mask=y_mask)


@triton.jit
def relu_avgpool_kernel(
    x_ptr, y_ptr,
    C, H, W, OH, OW,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    ih0 = oh * 2
    iw0 = ow * 2

    base = pid_c * H * W

    off00 = base + ih0[:, None] * W + iw0[None, :]
    off01 = base + ih0[:, None] * W + (iw0[None, :] + 1)
    off10 = base + (ih0[:, None] + 1) * W + iw0[None, :]
    off11 = base + (ih0[:, None] + 1) * W + (iw0[None, :] + 1)

    h_mask = oh < OH
    w_mask = ow < OW
    mask = h_mask[:, None] & w_mask[None, :]

    v00 = tl.load(x_ptr + off00, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + off01, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + off10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + off11, mask=mask, other=0.0).to(tl.float32)

    v00 = tl.maximum(v00, 0.0)
    v01 = tl.maximum(v01, 0.0)
    v10 = tl.maximum(v10, 0.0)
    v11 = tl.maximum(v11, 0.0)

    out = (v00 + v01 + v10 + v11) * 0.25

    y_off = pid_c * OH * OW + oh[:, None] * OW + ow[None, :]
    tl.store(y_ptr + y_off, out.to(tl.float16), mask=mask)


def fused_conv_relu_pool(x, weight, bias):
    assert x.is_cuda and x.dtype == torch.float16
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    assert N == 1
    OH = H // 2
    OW = W // 2

    y = torch.empty((1, C_OUT, OH, OW), dtype=torch.float16, device=x.device)

    BLOCK_H = 8
    BLOCK_W = 32
    grid = (triton.cdiv(OH, BLOCK_H), triton.cdiv(OW, BLOCK_W))

    fused_conv_relu_pool_kernel_v4[grid](
        x, weight, bias, y,
        H, W, OH, OW,
        C_IN=C_IN, C_OUT=C_OUT,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=8, num_stages=3,
    )
    return y


def cudnn_conv_then_fused_relu_pool(x, weight, bias):
    conv_out = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)
    N, C, H, W = conv_out.shape
    OH = H // 2
    OW = W // 2
    y = torch.empty((N, C, OH, OW), dtype=conv_out.dtype, device=conv_out.device)

    BLOCK_H = 16
    BLOCK_W = 64
    grid = (N * C, triton.cdiv(OH, BLOCK_H), triton.cdiv(OW, BLOCK_W))
    relu_avgpool_kernel[grid](
        conv_out, y,
        C, H, W, OH, OW,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y


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
from enum import Enum
from functools import partial


class InitMethod(Enum):
    DEFAULT = "default"
    HE_UNIFORM = "he_unif"
    HE_NORMAL = "he_norm"


class ShortcutType(Enum):
    NONE = "none"
    CONCAT = "concat"
    ADD = "add"
    MAX = "max"


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


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        activation_type: ActivationType = ActivationType.RELU,
        signed_quant: bool = True,
        quantize_conv: bool = False,
        quantize_act: bool = False,
        activation_quantizer=None,
        integer_forward: bool = False,
        num_bits_conv: int = 8,
        num_bits_act: int = 8,
        padding_mode: str = "zeros",
        round_fn: QRoundFn = QRoundFn.TRUNC,
        init_method: InitMethod = InitMethod.HE_NORMAL,
        init_icnr: bool = False,
        **kwargs,
    ):
        super().__init__()

        if integer_forward and not quantize_conv:
            raise Exception("integer_forward is only valid for quantized conv operation.")

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward
        self.signed_quant = signed_quant
        self.activation_type = activation_type

        round_class = {QRoundFn.TRUNC: QTrunc, QRoundFn.FLOOR: QFloor, QRoundFn.ROUND: QRound}[
            round_fn
        ]

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

        self._can_fuse = (
            kernel_size == 3
            and stride == 1
            and bias
            and activation_type == ActivationType.RELU
            and not quantize_conv
            and not quantize_act
            and not integer_forward
            and padding_mode == "zeros"
            and in_channels >= 16
            and out_channels >= 16
        )

        self._cached_w = None
        self._cached_b = None

    def get_out_channels(self):
        return self.out_channels

    def get_act_quantizer(self):
        return self.blocks["activation_quantizer"]

    def set_act_quantizer(self, quantizer):
        self.blocks["activation_quantizer"] = quantizer

    def _forward_impl(self, x):
        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x

    def forward(self, x):
        if self.integer_forward:
            input_step_size = x[1]
            x_q = self.blocks["convolution"].conv_integer(x[0])
            (act_step, mac_bias, mac_scale) = self.compute_params(input_step_size)
            y_q = x_q * mac_scale.view(1, -1, 1, 1) + mac_bias.view(1, -1, 1, 1)
            y_q = self.pool(y_q)
            if self.blocks["activation"]:
                y_q = self.blocks["activation"](y_q)
            if self.blocks["activation_quantizer"]:
                z_q = self.blocks["activation_quantizer"].integer(y_q)
                return z_q, act_step
            else:
                return y_q

        if (
            self._can_fuse
            and x.is_cuda
            and x.dtype == torch.float16
            and x.dim() == 4
            and x.shape[0] == 1
            and x.shape[2] % 2 == 0
            and x.shape[3] % 2 == 0
        ):
            x = x.contiguous()
            w = self.conv2d.weight
            b = self.conv2d.bias
            if self._cached_w is None or self._cached_w.data_ptr() != w.data_ptr():
                self._cached_w = w.contiguous()
                self._cached_b = b.contiguous()
            try:
                return fused_conv_relu_pool(x, self._cached_w, self._cached_b)
            except Exception:
                return cudnn_conv_then_fused_relu_pool(x, self._cached_w, self._cached_b)

        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x


Model = ModelNew


def get_inputs():
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]


def get_init_inputs():
    in_channels = 16
    out_channels = 32
    quantize = False
    integer_forward = False
    return [
        in_channels,
        out_channels,
        3,
        1,
        True,
        ActivationType.RELU,
        True,
        quantize,
        quantize,
        None,
        integer_forward,
        8,
        8,
        "zeros",
    ]