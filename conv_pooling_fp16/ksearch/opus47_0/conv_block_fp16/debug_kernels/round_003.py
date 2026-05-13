import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    from activations import ActivationType, create_activation
    from quantization import QConv2D, QRoundFn, QTrunc, QFloor, QRound, create_quantizer
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False
    from enum import Enum

    class ActivationType(Enum):
        RELU = "relu"

    def create_activation(t):
        return torch.nn.ReLU()

    class QRoundFn(Enum):
        TRUNC = "trunc"
        FLOOR = "floor"
        ROUND = "round"

    QTrunc = QFloor = QRound = None

    def create_quantizer(**kwargs):
        return None

    class QConv2D(torch.nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                     stride=1, groups=1, padding_mode="zeros", bias=True,
                     quantize=False, num_bits=8, round_fn=None):
            super().__init__(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, groups=groups,
                             bias=bias, padding_mode=padding_mode)


@triton.jit
def fused_conv_relu_pool_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, OH, OW,
    stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yc, stride_yh, stride_yw,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    C_IN,
):
    pid_oc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_tiles = tl.cdiv(OW, BLOCK_W)
    pid_h = pid_hw // num_w_tiles
    pid_w = pid_hw % num_w_tiles

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    offs_h = oh_start + tl.arange(0, BLOCK_H)
    offs_w = ow_start + tl.arange(0, BLOCK_W)

    conv_h0 = offs_h * 2
    conv_w0 = offs_w * 2

    acc00 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    h_in_r = conv_h0[:, None] - 1
    w_in_c = conv_w0[None, :] - 1

    for ic in range(0, C_IN):
        w_base = pid_oc * stride_wo + ic * stride_wi
        w00 = tl.load(w_ptr + w_base + 0 * stride_wkh + 0 * stride_wkw).to(tl.float32)
        w01 = tl.load(w_ptr + w_base + 0 * stride_wkh + 1 * stride_wkw).to(tl.float32)
        w02 = tl.load(w_ptr + w_base + 0 * stride_wkh + 2 * stride_wkw).to(tl.float32)
        w10 = tl.load(w_ptr + w_base + 1 * stride_wkh + 0 * stride_wkw).to(tl.float32)
        w11 = tl.load(w_ptr + w_base + 1 * stride_wkh + 1 * stride_wkw).to(tl.float32)
        w12 = tl.load(w_ptr + w_base + 1 * stride_wkh + 2 * stride_wkw).to(tl.float32)
        w20 = tl.load(w_ptr + w_base + 2 * stride_wkh + 0 * stride_wkw).to(tl.float32)
        w21 = tl.load(w_ptr + w_base + 2 * stride_wkh + 1 * stride_wkw).to(tl.float32)
        w22 = tl.load(w_ptr + w_base + 2 * stride_wkh + 2 * stride_wkw).to(tl.float32)

        x_base = ic * stride_xc

        # Load 4x4 region of input per pooled pixel
        # rows r in [0..3], cols c in [0..3]
        # row 0
        h0 = h_in_r + 0
        m_h0 = (h0 >= 0) & (h0 < H)
        h1 = h_in_r + 1
        m_h1 = (h1 >= 0) & (h1 < H)
        h2 = h_in_r + 2
        m_h2 = (h2 >= 0) & (h2 < H)
        h3 = h_in_r + 3
        m_h3 = (h3 >= 0) & (h3 < H)

        c0 = w_in_c + 0
        m_c0 = (c0 >= 0) & (c0 < W)
        c1 = w_in_c + 1
        m_c1 = (c1 >= 0) & (c1 < W)
        c2 = w_in_c + 2
        m_c2 = (c2 >= 0) & (c2 < W)
        c3 = w_in_c + 3
        m_c3 = (c3 >= 0) & (c3 < W)

        # row 0
        x00 = tl.load(x_ptr + x_base + h0 * stride_xh + c0 * stride_xw, mask=m_h0 & m_c0, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + x_base + h0 * stride_xh + c1 * stride_xw, mask=m_h0 & m_c1, other=0.0).to(tl.float32)
        x02 = tl.load(x_ptr + x_base + h0 * stride_xh + c2 * stride_xw, mask=m_h0 & m_c2, other=0.0).to(tl.float32)
        x03 = tl.load(x_ptr + x_base + h0 * stride_xh + c3 * stride_xw, mask=m_h0 & m_c3, other=0.0).to(tl.float32)
        # row 1
        x10 = tl.load(x_ptr + x_base + h1 * stride_xh + c0 * stride_xw, mask=m_h1 & m_c0, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + x_base + h1 * stride_xh + c1 * stride_xw, mask=m_h1 & m_c1, other=0.0).to(tl.float32)
        x12 = tl.load(x_ptr + x_base + h1 * stride_xh + c2 * stride_xw, mask=m_h1 & m_c2, other=0.0).to(tl.float32)
        x13 = tl.load(x_ptr + x_base + h1 * stride_xh + c3 * stride_xw, mask=m_h1 & m_c3, other=0.0).to(tl.float32)
        # row 2
        x20 = tl.load(x_ptr + x_base + h2 * stride_xh + c0 * stride_xw, mask=m_h2 & m_c0, other=0.0).to(tl.float32)
        x21 = tl.load(x_ptr + x_base + h2 * stride_xh + c1 * stride_xw, mask=m_h2 & m_c1, other=0.0).to(tl.float32)
        x22 = tl.load(x_ptr + x_base + h2 * stride_xh + c2 * stride_xw, mask=m_h2 & m_c2, other=0.0).to(tl.float32)
        x23 = tl.load(x_ptr + x_base + h2 * stride_xh + c3 * stride_xw, mask=m_h2 & m_c3, other=0.0).to(tl.float32)
        # row 3
        x30 = tl.load(x_ptr + x_base + h3 * stride_xh + c0 * stride_xw, mask=m_h3 & m_c0, other=0.0).to(tl.float32)
        x31 = tl.load(x_ptr + x_base + h3 * stride_xh + c1 * stride_xw, mask=m_h3 & m_c1, other=0.0).to(tl.float32)
        x32 = tl.load(x_ptr + x_base + h3 * stride_xh + c2 * stride_xw, mask=m_h3 & m_c2, other=0.0).to(tl.float32)
        x33 = tl.load(x_ptr + x_base + h3 * stride_xh + c3 * stride_xw, mask=m_h3 & m_c3, other=0.0).to(tl.float32)

        # conv outputs at the 4 positions (2x2 pooling region)
        # output (0,0): uses x[0:3, 0:3] with weights w
        acc00 += (w00*x00 + w01*x01 + w02*x02 +
                  w10*x10 + w11*x11 + w12*x12 +
                  w20*x20 + w21*x21 + w22*x22)
        # output (0,1): uses x[0:3, 1:4]
        acc01 += (w00*x01 + w01*x02 + w02*x03 +
                  w10*x11 + w11*x12 + w12*x13 +
                  w20*x21 + w21*x22 + w22*x23)
        # output (1,0): uses x[1:4, 0:3]
        acc10 += (w00*x10 + w01*x11 + w02*x12 +
                  w10*x20 + w11*x21 + w12*x22 +
                  w20*x30 + w21*x31 + w22*x32)
        # output (1,1): uses x[1:4, 1:4]
        acc11 += (w00*x11 + w01*x12 + w02*x13 +
                  w10*x21 + w11*x22 + w12*x23 +
                  w20*x31 + w21*x32 + w22*x33)

    bias = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc00 += bias
    acc01 += bias
    acc10 += bias
    acc11 += bias

    acc00 = tl.maximum(acc00, 0.0)
    acc01 = tl.maximum(acc01, 0.0)
    acc10 = tl.maximum(acc10, 0.0)
    acc11 = tl.maximum(acc11, 0.0)

    pooled = (acc00 + acc01 + acc10 + acc11) * 0.25

    mask_h = offs_h < OH
    mask_w = offs_w < OW
    mask = mask_h[:, None] & mask_w[None, :]
    y_ptrs = (y_ptr + pid_oc * stride_yc
              + offs_h[:, None] * stride_yh
              + offs_w[None, :] * stride_yw)
    tl.store(y_ptrs, pooled.to(tl.float16), mask=mask)


def _fused_conv_relu_pool(x, weight, bias):
    assert x.is_contiguous() and weight.is_contiguous()
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    assert N == 1 and KH == 3 and KW == 3

    OH = H // 2
    OW = W // 2
    y = torch.empty((N, C_out, OH, OW), dtype=torch.float16, device=x.device)

    BLOCK_H = 8
    BLOCK_W = 16

    grid = (C_out, triton.cdiv(OH, BLOCK_H) * triton.cdiv(OW, BLOCK_W))

    fused_conv_relu_pool_kernel[grid](
        x, weight, bias, y,
        H, W, OH, OW,
        x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(1), y.stride(2), y.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        C_IN=C_in,
    )
    return y


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        activation_type=ActivationType.RELU if _HAS_DEPS else None,
        signed_quant=True,
        quantize_conv=False,
        quantize_act=False,
        activation_quantizer=None,
        integer_forward=False,
        num_bits_conv=8,
        num_bits_act=8,
        padding_mode="zeros",
        round_fn=QRoundFn.TRUNC if _HAS_DEPS else None,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_type = activation_type
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward

        if _HAS_DEPS:
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
        else:
            self.blocks = {
                "activation": torch.nn.ReLU(),
                "convolution": torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size,
                    stride=stride, padding=kernel_size // 2, bias=bias,
                    padding_mode=padding_mode,
                ),
                "activation_quantizer": None,
            }

        self.activation = self.blocks["activation"]
        self.act_quantizer = self.blocks["activation_quantizer"]
        self.conv2d = self.blocks["convolution"]
        self.pool = torch.nn.AvgPool2d(kernel_size=2)

        self._can_fuse = (
            kernel_size == 3
            and stride == 1
            and bias
            and not quantize_conv
            and not integer_forward
            and isinstance(self.activation, torch.nn.ReLU)
            and padding_mode == "zeros"
        )

    def _fallback(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.act_quantizer is not None:
            x = self.act_quantizer(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        if self.integer_forward:
            return self._fallback(x)

        if not self._can_fuse:
            return self._fallback(x)

        if not x.is_contiguous():
            x = x.contiguous()

        weight = self.conv2d.weight
        bias = self.conv2d.bias
        if weight.dtype != torch.float16:
            weight = weight.half()
        if bias.dtype != torch.float16:
            bias = bias.half()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        return _fused_conv_relu_pool(x, weight, bias)