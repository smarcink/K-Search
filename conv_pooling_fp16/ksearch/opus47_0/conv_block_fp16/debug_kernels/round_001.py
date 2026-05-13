import torch
import torch.nn as nn
import triton
import triton.language as tl

from activations import ActivationType, create_activation
from quantization import QConv2D, QRoundFn, QTrunc, QFloor, QRound, create_quantizer


@triton.jit
def fused_conv_relu_pool_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    C_IN, KH, KW,
):
    # Each program produces a BLOCK_H x BLOCK_W tile of pooled outputs for one (n, oc).
    pid_oc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_tiles = tl.cdiv(OW, BLOCK_W)
    pid_h = pid_hw // num_w_tiles
    pid_w = pid_hw % num_w_tiles

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    offs_h = oh_start + tl.arange(0, BLOCK_H)  # pooled row indices
    offs_w = ow_start + tl.arange(0, BLOCK_W)  # pooled col indices

    # Each pooled pixel corresponds to 2x2 conv outputs starting at (2*oh, 2*ow).
    # 3x3 conv with pad=1 means input region per conv pixel: [conv_h-1..conv_h+1].
    # For 2x2 conv block at (ch, cw) = (2*oh+{0,1}, 2*ow+{0,1}), we need input rows
    # ch-1..ch+2 = [2*oh-1 .. 2*oh+3] (5 rows), cols similar (5 cols).
    conv_h0 = offs_h * 2  # [BLOCK_H]
    conv_w0 = offs_w * 2  # [BLOCK_W]

    # Accumulators for 4 conv outputs per pooled pixel
    acc00 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels
    for ic in range(0, C_IN):
        # Load weight 3x3 for this (oc, ic)
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

        # Load 5x5 input patch per pooled pixel into registers.
        # Input rows: conv_h0 - 1 + r for r in 0..4 ; cols: conv_w0 - 1 + c for c in 0..4
        # We'll load each of 5x5 input positions as (BLOCK_H, BLOCK_W) tiles.
        x_base = ic * stride_xc

        # Precompute row pointers offsets
        # For each r in 0..4, h_in = conv_h0 - 1 + r ; mask valid if 0<=h_in<H
        # For each c in 0..4, w_in = conv_w0 - 1 + c ; mask valid if 0<=w_in<W
        def load_patch(r, c):
            h_in = conv_h0[:, None] - 1 + r
            w_in = conv_w0[None, :] - 1 + c
            mask = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            ptrs = x_ptr + x_base + h_in * stride_xh + w_in * stride_xw
            return tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        x00 = load_patch(0, 0); x01 = load_patch(0, 1); x02 = load_patch(0, 2); x03 = load_patch(0, 3); x04 = load_patch(0, 4)
        x10 = load_patch(1, 0); x11 = load_patch(1, 1); x12 = load_patch(1, 2); x13 = load_patch(1, 3); x14 = load_patch(1, 4)
        x20 = load_patch(2, 0); x21 = load_patch(2, 1); x22 = load_patch(2, 2); x23 = load_patch(2, 3); x24 = load_patch(2, 4)
        x30 = load_patch(3, 0); x31 = load_patch(3, 1); x32 = load_patch(3, 2); x33 = load_patch(3, 3); x34 = load_patch(3, 4)
        x40 = load_patch(4, 0); x41 = load_patch(4, 1); x42 = load_patch(4, 2); x43 = load_patch(4, 3); x44 = load_patch(4, 4)

        # Conv output at (2*oh, 2*ow): input rows 2*oh-1..2*oh+1 -> r=0..2, cols 0..2
        acc00 += (w00*x00 + w01*x01 + w02*x02 +
                  w10*x10 + w11*x11 + w12*x12 +
                  w20*x20 + w21*x21 + w22*x22)
        # Conv output at (2*oh, 2*ow+1): input cols 2*ow..2*ow+2 -> c=1..3
        acc01 += (w00*x01 + w01*x02 + w02*x03 +
                  w10*x11 + w11*x12 + w12*x13 +
                  w20*x21 + w21*x22 + w22*x23)
        # Conv output at (2*oh+1, 2*ow): input rows 2*oh..2*oh+2 -> r=1..3, cols 0..2
        acc10 += (w00*x10 + w01*x11 + w02*x12 +
                  w10*x20 + w11*x21 + w12*x22 +
                  w20*x30 + w21*x31 + w22*x32)
        # Conv output at (2*oh+1, 2*ow+1): r=1..3, c=1..3
        acc11 += (w00*x11 + w01*x12 + w02*x13 +
                  w10*x21 + w11*x22 + w12*x23 +
                  w20*x31 + w21*x32 + w22*x33)

    # Add bias
    bias = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc00 += bias
    acc01 += bias
    acc10 += bias
    acc11 += bias

    # ReLU
    acc00 = tl.maximum(acc00, 0.0)
    acc01 = tl.maximum(acc01, 0.0)
    acc10 = tl.maximum(acc10, 0.0)
    acc11 = tl.maximum(acc11, 0.0)

    # AvgPool2d 2x2
    pooled = (acc00 + acc01 + acc10 + acc11) * 0.25

    # Store
    mask_h = offs_h < OH
    mask_w = offs_w < OW
    mask = mask_h[:, None] & mask_w[None, :]
    y_ptrs = (y_ptr + pid_oc * stride_yc
              + offs_h[:, None] * stride_yh
              + offs_w[None, :] * stride_yw)
    tl.store(y_ptrs, pooled.to(tl.float16), mask=mask)


def _fused_conv_relu_pool(x, weight, bias):
    # x: [1, 16, H, W] fp16, weight: [32, 16, 3, 3] fp16, bias: [32] fp16
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
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        C_IN=C_in, KH=KH, KW=KW,
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

        # Determine if we can use the fused fast path:
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
            # delegate to original logic
            return self._fallback(x)

        if not self._can_fuse:
            return self._fallback(x)

        # act_quantizer in this config (quantize_act=False) is identity.
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