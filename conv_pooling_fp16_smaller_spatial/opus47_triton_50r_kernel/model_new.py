import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

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


class InitMethod(Enum):
    DEFAULT = "default"
    HE_UNIFORM = "he_unif"
    HE_NORMAL = "he_norm"


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 32, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 16, 'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 16, 'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 32, 'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 8, 'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 8, 'BLOCK_C': 256}, num_warps=4),
    ],
    key=['H_in', 'W_in', 'C'],
)
@triton.jit
def fused_relu_avgpool_nhwc_kernel(
    x_ptr, y_ptr,
    H_in, W_in, C,
    H_out, W_out, N,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)

    HW_out = H_out * W_out
    num_hw_blocks = tl.cdiv(HW_out, BLOCK_HW)
    pid_n = pid // num_hw_blocks
    pid_hw = pid % num_hw_blocks

    hw_start = pid_hw * BLOCK_HW
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    base_in = pid_n * H_in * W_in * C
    base_out = pid_n * H_out * W_out * C

    for i in tl.static_range(BLOCK_HW):
        hw = hw_start + i
        if hw < HW_out:
            h_out = hw // W_out
            w_out = hw % W_out
            h_in = h_out * 2
            w_in = w_out * 2

            row0 = base_in + h_in * W_in * C + w_in * C
            row1 = row0 + W_in * C
            p00 = row0 + c_offs
            p01 = row0 + C + c_offs
            p10 = row1 + c_offs
            p11 = row1 + C + c_offs

            v00 = tl.load(x_ptr + p00, mask=c_mask, other=0.0).to(tl.float32)
            v01 = tl.load(x_ptr + p01, mask=c_mask, other=0.0).to(tl.float32)
            v10 = tl.load(x_ptr + p10, mask=c_mask, other=0.0).to(tl.float32)
            v11 = tl.load(x_ptr + p11, mask=c_mask, other=0.0).to(tl.float32)

            v00 = tl.maximum(v00, 0.0)
            v01 = tl.maximum(v01, 0.0)
            v10 = tl.maximum(v10, 0.0)
            v11 = tl.maximum(v11, 0.0)

            out = (v00 + v01 + v10 + v11) * 0.25

            p_out = base_out + h_out * W_out * C + w_out * C + c_offs
            tl.store(y_ptr + p_out, out.to(tl.float16), mask=c_mask)


class ModelNew(nn.Module):
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
        self.quantize_conv = quantize_conv
        self.integer_forward = integer_forward
        self.signed_quant = signed_quant
        self.activation_type = activation_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

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

        self._is_relu = activation_type == ActivationType.RELU
        self._quantize_conv = quantize_conv
        self._quantize_act = quantize_act
        self._weights_converted = False

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _convert_weights_channels_last(self):
        if self._weights_converted:
            return
        with torch.no_grad():
            conv = self.conv2d
            if hasattr(conv, 'weight') and conv.weight is not None:
                w = conv.weight.data
                if w.dim() == 4:
                    conv.weight.data = w.contiguous(memory_format=torch.channels_last)
        self._weights_converted = True

    def forward(self, x):
        if (not self._quantize_conv) and self._is_relu and (
            self.act_quantizer is None or not self._quantize_act
        ):
            self._convert_weights_channels_last()
            conv = self.conv2d
            x_cl = x.contiguous(memory_format=torch.channels_last)
            y = F.conv2d(
                x_cl,
                conv.weight,
                conv.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=1,
                groups=1,
            )
            N, C, H, W = y.shape
            assert H % 2 == 0 and W % 2 == 0
            H_out = H // 2
            W_out = W // 2

            out = torch.empty((N, C, H_out, W_out), dtype=y.dtype, device=y.device,
                              memory_format=torch.channels_last)

            HW_out = H_out * W_out

            def grid(meta):
                return (
                    N * triton.cdiv(HW_out, meta['BLOCK_HW']),
                    triton.cdiv(C, meta['BLOCK_C']),
                )

            fused_relu_avgpool_nhwc_kernel[grid](
                y, out,
                H, W, C,
                H_out, W_out, N,
            )
            return out

        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x


Model = ModelNew