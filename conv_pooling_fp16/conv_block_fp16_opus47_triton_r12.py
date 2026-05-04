import torch
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


class ShortcutType(Enum):
    NONE = "none"
    CONCAT = "concat"
    ADD = "add"
    MAX = "max"


def _init_conv_weights(
    conv2d: torch.nn.Conv2d,
    activation: torch.nn.Module,
    init_method: InitMethod,
    init_icnr: bool = False,
):
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
    N, H, W,
    OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_hw // num_w_blocks
    pid_ow = pid_hw % num_w_blocks

    oh_start = pid_oh * BLOCK_OH
    ow_start = pid_ow * BLOCK_OW

    PRE_BH: tl.constexpr = BLOCK_OH * 2
    PRE_BW: tl.constexpr = BLOCK_OW * 2

    conv_h_start = oh_start * 2
    conv_w_start = ow_start * 2

    ph = tl.arange(0, PRE_BH)
    pw = tl.arange(0, PRE_BW)

    acc = tl.zeros((C_OUT, PRE_BH, PRE_BW), dtype=tl.float32)

    for ic in tl.static_range(C_IN):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = conv_h_start + ph + kh - 1
                in_w = conv_w_start + pw + kw - 1
                h_mask = (in_h >= 0) & (in_h < H)
                w_mask = (in_w >= 0) & (in_w < W)
                offs = (pid_n * stride_xn
                        + ic * stride_xc
                        + in_h[:, None] * stride_xh
                        + in_w[None, :] * stride_xw)
                mask = h_mask[:, None] & w_mask[None, :]
                xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

                w_offs = tl.arange(0, C_OUT) * (C_IN * 9) + ic * 9 + kh * 3 + kw
                wv = tl.load(w_ptr + w_offs).to(tl.float32)

                acc += wv[:, None, None] * xv[None, :, :]

    bias = tl.load(b_ptr + tl.arange(0, C_OUT)).to(tl.float32)
    acc = acc + bias[:, None, None]
    acc = tl.maximum(acc, 0.0)

    acc2 = tl.reshape(acc, (C_OUT, BLOCK_OH, 2, BLOCK_OW, 2))
    pooled = tl.sum(acc2, axis=4)
    pooled = tl.sum(pooled, axis=2) * 0.25

    oh_idx = oh_start + tl.arange(0, BLOCK_OH)
    ow_idx = ow_start + tl.arange(0, BLOCK_OW)
    oh_mask = oh_idx < OH
    ow_mask = ow_idx < OW

    oc_idx = tl.arange(0, C_OUT)
    y_offs = (pid_n * stride_yn
              + oc_idx[:, None, None] * stride_yc
              + oh_idx[None, :, None] * stride_yh
              + ow_idx[None, None, :] * stride_yw)
    y_mask = oh_mask[None, :, None] & ow_mask[None, None, :]
    tl.store(y_ptr + y_offs, pooled.to(tl.float16), mask=y_mask)


@triton.jit
def fused_conv_relu_pool_kernel_oc(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H, W,
    OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_oc = tl.program_id(2)

    num_w_blocks = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_hw // num_w_blocks
    pid_ow = pid_hw % num_w_blocks

    oh_start = pid_oh * BLOCK_OH
    ow_start = pid_ow * BLOCK_OW
    oc_start = pid_oc * BLOCK_OC

    PRE_BH: tl.constexpr = BLOCK_OH * 2
    PRE_BW: tl.constexpr = BLOCK_OW * 2

    conv_h_start = oh_start * 2
    conv_w_start = ow_start * 2

    ph = tl.arange(0, PRE_BH)
    pw = tl.arange(0, PRE_BW)

    acc = tl.zeros((BLOCK_OC, PRE_BH, PRE_BW), dtype=tl.float32)

    oc_range = oc_start + tl.arange(0, BLOCK_OC)

    for ic in tl.static_range(C_IN):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = conv_h_start + ph + kh - 1
                in_w = conv_w_start + pw + kw - 1
                h_mask = (in_h >= 0) & (in_h < H)
                w_mask = (in_w >= 0) & (in_w < W)
                offs = (pid_n * stride_xn
                        + ic * stride_xc
                        + in_h[:, None] * stride_xh
                        + in_w[None, :] * stride_xw)
                mask = h_mask[:, None] & w_mask[None, :]
                xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

                w_offs = oc_range * (C_IN * 9) + ic * 9 + kh * 3 + kw
                wv = tl.load(w_ptr + w_offs).to(tl.float32)

                acc += wv[:, None, None] * xv[None, :, :]

    bias = tl.load(b_ptr + oc_range).to(tl.float32)
    acc = acc + bias[:, None, None]
    acc = tl.maximum(acc, 0.0)

    acc2 = tl.reshape(acc, (BLOCK_OC, BLOCK_OH, 2, BLOCK_OW, 2))
    pooled = tl.sum(acc2, axis=4)
    pooled = tl.sum(pooled, axis=2) * 0.25

    oh_idx = oh_start + tl.arange(0, BLOCK_OH)
    ow_idx = ow_start + tl.arange(0, BLOCK_OW)
    oh_mask = oh_idx < OH
    ow_mask = ow_idx < OW

    y_offs = (pid_n * stride_yn
              + oc_range[:, None, None] * stride_yc
              + oh_idx[None, :, None] * stride_yh
              + ow_idx[None, None, :] * stride_yw)
    y_mask = oh_mask[None, :, None] & ow_mask[None, None, :]
    tl.store(y_ptr + y_offs, pooled.to(tl.float16), mask=y_mask)


_kernel_cache = {}


def fused_conv_relu_pool(x, weight, bias):
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    OH = H // 2
    OW = W // 2
    y = torch.empty((N, C_OUT, OH, OW), dtype=torch.float16, device=x.device)

    key = (N, C_IN, C_OUT, H, W)
    if key not in _kernel_cache:
        # Configs: (kernel_variant, BLOCK_OC, BLOCK_OH, BLOCK_OW, num_warps)
        # variant 0 = full OC kernel; variant 1 = OC-tiled kernel
        configs = [
            (0, C_OUT, 8, 16, 4),
            (0, C_OUT, 8, 32, 4),
            (0, C_OUT, 4, 32, 4),
            (0, C_OUT, 16, 16, 8),
            (0, C_OUT, 8, 16, 8),
            (0, C_OUT, 4, 64, 8),
        ]
        if C_OUT % 16 == 0 and C_OUT > 16:
            configs += [
                (1, 16, 8, 32, 4),
                (1, 16, 16, 32, 8),
                (1, 16, 8, 64, 8),
                (1, 16, 4, 64, 4),
                (1, 16, 16, 16, 4),
            ]
        if C_OUT % 8 == 0 and C_OUT > 8:
            configs += [
                (1, 8, 16, 32, 4),
                (1, 8, 8, 64, 4),
            ]

        best = None
        best_t = float("inf")
        for cfg in configs:
            variant, BOC, BOH, BOW, nw = cfg
            if not (OH >= BOH and OW >= BOW):
                continue
            if variant == 1 and C_OUT % BOC != 0:
                continue
            try:
                if variant == 0:
                    grid = (N, triton.cdiv(OH, BOH) * triton.cdiv(OW, BOW))
                    args = lambda: fused_conv_relu_pool_kernel[grid](
                        x, weight, bias, y,
                        N, H, W, OH, OW,
                        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                        C_IN=C_IN, C_OUT=C_OUT,
                        BLOCK_OH=BOH, BLOCK_OW=BOW,
                        num_warps=nw,
                    )
                else:
                    grid = (N, triton.cdiv(OH, BOH) * triton.cdiv(OW, BOW), C_OUT // BOC)
                    args = lambda: fused_conv_relu_pool_kernel_oc[grid](
                        x, weight, bias, y,
                        N, H, W, OH, OW,
                        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                        C_IN=C_IN, C_OUT=C_OUT,
                        BLOCK_OC=BOC, BLOCK_OH=BOH, BLOCK_OW=BOW,
                        num_warps=nw,
                    )
                for _ in range(2):
                    args()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    args()
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end)
                if t < best_t:
                    best_t = t
                    best = cfg
            except Exception:
                continue
        if best is None:
            best = (0, C_OUT, 8, 16, 4)
        _kernel_cache[key] = best

    cfg = _kernel_cache[key]
    variant, BOC, BOH, BOW, nw = cfg
    if variant == 0:
        grid = (N, triton.cdiv(OH, BOH) * triton.cdiv(OW, BOW))
        fused_conv_relu_pool_kernel[grid](
            x, weight, bias, y,
            N, H, W, OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            C_IN=C_IN, C_OUT=C_OUT,
            BLOCK_OH=BOH, BLOCK_OW=BOW,
            num_warps=nw,
        )
    else:
        grid = (N, triton.cdiv(OH, BOH) * triton.cdiv(OW, BOW), C_OUT // BOC)
        fused_conv_relu_pool_kernel_oc[grid](
            x, weight, bias, y,
            N, H, W, OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            C_IN=C_IN, C_OUT=C_OUT,
            BLOCK_OC=BOC, BLOCK_OH=BOH, BLOCK_OW=BOW,
            num_warps=nw,
        )
    return y


class ModuleRef:
    def __init__(self, module):
        self.__module = module

    def __getattr__(self, item):
        return getattr(self.__module, item)


class ConvBlock(torch.nn.Module):
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

        self._cached_w = None
        self._cached_b = None

    def get_out_channels(self) -> int:
        return self.out_channels

    def get_act_quantizer(self):
        return self.blocks["activation_quantizer"]

    def set_act_quantizer(self, quantizer):
        self.blocks["activation_quantizer"] = quantizer

    def compute_params(self, input_step_size):
        if self.blocks["activation_quantizer"]:
            act_step = self.blocks["activation_quantizer"].get_quant_step()
        else:
            act_step = 1
        mac_bias = self.blocks["convolution"].bias / act_step
        wt_step = self.blocks["convolution"].get_quant_step().flatten()
        mac_scale = input_step_size * wt_step / act_step
        return act_step, mac_bias, mac_scale

    def get_params(self, input_params):
        params = {}
        params["weights"] = self.blocks["convolution"].get_quant_weights()
        params["ichannels"] = self.blocks["convolution"].in_channels
        params["ochannels"] = self.blocks["convolution"].out_channels
        params["padding_mode"] = self.blocks["convolution"].padding_mode
        params["kernel_size"] = self.blocks["convolution"].kernel_size
        params["stride"] = self.blocks["convolution"].stride
        params["act_type"] = str(self.activation_type)
        if self.signed_quant:
            params["out_quant_type"] = "signed"
        else:
            params["out_quant_type"] = "unsigned"
        if self.quantize_conv:
            (act_step, mac_bias, mac_scale) = self.compute_params(input_params["act_step"])
            params["act_step"] = act_step
            params["mac_bias"] = mac_bias
            params["mac_scale"] = mac_scale
        else:
            params["bias"] = self.blocks["convolution"].bias
        return params

    def _integer_forward_impl(self, x):
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

    def _can_use_fused(self, x):
        if self.integer_forward or self.quantize_conv:
            return False
        if self.kernel_size != 3 or self.stride != 1:
            return False
        if not isinstance(self.activation, torch.nn.ReLU):
            return False
        if self.act_quantizer is not None:
            try:
                test = torch.zeros(1, device=x.device, dtype=x.dtype)
                out = self.act_quantizer(test)
                if not torch.equal(out, test):
                    return False
            except Exception:
                return False
        if x.dtype != torch.float16:
            return False
        if x.dim() != 4:
            return False
        N, C, H, Wd = x.shape
        if H % 2 != 0 or Wd % 2 != 0:
            return False
        if not x.is_contiguous():
            return False
        return True

    def _get_cached_wb(self):
        w = self.conv2d.weight
        b = self.conv2d.bias
        if self._cached_w is None or self._cached_w.data_ptr() != w.data_ptr() or self._cached_w.dtype != torch.float16:
            wh = w.half().contiguous() if w.dtype != torch.float16 else w.contiguous()
            self._cached_w = wh
        if b is not None and (self._cached_b is None or self._cached_b.data_ptr() != b.data_ptr() or self._cached_b.dtype != torch.float16):
            bh = b.half().contiguous() if b.dtype != torch.float16 else b.contiguous()
            self._cached_b = bh
        return self._cached_w, self._cached_b

    def _forward_impl(self, x):
        if self._can_use_fused(x):
            w, b = self._get_cached_wb()
            return fused_conv_relu_pool(x, w, b)

        x = self.blocks["convolution"](x)
        if self.blocks["activation"]:
            x = self.blocks["activation"](x)
        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x

    def forward(self, x):
        if self.integer_forward:
            return self._integer_forward_impl(x)
        return self._forward_impl(x)


Model = ConvBlock
ModelNew = ConvBlock


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