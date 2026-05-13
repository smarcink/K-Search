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
    pid_hw = tl.program_id(0)
    n_tiles_w = (Wout + BLOCK_W - 1) // BLOCK_W
    pid_h = pid_hw // n_tiles_w
    pid_w = pid_hw % n_tiles_w

    oh_off = pid_h * BLOCK_H
    ow_off = pid_w * BLOCK_W

    # conv output coords for this tile: 2*BLOCK_H rows, 2*BLOCK_W cols
    CONV_H: tl.constexpr = BLOCK_H * 2
    CONV_W: tl.constexpr = BLOCK_W * 2

    cy_base = oh_off * 2
    cx_base = ow_off * 2

    cout_idx = tl.arange(0, COUT)
    cout_mask = cout_idx < Cout
    bias = tl.load(b_ptr + cout_idx, mask=cout_mask, other=0.0).to(tl.float32)

    cin_idx = tl.arange(0, CIN)
    cin_mask = cin_idx < Cin

    # Accumulator for all conv outputs in this tile: [CONV_H, CONV_W, COUT]
    conv_acc = tl.zeros((CONV_H * CONV_W, COUT), dtype=tl.float32)

    # Load all required input data once: we need conv coords cy in [cy_base, cy_base+CONV_H),
    # cx in [cx_base, cx_base+CONV_W). For 3x3 with padding 1, input coords:
    # iy = cy + ky - 1, ix = cx + kx - 1
    # Range of iy: [cy_base-1, cy_base+CONV_H), width CONV_H+2
    # Range of ix: [cx_base-1, cx_base+CONV_W), width CONV_W+2
    IN_H: tl.constexpr = CONV_H + 2
    IN_W: tl.constexpr = CONV_W + 2

    # IN_H, IN_W must be powers of 2 for tl.arange. CONV_H=16 -> IN_H=18 (not pow2).
    # Round up.
    IN_H_P2: tl.constexpr = 32 if BLOCK_H >= 8 else 16
    IN_W_P2: tl.constexpr = 64 if BLOCK_W >= 16 else 32

    iy_idx = cy_base - 1 + tl.arange(0, IN_H_P2)  # [IN_H_P2]
    ix_idx = cx_base - 1 + tl.arange(0, IN_W_P2)  # [IN_W_P2]
    iy_valid = (iy_idx >= 0) & (iy_idx < H) & (tl.arange(0, IN_H_P2) < IN_H)
    ix_valid = (ix_idx >= 0) & (ix_idx < W) & (tl.arange(0, IN_W_P2) < IN_W)

    # Load input patch [CIN, IN_H_P2, IN_W_P2]
    # offsets: cin*H*W + iy*W + ix
    x_offsets = (
        cin_idx[:, None, None] * (H * W)
        + iy_idx[None, :, None] * W
        + ix_idx[None, None, :]
    )
    x_mask = (
        cin_mask[:, None, None]
        & iy_valid[None, :, None]
        & ix_valid[None, None, :]
    )
    x_patch = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
    # x_patch: [CIN, IN_H_P2, IN_W_P2]

    # For each (ky, kx), extract slice and contract with weights
    # Build conv output row/col indices within patch:
    # for conv output position (h, w), input position for (ky, kx) is at (h+ky, w+kx) in patch coords
    # (since patch starts at cy_base-1 and we need cy+ky-1 = cy_base-1 + (cy-cy_base) + ky = h+ky)
    h_local = tl.arange(0, CONV_H)  # [CONV_H]
    w_local = tl.arange(0, CONV_W)  # [CONV_W]

    for ky in tl.static_range(0, 3):
        for kx in tl.static_range(0, 3):
            # Extract slice: x_patch[:, h_local+ky, w_local+kx]
            # Build flat indices into patch
            patch_offsets = (
                cin_idx[:, None, None] * (IN_H_P2 * IN_W_P2)
                + (h_local + ky)[None, :, None] * IN_W_P2
                + (w_local + kx)[None, None, :]
            )
            # Reshape patch to flat to index, but we can directly gather.
            # Simpler: re-load from global with the proper indices to avoid in-kernel gather.
            cy_all = cy_base + h_local  # [CONV_H]
            cx_all = cx_base + w_local  # [CONV_W]
            iy_all = cy_all + ky - 1
            ix_all = cx_all + kx - 1
            iy_m = (iy_all >= 0) & (iy_all < H)
            ix_m = (ix_all >= 0) & (ix_all < W)

            x_off2 = (
                cin_idx[None, None, :] * (H * W)
                + iy_all[:, None, None] * W
                + ix_all[None, :, None]
            )
            mask2 = iy_m[:, None, None] & ix_m[None, :, None] & cin_mask[None, None, :]
            x_v = tl.load(x_ptr + x_off2, mask=mask2, other=0.0).to(tl.float32)
            # x_v: [CONV_H, CONV_W, CIN]

            w_offsets = (
                cout_idx[None, :] * (Cin * 9)
                + cin_idx[:, None] * 9
                + ky * 3
                + kx
            )
            w_mask = cin_mask[:, None] & cout_mask[None, :]
            w_v = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
            # w_v: [CIN, COUT]

            x_2d = tl.reshape(x_v, (CONV_H * CONV_W, CIN))
            out_2d = tl.dot(x_2d, w_v, out_dtype=tl.float32)
            conv_acc += out_2d

    # Add bias, ReLU
    conv_acc = conv_acc + bias[None, :]
    conv_acc = tl.maximum(conv_acc, 0.0)

    # Reshape to [CONV_H, CONV_W, COUT] and average-pool 2x2
    conv_3d = tl.reshape(conv_acc, (CONV_H, CONV_W, COUT))

    # Pool: take (2h, 2w), (2h, 2w+1), (2h+1, 2w), (2h+1, 2w+1) and average
    # Use strided arange
    h2 = tl.arange(0, BLOCK_H) * 2
    w2 = tl.arange(0, BLOCK_W) * 2

    # Gather 4 sub-positions
    # Flatten conv_3d for indexing: idx = h*CONV_W*COUT + w*COUT + c
    # We use reshape tricks
    # conv_3d shape: (CONV_H, CONV_W, COUT)
    # We need: pool[h, w, c] = 0.25 * (conv_3d[2h, 2w, c] + conv_3d[2h, 2w+1, c] + conv_3d[2h+1, 2w, c] + conv_3d[2h+1, 2w+1, c])

    # Reshape conv_3d to (BLOCK_H, 2, BLOCK_W, 2, COUT) and sum over the two size-2 dims
    pooled = tl.reshape(conv_3d, (BLOCK_H, 2, BLOCK_W, 2, COUT))
    pooled = tl.sum(pooled, axis=3)  # (BLOCK_H, 2, BLOCK_W, COUT)
    pooled = tl.sum(pooled, axis=1)  # (BLOCK_H, BLOCK_W, COUT)
    pooled = pooled * 0.25

    # Store
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
        weight = self.conv2d.weight.contiguous()
        bias = self.conv2d.bias.contiguous()

        y = torch.empty((N, Cout, Hout, Wout), dtype=torch.float16, device=x.device)

        BLOCK_H = 8
        BLOCK_W = 16
        COUT = 32
        CIN = 16

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