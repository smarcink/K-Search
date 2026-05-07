from enum import Enum
from functools import partial

import torch

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


class ModuleRef:
    """Module reference class.

    Use to avoid registering a module member as a submodule.
    """

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

    def forward(self, x):
        x = self.blocks["convolution"](x)

        if self.blocks["activation"]:
            x = self.blocks["activation"](x)

        if self.blocks["activation_quantizer"]:
            x = self.blocks["activation_quantizer"](x)
        x = self.pool(x)
        return x


# KernelBench expects a class named "Model"
Model = ConvBlock


def get_inputs():
    # Half-precision inputs to match the half-precision Model.
    return [torch.randn(1, 64, 360, 640, dtype=torch.float16)]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    quantize=False
    integer_forward=False
    return [
        in_channels,              # in_channels
        out_channels,             # out_channels
        3,                        # kernel_size
        1,                        # stride (default)
        True,                     # bias (default)
        ActivationType.RELU,      # activation_type
        True,                     # signed_quant (default)
        quantize,                 # quantize_conv
        quantize,                 # quantize_act
        None,                     # activation_quantizer
        integer_forward,          # integer_forward
        8,                        # num_bits_conv
        8,                        # num_bits_act
        "zeros",                  # padding_mode
    ]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    print(f"Init inputs: {init_inputs}")

    model = ConvBlock(*init_inputs).half().cuda()
    print(f"Model created: {model}")

    inputs = get_inputs()
    print(f"Input shape: {inputs[0].shape}, dtype: {inputs[0].dtype}")

    inputs_cuda = [x.cuda() for x in inputs]
    output = model(*inputs_cuda)
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
