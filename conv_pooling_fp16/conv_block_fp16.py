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
            # ICNR with default init
            icnr(conv2d.weight, conv2d.bias)
        else:
            # ICNR partial with custom init
            init_fn = partial(icnr, bias=conv2d.bias, initializer=init_fn)

    if init_fn is None:
        return

    if isinstance(activation, torch.nn.LeakyReLU):
        init_fn(conv2d.weight, a=activation.negative_slope, nonlinearity="leaky_relu")
    if isinstance(activation, torch.nn.PReLU):
        # it's only work for single-element tensors
        init_fn(conv2d.weight, a=activation.weight.item(), nonlinearity="leaky_relu")
    else:
        init_fn(conv2d.weight, nonlinearity="relu")


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

        _init_conv_weights(
            self.blocks["convolution"], self.blocks["activation"], init_method, init_icnr
        )

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

        # integer convolution
        x_q = self.blocks["convolution"].conv_integer(x[0])

        (act_step, mac_bias, mac_scale) = self.compute_params(input_step_size)

        # scale and bias
        y_q = x_q * mac_scale.view(1, -1, 1, 1) + mac_bias.view(1, -1, 1, 1)
        y_q = self.pool(y_q)
        if self.blocks["activation"]:
            y_q = self.blocks["activation"](y_q)

        if self.blocks["activation_quantizer"]:
            z_q = self.blocks["activation_quantizer"].integer(y_q)
            return z_q, act_step
        else:
            return y_q

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
            return self._integer_forward_impl(x)

        return self._forward_impl(x)


# KernelBench expects a class named "Model"
Model = ConvBlock


def get_inputs():
    # Half-precision inputs to match the half-precision Model.
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]

def get_init_inputs():
    in_channels = 16
    out_channels = 32
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
