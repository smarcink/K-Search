"""Simplified Conv2d + ReLU + AvgPool2d reference for CudaKernelTask.

Matches the compute pattern from conv_pooling_fp16/conv_block_fp16.py but is
fully self-contained (no external deps).  All running in fp16.

Pipeline:  x -> Conv2d(3x3, pad=1) -> ReLU -> AvgPool2d(2x2) -> out
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]


def get_init_inputs():
    return [16, 32]  # in_channels, out_channels


if __name__ == "__main__":
    model = Model(*get_init_inputs()).half().cuda().eval()
    inputs = [x.cuda() for x in get_inputs()]
    with torch.no_grad():
        out = model(*inputs)
    print(f"Input:  {inputs[0].shape}  dtype={inputs[0].dtype}")
    print(f"Output: {out.shape}  dtype={out.dtype}")
