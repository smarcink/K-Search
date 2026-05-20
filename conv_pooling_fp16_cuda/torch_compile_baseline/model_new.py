"""Standalone torch.compile baseline for conv -> relu -> avgpool."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """Benchmark-only baseline that always runs through torch.compile."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self._compiled_forward = None

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is None:
            self._compiled_forward = torch.compile(self._forward_impl, mode="default")
        return self._compiled_forward(x)