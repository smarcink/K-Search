"""Small FP16 linear projection task for HLSL linalg smoke generation."""

import torch
from torch import nn


BATCH = 32768
IN_FEATURES = 2048
OUT_FEATURES = 256


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def get_inputs():
    device = get_device()
    x = torch.randn(BATCH, IN_FEATURES, dtype=torch.float16, device=device)
    return [x]


def get_init_inputs():
    return []


if __name__ == "__main__":
    device = get_device()
    model = Model().half().to(device)
    inputs = get_inputs()
    output = model(*inputs)
    print(f"Input shape: {tuple(inputs[0].shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    assert output.shape == (BATCH, OUT_FEATURES)
    assert output.dtype == torch.float16
    print("OK")
