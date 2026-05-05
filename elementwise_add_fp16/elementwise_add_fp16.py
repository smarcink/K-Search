"""Simple elementwise add of two tensors.

Reference implementation for K-Search CUDA kernel optimization.
Defines Model, get_inputs(), and get_init_inputs() following KernelBench conventions.
"""

import torch


class ElementwiseAdd(torch.nn.Module):
    """Simple elementwise addition of two tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


# KernelBench convention
Model = ElementwiseAdd


def get_inputs():
    """Return input tensors for benchmarking (fp16, on GPU)."""
    x = torch.randn(16, 256, 256, dtype=torch.float16, device="cuda")
    y = torch.randn(16, 256, 256, dtype=torch.float16, device="cuda")
    return [x, y]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []


if __name__ == "__main__":
    model = ElementwiseAdd().half().cuda()
    inputs = get_inputs()
    out = model(*inputs)
    print(f"Input shapes: {inputs[0].shape}, {inputs[1].shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    # Verify correctness
    expected = inputs[0] + inputs[1]
    assert torch.allclose(out, expected), "Mismatch!"
    print("OK")
