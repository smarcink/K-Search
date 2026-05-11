import torch
import triton
import triton.language as tl

def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out


# KernelBench convention
ElementwiseAdd = ModelNew


def get_inputs():
    """Return input tensors for benchmarking (fp16, on GPU)."""
    device = get_device()
    x = torch.randn(16, 256, 256, dtype=torch.float16, device=device)
    y = torch.randn(16, 256, 256, dtype=torch.float16, device=device)
    return [x, y]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []

if __name__ == "__main__":
    device = get_device()
    model = ElementwiseAdd().half().to(device)
    inputs = get_inputs()
    out = model(*inputs)
    print(f"Input shapes: {inputs[0].shape}, {inputs[1].shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    # Verify correctness
    expected = inputs[0] + inputs[1]
    assert torch.allclose(out, expected), "Mismatch!"
    print("OK")
