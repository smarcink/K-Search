import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def elementwise_add_fp16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    """Optimized elementwise addition using a custom Triton kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous layout
        x = x.contiguous()
        y = y.contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        # Grid size: determined by autotuner's chosen BLOCK_SIZE at runtime
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        elementwise_add_fp16_kernel[grid](x, y, out, N)
        return out


# KernelBench convention
Model = ModelNew


def get_inputs():
    """Return input tensors for benchmarking (fp16, on GPU)."""
    x = torch.randn(16, 256, 256, dtype=torch.float16, device="cuda")
    y = torch.randn(16, 256, 256, dtype=torch.float16, device="cuda")
    return [x, y]


def get_init_inputs():
    """Return constructor arguments for Model."""
    return []


if __name__ == "__main__":
    model = ModelNew().cuda()
    inputs = get_inputs()
    out = model(*inputs)
    print(f"Input shapes: {inputs[0].shape}, {inputs[1].shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    # Verify correctness
    expected = inputs[0] + inputs[1]
    assert torch.allclose(out, expected, atol=1e-3), "Mismatch!"
    print("OK")