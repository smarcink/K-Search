import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_add_fp16_autotuned_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    """Optimized elementwise addition using autotuned Triton kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        assert x.is_contiguous() and y.is_contiguous()

        out = torch.empty_like(x)
        n_elements = x.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        elementwise_add_fp16_autotuned_kernel[grid](
            x, y, out,
            n_elements,
        )

        return out