import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_add_kernel(
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
    x = tl.load(x_ptr + offsets, mask=mask, cache_modifier='.cs')
    y = tl.load(y_ptr + offsets, mask=mask, cache_modifier='.cs')
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask, cache_modifier='.cs')


class ModelNew(torch.nn.Module):
    """Optimized elementwise addition using autotuned Triton kernel with software pipelining."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        elementwise_add_kernel[grid](
            x, y, out,
            n_elements,
        )
        return out