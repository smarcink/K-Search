import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'UNROLL': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'UNROLL': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'UNROLL': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'UNROLL': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL': 8}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * UNROLL
    for i in tl.static_range(UNROLL):
        offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        y_c = y.contiguous()
        out = torch.empty_like(x_c)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['UNROLL']),)
        add_kernel[grid](x_c, y_c, out, n_elements)
        return out