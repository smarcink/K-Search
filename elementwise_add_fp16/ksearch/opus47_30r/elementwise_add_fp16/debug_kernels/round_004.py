import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_kernel_nomask(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        y_c = y.contiguous()
        out = torch.empty_like(x_c)
        n_elements = out.numel()
        # Fast path: large tile, no masking when divisible
        BLOCK = 8192
        if n_elements % BLOCK == 0:
            grid = (n_elements // BLOCK,)
            add_kernel_nomask[grid](x_c, y_c, out, BLOCK_SIZE=BLOCK, num_warps=8)
        else:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel[grid](x_c, y_c, out, n_elements)
        return out