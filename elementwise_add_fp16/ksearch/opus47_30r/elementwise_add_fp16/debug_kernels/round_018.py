import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_CTAS': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_CTAS': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CTAS': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CTAS': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CTAS': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CTAS': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CTAS': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CTAS': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768, 'NUM_CTAS': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768, 'NUM_CTAS': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_grid_stride(x_ptr, y_ptr, out_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr, NUM_CTAS: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    stride = BLOCK_SIZE * num_programs
    base = pid * BLOCK_SIZE
    for start in tl.range(base, n_elements, stride):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_fp16(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        grid = lambda meta: (meta['NUM_CTAS'],)
        add_kernel_grid_stride[grid](x, y, out, n)
        return out