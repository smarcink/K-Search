import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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


@triton.jit
def add_kernel_vec4(x_ptr, y_ptr, out_ptr, n_vec, BLOCK_SIZE: tl.constexpr):
    # Each thread processes 4 fp16 elements via vectorized load (uint64 = 4xfp16)
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vec
    # Treat as packed uint64; but simpler: load 4 contiguous fp16 per "element index"
    base = offsets * 4
    o0 = base
    o1 = base + 1
    o2 = base + 2
    o3 = base + 3
    x0 = tl.load(x_ptr + o0, mask=mask)
    x1 = tl.load(x_ptr + o1, mask=mask)
    x2 = tl.load(x_ptr + o2, mask=mask)
    x3 = tl.load(x_ptr + o3, mask=mask)
    y0 = tl.load(y_ptr + o0, mask=mask)
    y1 = tl.load(y_ptr + o1, mask=mask)
    y2 = tl.load(y_ptr + o2, mask=mask)
    y3 = tl.load(y_ptr + o3, mask=mask)
    tl.store(out_ptr + o0, x0 + y0, mask=mask)
    tl.store(out_ptr + o1, x1 + y1, mask=mask)
    tl.store(out_ptr + o2, x2 + y2, mask=mask)
    tl.store(out_ptr + o3, x3 + y3, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        y_c = y.contiguous()
        out = torch.empty_like(x_c)
        n_elements = out.numel()
        BLOCK = 8192
        if n_elements % BLOCK == 0:
            grid = (n_elements // BLOCK,)
            add_kernel_nomask[grid](x_c, y_c, out, BLOCK_SIZE=BLOCK, num_warps=8)
        else:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel[grid](x_c, y_c, out, n_elements)
        return out