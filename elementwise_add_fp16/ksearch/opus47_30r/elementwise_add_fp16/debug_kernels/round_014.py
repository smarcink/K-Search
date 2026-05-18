import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_vec'],
)
@triton.jit
def add_kernel_fp16_vec4(x_ptr, y_ptr, out_ptr, n_vec, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each "vec" element is 4 fp16 (8 bytes). We process BLOCK_SIZE vec elements per program.
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    # base fp16 offset for each vec
    base = offs * 4
    o0 = base
    o1 = base + 1
    o2 = base + 2
    o3 = base + 3
    m0 = o0 < n_elements
    m1 = o1 < n_elements
    m2 = o2 < n_elements
    m3 = o3 < n_elements
    x0 = tl.load(x_ptr + o0, mask=m0)
    x1 = tl.load(x_ptr + o1, mask=m1)
    x2 = tl.load(x_ptr + o2, mask=m2)
    x3 = tl.load(x_ptr + o3, mask=m3)
    y0 = tl.load(y_ptr + o0, mask=m0)
    y1 = tl.load(y_ptr + o1, mask=m1)
    y2 = tl.load(y_ptr + o2, mask=m2)
    y3 = tl.load(y_ptr + o3, mask=m3)
    tl.store(out_ptr + o0, x0 + y0, mask=m0)
    tl.store(out_ptr + o1, x1 + y1, mask=m1)
    tl.store(out_ptr + o2, x2 + y2, mask=m2)
    tl.store(out_ptr + o3, x3 + y3, mask=m3)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        # Use vec4 path when divisible by 4 and 8-byte aligned
        if (n % 4 == 0) and (x.data_ptr() % 8 == 0) and (y.data_ptr() % 8 == 0) and (out.data_ptr() % 8 == 0):
            n_vec = n // 4
            grid = lambda meta: (triton.cdiv(n_vec, meta['BLOCK_SIZE']),)
            add_kernel_fp16_vec4[grid](x, y, out, n_vec, n)
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out