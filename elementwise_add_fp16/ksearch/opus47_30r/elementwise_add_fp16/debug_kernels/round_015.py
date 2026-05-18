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
    key=['n_pairs'],
)
@triton.jit
def add_kernel_fp16_packed(x_ptr, y_ptr, out_ptr, n_pairs, BLOCK_SIZE: tl.constexpr):
    # x_ptr/y_ptr/out_ptr are uint32* views; each uint32 contains 2 packed fp16.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_pairs
    xv = tl.load(x_ptr + offs, mask=mask)
    yv = tl.load(y_ptr + offs, mask=mask)
    # Extract low/high fp16 from each uint32
    x_lo = (xv & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x_hi = ((xv >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y_lo = (yv & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y_hi = ((yv >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    s_lo = (x_lo + y_lo)
    s_hi = (x_hi + y_hi)
    # Repack
    s_lo_u = s_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    s_hi_u = s_hi.to(tl.uint16, bitcast=True).to(tl.uint32)
    out_packed = s_lo_u | (s_hi_u << 16)
    tl.store(out_ptr + offs, out_packed, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        if (n % 2 == 0) and (x.data_ptr() % 4 == 0) and (y.data_ptr() % 4 == 0) and (out.data_ptr() % 4 == 0):
            x32 = x.view(torch.int32)
            y32 = y.view(torch.int32)
            out32 = out.view(torch.int32)
            n_pairs = n // 2
            grid = lambda meta: (triton.cdiv(n_pairs, meta['BLOCK_SIZE']),)
            add_kernel_fp16_packed[grid](x32, y32, out32, n_pairs)
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out