import torch
import triton
import triton.language as tl


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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_pairs'],
)
@triton.jit
def add_kernel_fp16x2_packed(x_ptr, y_ptr, out_ptr, n_pairs, BLOCK_SIZE: tl.constexpr):
    # x_ptr, y_ptr, out_ptr are uint32 pointers; each uint32 packs 2 fp16.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_pairs
    xp = tl.load(x_ptr + offs, mask=mask)
    yp = tl.load(y_ptr + offs, mask=mask)
    # Reinterpret packed uint32 as 2x fp16 via bitcast to fp16 with doubled length
    # Triton supports reshape on 1D blocks via reinterpret using to(dtype, bitcast=True)
    # Use a 2D view: treat the BLOCK_SIZE uint32 as BLOCK_SIZE*2 fp16
    x_f16 = xp.to(tl.uint32, bitcast=True)
    y_f16 = yp.to(tl.uint32, bitcast=True)
    # Extract low/high fp16 halves
    x_lo = (x_f16 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x_hi = (x_f16 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    y_lo = (y_f16 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y_hi = (y_f16 >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    r_lo = x_lo + y_lo
    r_hi = x_hi + y_hi
    r_lo_u = r_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    r_hi_u = r_hi.to(tl.uint16, bitcast=True).to(tl.uint32)
    out_packed = r_lo_u | (r_hi_u << 16)
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
            n_pairs = n // 2
            x32 = x.view(torch.int32)
            y32 = y.view(torch.int32)
            out32 = out.view(torch.int32)
            grid = lambda meta: (triton.cdiv(n_pairs, meta['BLOCK_SIZE']),)
            add_kernel_fp16x2_packed[grid](x32, y32, out32, n_pairs)
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out