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
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_vec'],
)
@triton.jit
def add_kernel_fp16_vec8(x_ptr, y_ptr, out_ptr, n_vec, BLOCK_SIZE: tl.constexpr):
    # Each program processes BLOCK_SIZE vectors; each vector is 8 fp16 (128 bits).
    # We treat memory as int64 with 4 fp16 per int64, so 8 fp16 = 2 int64 loaded per thread (ILP).
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_vec
    # Two int64 per "vec" (vec = 8 fp16)
    base = offs * 2
    xa = tl.load(x_ptr + base, mask=mask)
    xb = tl.load(x_ptr + base + 1, mask=mask)
    ya = tl.load(y_ptr + base, mask=mask)
    yb = tl.load(y_ptr + base + 1, mask=mask)

    def add_i64(xv, yv):
        x_lo32 = (xv & 0xFFFFFFFF).to(tl.uint32)
        x_hi32 = ((xv >> 32) & 0xFFFFFFFF).to(tl.uint32)
        y_lo32 = (yv & 0xFFFFFFFF).to(tl.uint32)
        y_hi32 = ((yv >> 32) & 0xFFFFFFFF).to(tl.uint32)
        x0 = (x_lo32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        x1 = ((x_lo32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        x2 = (x_hi32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        x3 = ((x_hi32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        y0 = (y_lo32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        y1 = ((y_lo32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        y2 = (y_hi32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        y3 = ((y_hi32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        s0u = (x0 + y0).to(tl.uint16, bitcast=True).to(tl.uint32)
        s1u = (x1 + y1).to(tl.uint16, bitcast=True).to(tl.uint32)
        s2u = (x2 + y2).to(tl.uint16, bitcast=True).to(tl.uint32)
        s3u = (x3 + y3).to(tl.uint16, bitcast=True).to(tl.uint32)
        lo32 = s0u | (s1u << 16)
        hi32 = s2u | (s3u << 16)
        return (lo32.to(tl.uint64) | (hi32.to(tl.uint64) << 32)).to(tl.int64, bitcast=True)

    out_a = add_i64(xa, ya)
    out_b = add_i64(xb, yb)
    tl.store(out_ptr + base, out_a, mask=mask)
    tl.store(out_ptr + base + 1, out_b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        if (n % 8 == 0) and (x.data_ptr() % 16 == 0) and (y.data_ptr() % 16 == 0) and (out.data_ptr() % 16 == 0):
            x64 = x.view(torch.int64)
            y64 = y.view(torch.int64)
            out64 = out.view(torch.int64)
            n_vec = n // 8  # number of 128-bit vectors
            grid = lambda meta: (triton.cdiv(n_vec, meta['BLOCK_SIZE']),)
            add_kernel_fp16_vec8[grid](x64, y64, out64, n_vec)
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out