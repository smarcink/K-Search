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
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_vec4'],
)
@triton.jit
def add_kernel_fp16_vec4(x_ptr, y_ptr, out_ptr, n_vec4, BLOCK_SIZE: tl.constexpr):
    # Each "vec4" element is 4 packed fp16 (64 bits) viewed via int64 pointers.
    # Triton will emit 128-bit transactions when adjacent threads load adjacent int64s? Actually 64-bit. 
    # To get 128-bit loads we rely on contiguous int64 loads being widened by the compiler.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_vec4
    xv = tl.load(x_ptr + offs, mask=mask)
    yv = tl.load(y_ptr + offs, mask=mask)

    # Split each int64 into two int32 halves (each holds 2 fp16)
    x_lo32 = (xv & 0xFFFFFFFF).to(tl.uint32)
    x_hi32 = ((xv >> 32) & 0xFFFFFFFF).to(tl.uint32)
    y_lo32 = (yv & 0xFFFFFFFF).to(tl.uint32)
    y_hi32 = ((yv >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # Unpack each int32 into two fp16
    x0 = (x_lo32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x1 = ((x_lo32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x2 = (x_hi32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x3 = ((x_hi32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y0 = (y_lo32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y1 = ((y_lo32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y2 = (y_hi32 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y3 = ((y_hi32 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)

    s0 = x0 + y0
    s1 = x1 + y1
    s2 = x2 + y2
    s3 = x3 + y3

    s0u = s0.to(tl.uint16, bitcast=True).to(tl.uint32)
    s1u = s1.to(tl.uint16, bitcast=True).to(tl.uint32)
    s2u = s2.to(tl.uint16, bitcast=True).to(tl.uint32)
    s3u = s3.to(tl.uint16, bitcast=True).to(tl.uint32)

    lo32 = s0u | (s1u << 16)
    hi32 = s2u | (s3u << 16)

    out_packed = lo32.to(tl.uint64) | (hi32.to(tl.uint64) << 32)
    tl.store(out_ptr + offs, out_packed.to(tl.int64, bitcast=True), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        if (n % 4 == 0) and (x.data_ptr() % 8 == 0) and (y.data_ptr() % 8 == 0) and (out.data_ptr() % 8 == 0):
            x64 = x.view(torch.int64)
            y64 = y.view(torch.int64)
            out64 = out.view(torch.int64)
            n_vec4 = n // 4
            grid = lambda meta: (triton.cdiv(n_vec4, meta['BLOCK_SIZE']),)
            add_kernel_fp16_vec4[grid](x64, y64, out64, n_vec4)
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out