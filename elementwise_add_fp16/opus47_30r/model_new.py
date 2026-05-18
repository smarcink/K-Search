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
    ],
    key=['n_pairs'],
)
@triton.jit
def add_kernel_fp16x2_pack(x_ptr, y_ptr, out_ptr, n_pairs, BLOCK_SIZE: tl.constexpr):
    # x_ptr/y_ptr/out_ptr are uint32* (each word holds 2 fp16). We bitcast to fp16
    # vectors of length 2*BLOCK_SIZE so Triton emits wide (128-bit) loads/stores
    # and performs proper elementwise fp16 add.
    pid = tl.program_id(0)
    word_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_w = word_offs < n_pairs
    x_w = tl.load(x_ptr + word_offs, mask=mask_w, other=0)
    y_w = tl.load(y_ptr + word_offs, mask=mask_w, other=0)
    # Reinterpret packed uint32 as 2 fp16 lanes via join trick:
    # split into low/high fp16 halves.
    x_lo = (x_w & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    x_hi = ((x_w >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y_lo = (y_w & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    y_hi = ((y_w >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    z_lo = x_lo + y_lo
    z_hi = x_hi + y_hi
    z_w = (z_lo.to(tl.uint16, bitcast=True).to(tl.uint32)) | \
          (z_hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
    tl.store(out_ptr + word_offs, z_w, mask=mask_w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        # Use the simple fp16 kernel; Triton emits 128-bit vector loads for
        # large contiguous BLOCK_SIZE values. This was the best base score.
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out