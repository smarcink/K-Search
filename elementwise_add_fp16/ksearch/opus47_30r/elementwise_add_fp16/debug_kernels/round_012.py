import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel_packed(x_ptr, y_ptr, out_ptr, n_pairs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pairs
    # Load as uint32 (packed 2x fp16)
    x_packed = tl.load(x_ptr + offsets, mask=mask)
    y_packed = tl.load(y_ptr + offsets, mask=mask)
    # Reinterpret as 2x fp16, add, and pack back
    x_f16 = x_packed.to(tl.uint32, bitcast=True)
    y_f16 = y_packed.to(tl.uint32, bitcast=True)
    # We need to do the add as fp16 elementwise. Easier: load as fp16 with double-width vector.
    # Fallback: do via float view trick below.
    tl.store(out_ptr + offsets, x_packed + y_packed, mask=mask)


@triton.jit
def add_kernel_fp16x2(x_ptr, y_ptr, out_ptr, n_pairs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_pairs
    # offs indexes pairs; load 2 fp16 at a time by treating as contiguous fp16 with stride
    base = offs * 2
    idx0 = base
    idx1 = base + 1
    x0 = tl.load(x_ptr + idx0, mask=mask)
    x1 = tl.load(x_ptr + idx1, mask=mask)
    y0 = tl.load(y_ptr + idx0, mask=mask)
    y1 = tl.load(y_ptr + idx1, mask=mask)
    tl.store(out_ptr + idx0, x0 + y0, mask=mask)
    tl.store(out_ptr + idx1, x1 + y1, mask=mask)


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
    key=['n_elements'],
)
@triton.jit
def add_kernel_f32view(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # x_ptr, y_ptr, out_ptr are viewed as fp32 buffers; each fp32 element holds 2 fp16.
    # We bitcast load -> 2x fp16 -> add -> bitcast store.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    xv = tl.load(x_ptr + offs, mask=mask)
    yv = tl.load(y_ptr + offs, mask=mask)
    # bitcast fp32 -> 2x fp16 via reshape trick: reinterpret as uint32 then split
    # Simpler: cast to fp16x2 directly using .to with bitcast on a vector type isn't trivial.
    # Use the fact that we can load as fp16 with twice the elements:
    # But here we've already loaded as fp32. Add as fp32 won't be correct.
    # Instead: do nothing here; this kernel is unused. We use add_kernel_fp16_vec below.
    tl.store(out_ptr + offs, xv, mask=mask)


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

        # Packed-vector path: view as fp32 (2 fp16 per fp32) when n is even.
        if (n % 2 == 0) and (x.data_ptr() % 4 == 0) and (y.data_ptr() % 4 == 0) and (out.data_ptr() % 4 == 0):
            x32 = x.view(torch.float32)
            y32 = y.view(torch.float32)
            out32 = out.view(torch.float32)
            # Each fp32 element packs 2 fp16. We need to add as fp16x2, not fp32.
            # Trick: load fp16 directly with doubled element count - but we want wide transactions.
            # Use fp16 kernel directly; Triton emits 128-bit loads for contiguous fp16 with large BLOCK_SIZE.
            pass

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out