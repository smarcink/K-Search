import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_add_kernel_vec(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE fp16 elements, but we load them as
    # int32 (packing 2 fp16 values per int32) to get 64-bit loads per thread,
    # effectively issuing 128-bit loads per group of 2 threads (coalesced).
    # We use BLOCK_SIZE/8 int32x4 (128-bit) loads per block.
    VEC: tl.constexpr = 8  # 8 fp16 elements = 128 bits

    pid = tl.program_id(axis=0)
    # Number of vectorized elements per block
    block_start = pid * (BLOCK_SIZE // VEC)
    offsets = block_start + tl.arange(0, BLOCK_SIZE // VEC)
    n_vec = n_elements // VEC
    mask = offsets < n_vec

    # Cast pointers to int64 for vectorized 128-bit loads
    # Load as fp16x8 via reinterpretation: load 4 x int32 = 128 bits
    x_vec = tl.load(x_ptr + offsets * VEC, mask=mask, other=0, eviction_policy='evict_last')
    # We need to load 8 fp16 at a time; use tl.load with a 2D shape trick
    # Actually, load VEC=8 fp16 elements as a contiguous chunk per "lane"
    # Triton does coalesced 128-bit loads automatically when stride=1 and
    # BLOCK_SIZE is a multiple of 8 for fp16. We rely on that here.
    # The simplest correct approach: load with the original scalar indexing
    # but let Triton vectorize via the contiguous offset pattern.

    # Fall back to scalar loads (Triton auto-vectorizes contiguous access)
    scalar_start = pid * BLOCK_SIZE
    scalar_offsets = scalar_start + tl.arange(0, BLOCK_SIZE)
    scalar_mask = scalar_offsets < n_elements
    x = tl.load(x_ptr + scalar_offsets, mask=scalar_mask, other=0.0, cache_modifier='.cg')
    y = tl.load(y_ptr + scalar_offsets, mask=scalar_mask, other=0.0, cache_modifier='.cg')
    out = x + y
    tl.store(out_ptr + scalar_offsets, out, mask=scalar_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, cache_modifier='.cg')
    y = tl.load(y_ptr + offsets, mask=mask, cache_modifier='.cg')
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    """Optimized elementwise addition using autotuned Triton kernel with cache-streaming loads."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        elementwise_add_kernel[grid](
            x, y, out,
            n_elements,
        )
        return out