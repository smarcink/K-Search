import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_STAGES': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_STAGES': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_STAGES': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_STAGES': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_STAGES': 1}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_STAGES': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_STAGES': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_STAGES': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_STAGES': 2}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_fp16_gridstride(x_ptr, y_ptr, out_ptr, n_elements,
                                num_blocks, BLOCK_SIZE: tl.constexpr,
                                NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for b in tl.range(pid, num_blocks, num_programs, num_stages=NUM_STAGES):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # RTX5090 has ~170 SMs; use a multiple for grid-stride
        self.num_ctas = 680

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        def grid(meta):
            num_blocks = triton.cdiv(n, meta['BLOCK_SIZE'])
            return (min(self.num_ctas, num_blocks),)

        # Precompute num_blocks inside kernel using BLOCK_SIZE; pass as arg
        # We pass a placeholder and compute in launcher via lambda over meta
        def launcher():
            pass

        # Use a wrapping grid that also computes num_blocks param
        BLOCK_PROBE = None

        # Simpler: launch with grid lambda and pass num_blocks computed from meta
        def grid2(meta):
            nb = triton.cdiv(n, meta['BLOCK_SIZE'])
            return (min(self.num_ctas, nb),)

        # We need num_blocks as kernel arg; compute via autotune meta by passing n_elements
        # and recomputing num_blocks inside kernel from BLOCK_SIZE. Do that:
        add_kernel_fp16_gridstride[grid2](
            x, y, out, n,
            triton.cdiv(n, 1),  # placeholder, recomputed below via reassignment
        )
        return out


# Rewrite cleanly: compute num_blocks inside kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_fp16_gs(x_ptr, y_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for b in range(pid, num_blocks, num_programs):
        offs = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_ctas = 680

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        def grid(meta):
            nb = triton.cdiv(n, meta['BLOCK_SIZE'])
            return (min(self.num_ctas, nb),)

        add_kernel_fp16_gs[grid](x, y, out, n)
        return out