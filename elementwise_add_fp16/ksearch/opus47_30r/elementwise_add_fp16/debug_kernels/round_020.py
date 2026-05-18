import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_fp16x2_gs(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # n_elements counts uint32 words (2 fp16 each)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    stride = num_pid * BLOCK_SIZE
    base_offs = tl.arange(0, BLOCK_SIZE)
    for off in range(block_start, n_elements, stride):
        offs = off + base_offs
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
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


# Number of SMs on RTX5090 (Blackwell) - approximate; used for grid-stride sizing.
_NUM_SMS = None


def _get_num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        try:
            _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
        except Exception:
            _NUM_SMS = 170
    return _NUM_SMS


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous() and y.is_contiguous()
        assert x.dtype == torch.float16 and y.dtype == torch.float16
        out = torch.empty_like(x)
        n = x.numel()

        if (n % 2 == 0) and (x.data_ptr() % 4 == 0) and (y.data_ptr() % 4 == 0) and (out.data_ptr() % 4 == 0):
            x32 = x.view(torch.float32)
            y32 = y.view(torch.float32)
            out32 = out.view(torch.float32)
            n_pairs = n // 2
            num_sms = _get_num_sms()
            # Launch ~4 waves of CTAs across SMs for grid-stride loop.
            def grid(meta):
                ctas = triton.cdiv(n_pairs, meta['BLOCK_SIZE'])
                return (min(ctas, num_sms * 8),)
            # Use packed add: f32 add isn't correct for two fp16s, but we want fp16 add.
            # Reinterpret as fp16 vector by loading via fp16 pointer with 2*BLOCK_SIZE.
            # Simpler: use the original kernel viewing memory as fp16 directly.
            add_kernel_fp16x2_gs[grid](
                x.view(torch.float16), y.view(torch.float16), out.view(torch.float16), n
            )
            return out

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel_fp16[grid](x, y, out, n)
        return out