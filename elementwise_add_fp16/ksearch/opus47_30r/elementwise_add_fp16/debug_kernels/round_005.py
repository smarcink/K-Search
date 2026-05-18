import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel_vec(x_ptr, y_ptr, out_ptr, n_vec, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vec
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_kernel_vec_nomask(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@triton.jit
def add_kernel_scalar(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        y_c = y.contiguous()
        out = torch.empty_like(x_c)
        n_elements = out.numel()

        # Try to view as float4 (8 fp16 per element) for 128-bit loads
        # fp16 -> view as int64 packs 4 fp16 into 64 bits; use float4 via uint4 pattern
        # Simpler: view as uint4 by reinterpret -> use 8x fp16 per element through pack of 4 fp32
        if n_elements % 8 == 0 and x_c.data_ptr() % 16 == 0 and y_c.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0:
            # View tensors as float4 packed (each element = 8 fp16 = 16 bytes)
            n_vec = n_elements // 4  # process 4 fp16 per scalar via float2 trick? Use 2 fp16 as float.
            # Actually simplest: reinterpret as float32 (= 2 fp16) and add as int via bit trick won't work.
            # Use bfloat-safe approach: cast view to float32 not valid since add changes bits.
            # Stick with fp16 but use very large BLOCK_SIZE.
            pass

        BLOCK = 8192
        if n_elements % BLOCK == 0:
            grid = (n_elements // BLOCK,)
            add_kernel_vec_nomask[grid](x_c, y_c, out, BLOCK_SIZE=BLOCK, num_warps=4)
        else:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel_scalar[grid](x_c, y_c, out, n_elements, BLOCK_SIZE=BLOCK, num_warps=4)
        return out