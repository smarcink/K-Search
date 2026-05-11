import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_add_fp16_vectorized_persistent(
    x_ptr,
    y_ptr,
    z_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles multiple chunks via a persistent loop
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Each fp16 element is 2 bytes; we load pairs as int32 (4 bytes = 2 fp16)
    # So the number of int32 chunks is N // 2
    N_int32 = N // 2
    BLOCK_INT32: tl.constexpr = BLOCK_SIZE // 2

    chunk_id = pid
    while chunk_id * BLOCK_INT32 < N_int32:
        offsets = chunk_id * BLOCK_INT32 + tl.arange(0, BLOCK_INT32)
        mask = offsets < N_int32

        # Load as int32 to get 2 fp16 values per load (vectorized 32-bit load)
        x_int32 = tl.load(x_ptr.to(tl.pointer_type(tl.int32)) + offsets, mask=mask, other=0)
        y_int32 = tl.load(y_ptr.to(tl.pointer_type(tl.int32)) + offsets, mask=mask, other=0)

        # Unpack the two fp16 values from each int32
        # Low 16 bits = first element, high 16 bits = second element
        x_lo = (x_int32 & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        x_hi = ((x_int32 >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        y_lo = (y_int32 & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        y_hi = ((y_int32 >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)

        # Add the fp16 values
        z_lo = x_lo + y_lo
        z_hi = x_hi + y_hi

        # Repack into int32
        z_lo_int = z_lo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        z_hi_int = (z_hi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) << 16
        z_int32 = z_lo_int | z_hi_int

        # Store as int32
        tl.store(z_ptr.to(tl.pointer_type(tl.int32)) + offsets, z_int32, mask=mask)

        chunk_id += num_programs


class ModelNew(torch.nn.Module):
    """Optimized elementwise addition using vectorized persistent Triton kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Flatten to 1D
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        N = x_flat.numel()

        z_flat = torch.empty_like(x_flat)

        # Use 512 workgroups (2x the 256 EUs) for better occupancy
        NUM_WG = 512
        BLOCK_SIZE = 8192  # fp16 elements per block per iteration

        # Handle odd N: if N is odd, we can't perfectly vectorize
        # Fall back to plain kernel for the last element if needed
        if N % 2 != 0:
            # For simplicity, fall back to PyTorch for odd sizes
            return x + y

        elementwise_add_fp16_vectorized_persistent[(NUM_WG,)](
            x_flat,
            y_flat,
            z_flat,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return z_flat.view_as(x)


# KernelBench convention
def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_inputs():
    device = get_device()
    x = torch.randn(16, 256, 256, dtype=torch.float16, device=device)
    y = torch.randn(16, 256, 256, dtype=torch.float16, device=device)
    return [x, y]


def get_init_inputs():
    return []