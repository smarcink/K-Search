"""Check: does the Opus-style constexpr signature also crash, or only the GPT-style?"""
import torch, triton, triton.language as tl

# Opus-style signature (more constexpr params, different names)
@triton.jit
def _opus_style(
    X_ptr, Out_ptr,
    ln1_w_ptr, ln1_b_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C: tl.constexpr, Wh: tl.constexpr, Ww: tl.constexpr,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    nH: tl.constexpr, nW: tl.constexpr,
):
    return

dev = "xpu"
x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device=dev)
out = torch.empty_like(x)
ln_w = torch.randn(32, dtype=torch.float16, device=dev)

_opus_style[(57600,)](
    x, out, ln_w, ln_w,
    1, 720, 1280, 32, 4, 4, 128, 16, 180, 320,
    num_warps=2, num_stages=1,
)
torch.xpu.synchronize()
print("OK - opus-style constexpr works")
