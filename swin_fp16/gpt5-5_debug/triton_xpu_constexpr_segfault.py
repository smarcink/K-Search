"""
Minimal reproducer: Intel XPU Triton SIGSEGV with specific
pointer-count + tl.constexpr-count combination.

Bug: A @triton.jit kernel with 6 pointer args + 4 tl.constexpr args
segfaults on launch (even an empty kernel body). Removing tl.constexpr
from those 4 args fixes it. Different pointer/constexpr ratios
(e.g. 4 ptrs + 10 constexpr, or 1 ptr + 1 constexpr) work fine.

Hardware: Intel Arc Pro B70 (BMG)
To run:  python triton_xpu_constexpr_segfault.py
"""
import subprocess
import sys
import os
import tempfile

PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="triton_bug_")

CASES = {
    # -------- CRASHES --------
    "CRASH: 6 ptrs + 4 constexpr (empty body)": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3, p4, p5,
            A: tl.constexpr, B: tl.constexpr,
            C: tl.constexpr, D: tl.constexpr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, x, x, 4, 4, 1, 1, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "CRASH: 6 ptrs + 4 constexpr (large values)": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3, p4, p5,
            A: tl.constexpr, B: tl.constexpr,
            C: tl.constexpr, D: tl.constexpr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "CRASH: 6 ptrs + 4 constexpr, num_warps=2": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3, p4, p5,
            A: tl.constexpr, B: tl.constexpr,
            C: tl.constexpr, D: tl.constexpr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=2)
torch.xpu.synchronize()
print("OK")
""",

    # -------- WORKS --------
    "OK: 6 ptrs + 4 regular args (no constexpr)": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3, p4, p5, A, B, C, D):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "OK: 6 ptrs + 2 constexpr + 2 regular": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3, p4, p5,
            A: tl.constexpr, B: tl.constexpr, C, D):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "OK: 4 ptrs + 10 constexpr": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0, p1, p2, p3,
            A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
            D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
            G: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
            J: tl.constexpr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, x, x, x, 1, 720, 1280, 32, 4, 4, 128, 16, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "OK: 1 ptr + 0 constexpr (minimal)": """\
import torch, triton, triton.language as tl

@triton.jit
def _kernel(p0):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_kernel[(1,)](x, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",
}


if __name__ == "__main__":
    import platform
    print("=== Triton XPU constexpr SIGSEGV reproducer ===")
    print(f"Python: {sys.version}")
    try:
        import triton; print(f"Triton: {triton.__version__}")
    except Exception:
        pass
    try:
        import torch; print(f"PyTorch: {torch.__version__}")
        if torch.xpu.is_available():
            print(f"XPU device: {torch.xpu.get_device_name(0)}")
    except Exception:
        pass
    print(f"Platform: {platform.platform()}")
    print()

    for name, code in CASES.items():
        fpath = os.path.join(tmpdir, name.replace(" ", "_").replace(":", "").replace("/", "_") + ".py")
        with open(fpath, "w") as f:
            f.write(code)
        r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            status = "PASS"
        elif r.returncode == -11:
            status = "SEGFAULT"
        else:
            status = f"FAIL(exit={r.returncode})"
        print(f"  [{status:>8s}]  {name}")
