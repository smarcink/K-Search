"""Bisect SIGSEGV: each test is a .py file run in a subprocess."""
import subprocess
import sys
import os
import tempfile

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = {
    "baseline__stage0_warps1_biggrid_constexpr": """\
import torch, triton, triton.language as tl

@triton.jit
def _empty(x_ptr, out_ptr, H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return

x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_empty[(57600,)](x, out, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "warps2__stage0_biggrid_constexpr": """\
import torch, triton, triton.language as tl

@triton.jit
def _empty(x_ptr, out_ptr, H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return

x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_empty[(57600,)](x, out, 720, 1280, 180, 320, num_warps=2)
torch.xpu.synchronize()
print("OK")
""",

    "smallgrid__stage0_warps1_constexpr": """\
import torch, triton, triton.language as tl

@triton.jit
def _empty(x_ptr, out_ptr, H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return

x = torch.randn(1, 4, 4, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_empty[(1,)](x, out, 4, 4, 1, 1, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "no_constexpr__stage0_warps1_biggrid": """\
import torch, triton, triton.language as tl

@triton.jit
def _empty(x_ptr, out_ptr, H, W, NH, NW):
    return

x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_empty[(57600,)](x, out, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "warps2_no_constexpr__stage0_biggrid": """\
import torch, triton, triton.language as tl

@triton.jit
def _empty(x_ptr, out_ptr, H, W, NH, NW):
    return

x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_empty[(57600,)](x, out, 720, 1280, 180, 320, num_warps=2)
torch.xpu.synchronize()
print("OK")
""",

    "minimal__grid1_warps1": """\
import torch, triton, triton.language as tl

@triton.jit
def _noop(x_ptr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_noop[(1,)](x, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    "minimal__grid57600_warps1": """\
import torch, triton, triton.language as tl

@triton.jit
def _noop(x_ptr):
    return

x = torch.randn(16, dtype=torch.float16, device="xpu")
_noop[(57600,)](x, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",
}


if __name__ == "__main__":
    print(f"Python: {PYTHON}\n")
    tmpdir = tempfile.mkdtemp(prefix="triton_bisect_")
    for name, code in TESTS.items():
        fpath = os.path.join(tmpdir, f"{name}.py")
        with open(fpath, "w") as f:
            f.write(code)
        print(f"--- {name} ---", flush=True)
        result = subprocess.run(
            [PYTHON, fpath],
            capture_output=True, text=True, timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0:
            print(f"  PASS  {stdout}")
        elif result.returncode == -11:
            print(f"  SEGFAULT")
        else:
            print(f"  FAIL  (exit {result.returncode})")
        if stderr and result.returncode != 0:
            lines = stderr.splitlines()
            for line in lines[-3:]:
                print(f"  stderr: {line}")
        print(flush=True)
