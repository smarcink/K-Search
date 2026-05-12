"""Narrow down: which constexpr combination triggers the SIGSEGV?"""
import subprocess, sys, os, tempfile

PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="ce_bisect_")

TESTS = {
    # Exact GPT kernel 1 signature (4 constexpr after 6 pointer args)
    "gpt_sig_exact": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # Same but only 2 constexpr (H, W)
    "only_H_W_constexpr": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H: tl.constexpr, W: tl.constexpr, NH, NW):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # Same but only NH, NW as constexpr
    "only_NH_NW_constexpr": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H, W, NH: tl.constexpr, NW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # No constexpr at all
    "no_constexpr": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H, W, NH, NW):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 720, 1280, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # GPT sig with smaller values (H=4, W=4)
    "gpt_sig_small_values": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 4, 4, 1, 1, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # GPT sig with medium values
    "gpt_sig_medium_values": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 16, 16, 4, 4, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # Opus sig with GPT values (720, 1280, etc)
    "opus_sig_gpt_values": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(X_ptr, Out_ptr, ln1_w_ptr, ln1_b_ptr,
       B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
       C: tl.constexpr, Wh: tl.constexpr, Ww: tl.constexpr,
       HIDDEN: tl.constexpr, N: tl.constexpr,
       nH: tl.constexpr, nW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, 1, 720, 1280, 32, 4, 4, 128, 16, 180, 320, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # Only H constexpr with value 720
    "only_H_720": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, H: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, 720, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # Only W constexpr with value 1280
    "only_W_1280": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, W: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, 1280, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",

    # H and W constexpr with large values but not 720/1280
    "H256_W256_constexpr": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, qkv_out_ptr, ln_w_ptr, ln_b_ptr, qkv_w_ptr, qkv_b_ptr,
       H: tl.constexpr, W: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x, x, x, x, x, x, 256, 256, 64, 64, num_warps=1)
torch.xpu.synchronize()
print("OK")
""",
}

if __name__ == "__main__":
    for name, code in TESTS.items():
        fpath = os.path.join(tmpdir, f"{name}.py")
        with open(fpath, "w") as f:
            f.write(code)
        print(f"--- {name} ---", flush=True)
        r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(f"  PASS")
        elif r.returncode == -11:
            print(f"  SEGFAULT")
        else:
            print(f"  FAIL (exit {r.returncode})")
            for l in r.stderr.splitlines()[-3:]:
                print(f"  stderr: {l}")
        print(flush=True)
