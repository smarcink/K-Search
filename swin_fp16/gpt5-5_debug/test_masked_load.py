"""Bisect: is masked tl.load the crash trigger?"""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="mask_bisect_")

TESTS = {
    "load_no_mask": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :])
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], x)
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, 16, 32, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    "load_with_mask": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, total_tokens, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], x, mask=mask_m[:, None])
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, 16, 16, 32, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    "load_with_mask_num_stages_1": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, total_tokens, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], x, mask=mask_m[:, None])
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, 16, 16, 32, num_warps=4, num_stages=1)
torch.xpu.synchronize()
print("OK")
""",

    "load_with_mask_num_warps_1": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, total_tokens, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], x, mask=mask_m[:, None])
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, 16, 16, 32, num_warps=1, num_stages=1)
torch.xpu.synchronize()
print("OK")
""",

    "load_with_2d_mask_both_dims": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask = (offs_m[:, None] < M) & (offs_c[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], x, mask=mask)
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, 16, 32, 16, 32, num_warps=1, num_stages=1)
torch.xpu.synchronize()
print("OK")
""",

    "8_ptrs_masked_load": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(p0, p1, p2, p3, p4, p5, p6, p7, total_tokens, eps, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(p0 + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0)
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
_k[(1,)](x,x,x,x,x,x,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    "8_ptrs_no_mask": """\
import torch, triton, triton.language as tl
@triton.jit
def _k(p0, p1, p2, p3, p4, p5, p6, p7, total_tokens, eps, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    x = tl.load(p0 + offs_m[:, None] * BLOCK_C + offs_c[None, :])
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
_k[(1,)](x,x,x,x,x,x,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",
}

for name, code in TESTS.items():
    fpath = os.path.join(tmpdir, f"{name}.py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0: s = "PASS"
    elif r.returncode == -11: s = "SEGFAULT"
    else: s = f"FAIL({r.returncode})"
    print(f"  [{s:>8s}]  {name}")
