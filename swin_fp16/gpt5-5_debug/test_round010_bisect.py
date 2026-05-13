"""Bisect round_010 crash by progressively adding operations."""
import subprocess, sys, os, tempfile

PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="r010_bisect_")

TESTS = {
    # Stage 0: exact signature, empty body
    "s0_empty_body": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    return
x = torch.randn(16, dtype=torch.float16, device="xpu")
_fused_norm2_mlp_kernel[(1,)](x,x,x,x,x,x,x,x, 1, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    # Stage 1: load x
    "s1_load_x": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0)
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_fused_norm2_mlp_kernel[(1,)](x,out,x,x,x,x,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    # Stage 2: + layernorm
    "s2_layernorm": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
_fused_norm2_mlp_kernel[(1,)](x,x,ln_w,ln_b,x,x,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    # Stage 3: + fc1 dot
    "s3_fc1_dot": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
_fused_norm2_mlp_kernel[(1,)](x,x,ln_w,ln_b,fc1_w,fc1_b,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    # Stage 4: + gelu (tl.erf)
    "s4_gelu": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    h_half = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_half * (1.0 + tl.erf(h_half * 0.7071067811865476))
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
_fused_norm2_mlp_kernel[(1,)](x,x,ln_w,ln_b,fc1_w,fc1_b,x,x, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",

    # Stage 5: + fc2 dot + residual + store (full kernel)
    "s5_full": """\
import torch, triton, triton.language as tl
@triton.jit
def _fused_norm2_mlp_kernel(
    x_ptr, out_ptr, ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
    total_tokens, eps,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    h_half = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_half * (1.0 + tl.erf(h_half * 0.7071067811865476))
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * BLOCK_H + offs_h[:, None])
    acc2 = tl.dot(gelu.to(tl.float16), w2)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    mlp = acc2 + b2[None, :]
    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], out, mask=mask_m[:, None])
x = torch.randn(16*32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
out = torch.empty(16*32, dtype=torch.float16, device="xpu")
_fused_norm2_mlp_kernel[(1,)](x,out,ln_w,ln_b,fc1_w,fc1_b,fc2_w,fc2_b, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
""",
}

for name, code in TESTS.items():
    fpath = os.path.join(tmpdir, f"{name}.py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        s = "PASS"
    elif r.returncode == -11:
        s = "SEGFAULT"
    else:
        s = f"FAIL({r.returncode})"
        # show last error line
    print(f"  [{s:>8s}]  {name}")
    if r.returncode not in (0, -11) and r.stderr:
        print(f"            {r.stderr.splitlines()[-1][:100]}")
