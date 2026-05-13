"""Test: does body complexity affect crash for same param signature?
We know 8ptr+2reg+3cex (total=13) PASSes with a simple load body.
Does it also pass with a complex body (LN+2xdot+GELU)?"""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="body_complexity_")

# Test 1: 8ptr+2reg+3cex with simple body (should pass per sweep)
t1 = """\
import torch, triton, triton.language as tl
@triton.jit
def _k(p0,p1,p2,p3,p4,p5,p6,p7, r0, r1, C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    offs = tl.arange(0, C0)
    x = tl.load(p0 + offs)
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)](x,x,x,x,x,x,x,x, 1, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

# Test 2: 8ptr+2reg+3cex with complex body matching round_010 ops
t2 = """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, ln_w_ptr, ln_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
       total_tokens, eps,
       BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    # LayerNorm
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / BLOCK_C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    # FC1
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    # GELU
    h_f = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_f * (1.0 + tl.erf(h_f * 0.7071067811865476))
    # FC2
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * BLOCK_H + offs_h[:, None])
    acc2 = tl.dot(gelu.to(tl.float16), w2)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    mlp = acc2 + b2[None, :]
    # Residual
    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], out.to(tl.float16), mask=mask_m[:, None])

x = torch.randn(16, 32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

# Test 3: same complex body but with different ptr count (pad to 10)
t3 = """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, ln_w_ptr, ln_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
       _pad0, _pad1,
       total_tokens, eps,
       BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / BLOCK_C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    h_f = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_f * (1.0 + tl.erf(h_f * 0.7071067811865476))
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * BLOCK_H + offs_h[:, None])
    acc2 = tl.dot(gelu.to(tl.float16), w2)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    mlp = acc2 + b2[None, :]
    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], out.to(tl.float16), mask=mask_m[:, None])

x = torch.randn(16, 32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
dummy = torch.empty(1, dtype=torch.float16, device="xpu")
_k[(1,)](x, out, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b, dummy, dummy, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

# Test 4: strip all constexpr from the complex body kernel
t4 = """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, ln_w_ptr, ln_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
       total_tokens, eps, BLOCK_M, BLOCK_C, BLOCK_H):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, 16)
    offs_c = tl.arange(0, 32)
    offs_h = tl.arange(0, 128)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * 32 + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * 32 + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    h_f = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_f * (1.0 + tl.erf(h_f * 0.7071067811865476))
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * 128 + offs_h[:, None])
    acc2 = tl.dot(gelu.to(tl.float16), w2)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    mlp = acc2 + b2[None, :]
    res = tl.load(x_ptr + offs_m[:, None] * 32 + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * 32 + offs_c[None, :], out.to(tl.float16), mask=mask_m[:, None])

x = torch.randn(16, 32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b, 16, 1e-5, 16, 32, 128, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

# Test 5: pad to MORE constexpr (add 5 dummy cex to get 8ptr+2reg+8cex=total 18)
t5 = """\
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, out_ptr, ln_w_ptr, ln_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr,
       total_tokens, eps,
       BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
       _PAD0: tl.constexpr, _PAD1: tl.constexpr, _PAD2: tl.constexpr, _PAD3: tl.constexpr, _PAD4: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < total_tokens
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / BLOCK_C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    rstd = tl.rsqrt(var + eps)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    y = (xc * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]
    w1 = tl.load(fc1_w_ptr + offs_h[None, :] * BLOCK_C + offs_c[:, None])
    acc1 = tl.dot(y.to(tl.float16), w1)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h = acc1 + b1[None, :]
    h_f = h.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h_f * (1.0 + tl.erf(h_f * 0.7071067811865476))
    w2 = tl.load(fc2_w_ptr + offs_c[None, :] * BLOCK_H + offs_h[:, None])
    acc2 = tl.dot(gelu.to(tl.float16), w2)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    mlp = acc2 + b2[None, :]
    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :], out.to(tl.float16), mask=mask_m[:, None])

x = torch.randn(16, 32, dtype=torch.float16, device="xpu")
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
_k[(1,)](x, out, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b, 16, 1e-5, 16, 32, 128, 0,0,0,0,0, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

tests = {
    "t1_simple_body_8p2r3c": t1,
    "t2_complex_body_8p2r3c": t2,
    "t3_complex_body_10p2r3c_padded": t3,
    "t4_complex_body_8p5r0c_no_cex": t4,
    "t5_complex_body_8p2r8c_padcex": t5,
}

for name, code in tests.items():
    fpath = os.path.join(tmpdir, f"{name}.py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0: s = "PASS"
    elif r.returncode == -11: s = "SEGFAULT"
    else: s = f"FAIL({r.returncode})"
    print(f"  [{s:>8s}]  {name}")
    if r.returncode not in (0,) and r.stderr:
        lines = r.stderr.strip().splitlines()
        if lines:
            print(f"            {lines[-1][:120]}")
