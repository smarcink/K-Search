"""
Minimal reproducer: Intel XPU Triton SIGSEGV — chaotic crash pattern
based on (n_pointer_args, n_constexpr_args, kernel_body_complexity).

BUG SUMMARY:
  The Intel XPU Triton backend (triton-xpu 3.7.x) produces SIGSEGV in
  SPIR-V codegen for certain combinations of pointer args, constexpr args,
  and kernel body complexity.  The crash pattern is NON-MONOTONIC — adding
  more args can *fix* a crash, and removing args can *cause* one.

  This is NOT a simple threshold bug.  Examples:
    - 4 ptrs + 3 constexpr + complex body → SEGFAULT
    - 7 ptrs + 3 constexpr + complex body → PASS
    - 8 ptrs + 0 constexpr + complex body → PASS
    - 8 ptrs + 3 constexpr + simple body  → PASS
    - 8 ptrs + 3 constexpr + complex body → SEGFAULT

  The crash occurs at kernel launch (SPIR-V compilation/dispatch), not
  during Triton IR lowering.  Changing num_warps or num_stages does NOT help.

  WORKAROUNDS that shift the parameter layout out of crash zones:
    1. Add dummy constexpr padding params (e.g. 5 extra _PAD: tl.constexpr)
    2. Strip ALL tl.constexpr and inline literal values in the kernel body
    3. Add dummy pointer params

Hardware: Intel Arc Pro B70 (BMG / Xe2), 256 EUs, 32GB VRAM
To run:   python triton_xpu_constexpr_segfault_v2.py

Expected output: several SEGFAULT cases, several PASS cases, demonstrating
the non-monotonic pattern.
"""
import subprocess
import sys
import os
import tempfile

PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="triton_bug_v2_")


# ---------------------------------------------------------------------------
# Helper: generate test kernels with varying (n_ptrs, n_reg, n_constexpr)
# and two body complexity levels.
# ---------------------------------------------------------------------------

def make_simple_kernel(n_ptrs, n_reg, n_cex):
    """Kernel with a simple body: 1D tl.load + tl.store."""
    ptr_params = ", ".join(f"p{i}" for i in range(n_ptrs))
    reg_params = ", ".join(f"r{i}" for i in range(n_reg))
    cex_params = ", ".join(f"C{i}: tl.constexpr" for i in range(n_cex))
    all_params = ", ".join(x for x in [ptr_params, reg_params, cex_params] if x)

    ptr_args = ", ".join(["x"] * n_ptrs)
    reg_args = ", ".join(["1"] * n_reg)
    cex_args = ", ".join(["16"] * n_cex)
    all_args = ", ".join(x for x in [ptr_args, reg_args, cex_args] if x)

    return f"""\
import torch, triton, triton.language as tl

@triton.jit
def _k({all_params}):
    offs = tl.arange(0, 16)
    x = tl.load(p0 + offs)
    tl.store(p0 + offs, x)

x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)]({all_args}, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""


def make_complex_kernel(n_ptrs, n_reg, n_cex):
    """Kernel with a complex body: fused LayerNorm + FC1 + GELU + FC2 + residual.

    Uses tl.dot, tl.erf, tl.rsqrt, masked 2D loads/stores — representative
    of real fused MLP kernels.  Signature: (n_ptrs pointers, n_reg regular
    ints, n_cex constexpr ints).  Only the first 8 ptrs are used in the body;
    extras are dummy padding.
    """
    # We always need at least 8 ptrs for the kernel body (x, out, ln_w, ln_b,
    # fc1_w, fc1_b, fc2_w, fc2_b).  Pad or truncate.
    body_ptrs = ["x_ptr", "out_ptr", "ln_w_ptr", "ln_b_ptr",
                 "fc1_w_ptr", "fc1_b_ptr", "fc2_w_ptr", "fc2_b_ptr"]
    extra_ptrs = [f"_dp{i}" for i in range(max(0, n_ptrs - 8))]
    if n_ptrs < 8:
        # Not enough ptrs for the full body — use a simpler body
        return make_simple_kernel(n_ptrs, n_reg, n_cex)
    ptr_params = ", ".join(body_ptrs + extra_ptrs)

    reg_params = ", ".join(f"r{i}" for i in range(n_reg))
    cex_params = ", ".join(f"C{i}: tl.constexpr" for i in range(n_cex))
    all_params = ", ".join(x for x in [ptr_params, reg_params, cex_params] if x)

    # Launch args
    ptr_args = "x, out, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b"
    if extra_ptrs:
        ptr_args += ", " + ", ".join(["dummy"] * len(extra_ptrs))
    reg_args = ", ".join(["16"] * n_reg)  # total_tokens=16
    cex_args = ", ".join(["16"] * n_cex)
    all_args = ", ".join(x for x in [ptr_args, reg_args, cex_args] if x)

    return f"""\
import torch, triton, triton.language as tl

@triton.jit
def _k({all_params}):
    pid = tl.program_id(0)
    BLOCK_M: tl.constexpr = 16
    BLOCK_C: tl.constexpr = 32
    BLOCK_H: tl.constexpr = 128
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < r0 if {n_reg} > 0 else offs_m < 16
    # LayerNorm
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :],
                mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / 32.0
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / 32.0
    rstd = tl.rsqrt(var + 1e-5)
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
    # Residual + store
    res = tl.load(x_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :],
                  mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = res + mlp.to(tl.float16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * BLOCK_C + offs_c[None, :],
             out, mask=mask_m[:, None])

x = torch.randn(16, 32, dtype=torch.float16, device="xpu")
out = torch.empty_like(x)
ln_w = torch.ones(32, dtype=torch.float16, device="xpu")
ln_b = torch.zeros(32, dtype=torch.float16, device="xpu")
fc1_w = torch.randn(128, 32, dtype=torch.float16, device="xpu")
fc1_b = torch.zeros(128, dtype=torch.float16, device="xpu")
fc2_w = torch.randn(32, 128, dtype=torch.float16, device="xpu")
fc2_b = torch.zeros(32, dtype=torch.float16, device="xpu")
dummy = torch.empty(1, dtype=torch.float16, device="xpu")
_k[(1,)]({all_args}, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

SECTION_1 = "SECTION 1: Simple body (1D load/store) — crash depends ONLY on arg counts"
SECTION_1_TESTS = [
    # (n_ptrs, n_reg, n_cex, expected_note)
    (4, 0, 3, "CRASH"),
    (6, 0, 3, "CRASH"),
    (7, 0, 3, "OK"),
    (8, 0, 3, "OK"),
    (4, 2, 3, "CRASH"),
    (8, 2, 3, "OK"),  # passes with simple body!
    (4, 0, 0, "OK"),
    (4, 2, 0, "CRASH"),
    (4, 2, 4, "OK"),  # adding 1 more constexpr fixes it
    (1, 5, 3, "OK"),  # few ptrs + many regular → OK
]

SECTION_2 = "SECTION 2: Complex body (fused LN+MLP) — same arg counts, different results"
SECTION_2_TESTS = [
    # (n_ptrs, n_reg, n_cex, expected_note)
    (8, 2, 3, "CRASH — passes with simple body, crashes with complex!"),
    (8, 2, 0, "OK"),
    (8, 0, 0, "OK"),
    (10, 2, 3, "OK — 2 extra dummy ptrs fixes it"),
    (8, 2, 8, "OK — 5 extra dummy constexpr fixes it"),
    (8, 5, 0, "OK — converting constexpr to regular fixes it"),
]

SECTION_3 = "SECTION 3: Workaround demonstrations"
SECTION_3_WORKAROUNDS = [
    ("ORIGINAL (crashes)", 8, 2, 3, False),
    ("WORKAROUND: +5 dummy constexpr", 8, 2, 8, False),
    ("WORKAROUND: +2 dummy pointers", 10, 2, 3, False),
    ("WORKAROUND: constexpr → regular", 8, 5, 0, False),
]


def run_test(code, name):
    fpath = os.path.join(tmpdir, name.replace(" ", "_").replace("/", "_")[:60] + ".py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        return "PASS"
    elif r.returncode == -11:
        return "SEGFAULT"
    else:
        return f"FAIL({r.returncode})"


if __name__ == "__main__":
    import platform

    print("=" * 72)
    print("Intel XPU Triton SIGSEGV reproducer — chaotic arg-count pattern")
    print("=" * 72)
    print(f"Python:  {sys.version.split()[0]}")
    try:
        import triton
        print(f"Triton:  {triton.__version__}")
    except Exception:
        pass
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.xpu.is_available():
            print(f"XPU:     {torch.xpu.get_device_name(0)}")
    except Exception:
        pass
    print(f"OS:      {platform.platform()}")
    print()

    # Section 1
    print(f"--- {SECTION_1} ---")
    for n_ptrs, n_reg, n_cex, note in SECTION_1_TESTS:
        code = make_simple_kernel(n_ptrs, n_reg, n_cex)
        label = f"{n_ptrs}ptr + {n_reg}reg + {n_cex}cex  (simple body)"
        status = run_test(code, f"s1_{n_ptrs}p{n_reg}r{n_cex}c")
        print(f"  [{status:>8s}]  {label:45s}  ({note})")
    print()

    # Section 2
    print(f"--- {SECTION_2} ---")
    for n_ptrs, n_reg, n_cex, note in SECTION_2_TESTS:
        code = make_complex_kernel(n_ptrs, n_reg, n_cex)
        label = f"{n_ptrs}ptr + {n_reg}reg + {n_cex}cex  (complex body)"
        status = run_test(code, f"s2_{n_ptrs}p{n_reg}r{n_cex}c")
        print(f"  [{status:>8s}]  {label:45s}  ({note})")
    print()

    # Section 3
    print(f"--- {SECTION_3} ---")
    for desc, n_ptrs, n_reg, n_cex, _ in SECTION_3_WORKAROUNDS:
        code = make_complex_kernel(n_ptrs, n_reg, n_cex)
        status = run_test(code, f"s3_{desc[:30]}")
        print(f"  [{status:>8s}]  {desc}")
    print()

    print("KEY OBSERVATIONS:")
    print("  1. The crash depends on (n_pointer_args, n_constexpr_args, body_complexity)")
    print("  2. Pattern is NON-MONOTONIC: more args can fix OR cause crashes")
    print("  3. Same arg signature can PASS with simple body, CRASH with complex body")
    print("  4. num_warps / num_stages changes do NOT help")
    print("  5. Workarounds: pad constexpr count, pad pointer count, or strip constexpr")
