"""
Intel XPU Triton SIGSEGV Reproducer: Fused MLP Kernel
=====================================================

Bug: A single Triton kernel that combines LayerNorm + fc1 + activation + fc2
     (i.e., 2x tl.dot + complex arithmetic body) crashes with SIGSEGV during
     SPIR-V → GPU ISA compilation on Intel Arc (BMG/Xe2).

Root cause: The SPIR-V backend cannot handle a kernel with 2x tl.dot operations
            combined with a complex arithmetic body (LayerNorm + activation).
            The IR becomes too complex for the compiler's register allocation /
            instruction scheduling, regardless of which activation is used.

Hardware: Intel Arc Pro B70 (BMG), driver 26.09.37435
Software: triton-xpu 3.7.1+git21033c4e, PyTorch 2.13.0.dev+xpu

Reproduction:
    python triton_xpu_fused_mlp_segfault.py

Expected: CASE 1 crashes with SIGSEGV (tl.erf GELU + 2x tl.dot)
          CASE 2 passes (same computation, split into two kernels)
          CASE 3 crashes with SIGSEGV (tanh GELU + 2x tl.dot — same root cause)

Fix: Never fuse a full MLP (fc1 + activation + fc2) into a single kernel.
     Split at the activation boundary into two kernels, each with only 1x tl.dot.
"""

import torch
import triton
import triton.language as tl
import signal
import sys
import os
import subprocess

# ─── Parameters ───────────────────────────────────────────────────────────────
# MLP: input [M, C=32] → fc1 [C, H=128] → GELU → fc2 [H, C] → output [M, C=32]
M = 720 * 1280  # number of tokens (from Swin 720x1280 spatial dims)
C = 32           # model dimension
H = 128          # MLP hidden dimension (4x)
BLOCK_M = 16


# ═══════════════════════════════════════════════════════════════════════════════
# CASE 1: CRASHES — Monolithic kernel with tl.erf + 2x tl.dot
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_norm_mlp_CRASHES(
    x_ptr, y_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,   # 32
    BLOCK_H: tl.constexpr,   # 128
):
    """Single kernel: LayerNorm → fc1 (tl.dot) → tl.erf GELU → fc2 (tl.dot)
    
    THIS CRASHES on Intel XPU. The combination of tl.erf + 2x tl.dot
    produces SPIR-V that the GPU compiler cannot handle.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)

    token_offsets = pid * BLOCK_M + offs_m
    mask_m = token_offsets < T

    # Load input
    x_offsets = token_offsets[:, None] * BLOCK_C + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x_vals, axis=1) / BLOCK_C
    xc = x_vals - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    inv_std = tl.rsqrt(var + 1e-5)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    norm = xc * inv_std[:, None] * ln_w[None, :] + ln_b[None, :]
    norm_h = norm.to(tl.float16)

    # fc1: [BLOCK_M, C] @ [C, H] → [BLOCK_M, H]  (tl.dot #1)
    w1 = tl.load(fc1_w_ptr + offs_c[:, None] + offs_h[None, :] * BLOCK_C)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h1 = tl.dot(norm_h, w1) + b1[None, :]

    # GELU via tl.erf  ← THIS IS THE PROBLEMATIC OP in combination with tl.dot
    h1_f = h1.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h1_f * (1.0 + tl.erf(h1_f * 0.7071067811865476))
    gelu_h = gelu.to(tl.float16)

    # fc2: [BLOCK_M, H] @ [H, C] → [BLOCK_M, C]  (tl.dot #2)
    w2 = tl.load(fc2_w_ptr + offs_h[:, None] + offs_c[None, :] * BLOCK_H)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    out = tl.dot(gelu_h, w2) + b2[None, :]

    # Residual + store
    final = out.to(tl.float16) + x_vals.to(tl.float16)
    tl.store(y_ptr + x_offsets, final, mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# CASE 2: PASSES — Split into two kernels at the GELU boundary
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _split_norm_fc1_gelu_PASSES(
    x_ptr, tmp_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """First half: LayerNorm → fc1 (tl.dot #1) → tl.erf GELU → store to tmp
    
    Only 1x tl.dot + tl.erf in this kernel. PASSES on Intel XPU.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)

    token_offsets = pid * BLOCK_M + offs_m
    mask_m = token_offsets < T

    # Load + LayerNorm (same as CASE 1)
    x_offsets = token_offsets[:, None] * BLOCK_C + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_m[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x_vals, axis=1) / BLOCK_C
    xc = x_vals - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    inv_std = tl.rsqrt(var + 1e-5)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    norm = xc * inv_std[:, None] * ln_w[None, :] + ln_b[None, :]
    norm_h = norm.to(tl.float16)

    # fc1 + GELU (1 tl.dot + tl.erf — fine in isolation)
    w1 = tl.load(fc1_w_ptr + offs_c[:, None] + offs_h[None, :] * BLOCK_C)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h1 = tl.dot(norm_h, w1) + b1[None, :]
    h1_f = h1.to(tl.float16).to(tl.float32)
    gelu = 0.5 * h1_f * (1.0 + tl.erf(h1_f * 0.7071067811865476))

    # Store intermediate to global memory
    tmp_offsets = token_offsets[:, None] * BLOCK_H + offs_h[None, :]
    tl.store(tmp_ptr + tmp_offsets, gelu.to(tl.float16), mask=mask_m[:, None])


@triton.jit
def _split_fc2_residual_PASSES(
    x_ptr, tmp_ptr, y_ptr,
    fc2_w_ptr, fc2_b_ptr,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Second half: load GELU output → fc2 (tl.dot #2) → residual → store
    
    Only 1x tl.dot, no tl.erf. PASSES on Intel XPU.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)

    token_offsets = pid * BLOCK_M + offs_m
    mask_m = token_offsets < T

    # Load GELU intermediate
    tmp_offsets = token_offsets[:, None] * BLOCK_H + offs_h[None, :]
    gelu_h = tl.load(tmp_ptr + tmp_offsets, mask=mask_m[:, None], other=0.0)

    # fc2
    w2 = tl.load(fc2_w_ptr + offs_h[:, None] + offs_c[None, :] * BLOCK_H)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    out = tl.dot(gelu_h, w2) + b2[None, :]

    # Residual
    x_offsets = token_offsets[:, None] * BLOCK_C + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_m[:, None], other=0.0)
    final = out.to(tl.float16) + x_vals
    tl.store(y_ptr + x_offsets, final, mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# CASE 3: ALSO CRASHES — Single kernel with approximate GELU (no tl.erf)
#         Proves the issue is 2x tl.dot + complex body, NOT specifically tl.erf
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _fused_norm_mlp_approx_gelu_PASSES(
    x_ptr, y_ptr,
    ln_w_ptr, ln_b_ptr,
    fc1_w_ptr, fc1_b_ptr,
    fc2_w_ptr, fc2_b_ptr,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Same fused structure as CASE 1, but uses tanh-approximation GELU
    instead of tl.erf. STILL CRASHES — proves the issue is kernel complexity
    (2x tl.dot + LayerNorm + activation), not the specific tl.erf op.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)

    token_offsets = pid * BLOCK_M + offs_m
    mask_m = token_offsets < T

    # Load input
    x_offsets = token_offsets[:, None] * BLOCK_C + offs_c[None, :]
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x_vals, axis=1) / BLOCK_C
    xc = x_vals - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / BLOCK_C
    inv_std = tl.rsqrt(var + 1e-5)
    ln_w = tl.load(ln_w_ptr + offs_c).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c).to(tl.float32)
    norm = xc * inv_std[:, None] * ln_w[None, :] + ln_b[None, :]
    norm_h = norm.to(tl.float16)

    # fc1 (tl.dot #1)
    w1 = tl.load(fc1_w_ptr + offs_c[:, None] + offs_h[None, :] * BLOCK_C)
    b1 = tl.load(fc1_b_ptr + offs_h).to(tl.float32)
    h1 = tl.dot(norm_h, w1) + b1[None, :]

    # Approximate GELU via tanh (NO tl.erf)
    # gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    h1_f = h1.to(tl.float16).to(tl.float32)
    cube = h1_f * h1_f * h1_f
    inner = 0.7978845608028654 * (h1_f + 0.044715 * cube)  # sqrt(2/pi)
    # tanh via (exp(2x)-1)/(exp(2x)+1) — avoids tl.erf entirely
    exp2x = tl.exp(2.0 * inner)
    tanh_val = (exp2x - 1.0) / (exp2x + 1.0)
    gelu = 0.5 * h1_f * (1.0 + tanh_val)
    gelu_h = gelu.to(tl.float16)

    # fc2 (tl.dot #2)
    w2 = tl.load(fc2_w_ptr + offs_h[:, None] + offs_c[None, :] * BLOCK_H)
    b2 = tl.load(fc2_b_ptr + offs_c).to(tl.float32)
    out = tl.dot(gelu_h, w2) + b2[None, :]

    # Residual + store
    final = out.to(tl.float16) + x_vals.to(tl.float16)
    tl.store(y_ptr + x_offsets, final, mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

def make_inputs(device):
    """Create MLP weight tensors on device."""
    x = torch.randn(M, C, device=device, dtype=torch.float16)
    ln_w = torch.ones(C, device=device, dtype=torch.float16)
    ln_b = torch.zeros(C, device=device, dtype=torch.float16)
    fc1_w = torch.randn(C, H, device=device, dtype=torch.float16) * 0.05
    fc1_b = torch.zeros(H, device=device, dtype=torch.float16)
    fc2_w = torch.randn(H, C, device=device, dtype=torch.float16) * 0.05
    fc2_b = torch.zeros(C, device=device, dtype=torch.float16)
    return x, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b


def run_case_in_subprocess(case_name):
    """Run a single test case in a subprocess to catch SIGSEGV cleanly."""
    script = f"""
import torch, triton, triton.language as tl, sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from triton_xpu_fused_mlp_segfault import (
    make_inputs, _fused_norm_mlp_CRASHES,
    _split_norm_fc1_gelu_PASSES, _split_fc2_residual_PASSES,
    _fused_norm_mlp_approx_gelu_PASSES,
    M, C, H, BLOCK_M,
)
device = 'xpu'
x, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b = make_inputs(device)
y = torch.empty_like(x)
grid = (triton.cdiv(M, BLOCK_M),)

if '{case_name}' == 'fused_erf':
    # CASE 1: expected SIGSEGV
    _fused_norm_mlp_CRASHES[grid](
        x, y, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b,
        M, BLOCK_M=BLOCK_M, BLOCK_C={C}, BLOCK_H={H}, num_warps=4,
    )
    torch.xpu.synchronize()

elif '{case_name}' == 'split':
    # CASE 2: split at GELU boundary
    tmp = torch.empty(M, {H}, device=device, dtype=torch.float16)
    _split_norm_fc1_gelu_PASSES[grid](
        x, tmp, ln_w, ln_b, fc1_w, fc1_b,
        M, BLOCK_M=BLOCK_M, BLOCK_C={C}, BLOCK_H={H}, num_warps=4,
    )
    _split_fc2_residual_PASSES[grid](
        x, tmp, y, fc2_w, fc2_b,
        M, BLOCK_M=BLOCK_M, BLOCK_C={C}, BLOCK_H={H}, num_warps=4,
    )
    torch.xpu.synchronize()

elif '{case_name}' == 'approx_gelu':
    # CASE 3: single kernel, tanh-approx GELU (no tl.erf)
    _fused_norm_mlp_approx_gelu_PASSES[grid](
        x, y, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b,
        M, BLOCK_M=BLOCK_M, BLOCK_C={C}, BLOCK_H={H}, num_warps=4,
    )
    torch.xpu.synchronize()

print('PASS')
"""
    env = {**os.environ, 'PYTHONPATH': os.path.dirname(os.path.abspath(__file__))}
    r = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=120, env=env,
    )
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    print("=" * 70)
    print("Intel XPU Triton SIGSEGV Reproducer: Fused MLP with tl.erf + 2x tl.dot")
    print("=" * 70)
    print()

    if not torch.xpu.is_available():
        print("ERROR: No XPU device available. This reproducer requires Intel XPU.")
        sys.exit(1)

    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Problem size: M={M}, C={C}, H={H}, BLOCK_M={BLOCK_M}")
    print()

    cases = [
        ("fused_erf",    "CASE 1: Monolithic kernel (LN + fc1 + tl.erf + fc2)",
         "CRASHES", "tl.erf + 2x tl.dot in single kernel → SPIR-V compiler SIGSEGV"),
        ("split",        "CASE 2: Split at GELU boundary (two kernels)",
         "PASSES",  "Each kernel has only 1x tl.dot; compiler handles it fine"),
        ("approx_gelu", "CASE 3: Monolithic kernel with tanh-approx GELU (no tl.erf)",
         "CRASHES", "Same 2x tl.dot + complex body → same SIGSEGV (not tl.erf specific)"),
    ]

    results = []
    for case_name, desc, expected, explanation in cases:
        print(f"─── {desc} ───")
        print(f"  Expected: {expected} ({explanation})")
        rc, stdout, stderr = run_case_in_subprocess(case_name)

        if rc == -11:
            status = "SIGSEGV (rc=-11)"
        elif rc == 0 and "PASS" in stdout:
            status = "PASS"
        else:
            status = f"FAIL (rc={rc})"
            if stderr:
                # Print last 3 lines of error
                err_lines = stderr.strip().splitlines()[-3:]
                for line in err_lines:
                    print(f"    {line}")

        match = "✓" if (expected == "CRASHES" and rc == -11) or \
                       (expected == "PASSES" and rc == 0) else "✗ UNEXPECTED"
        print(f"  Actual:   {status}  [{match}]")
        print()
        results.append((case_name, expected, rc))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_match = all(
        (exp == "CRASHES" and rc == -11) or (exp == "PASSES" and rc == 0)
        for _, exp, rc in results
    )
    if all_match:
        print("All cases behaved as expected.")
        print()
        print("The bug: Intel XPU Triton's SPIR-V backend crashes (SIGSEGV) when a")
        print("single kernel has 2+ tl.dot() operations combined with a complex")
        print("arithmetic body (LayerNorm + any activation function).")
        print()
        print("Key insight: It's NOT about tl.erf specifically — tanh-approx also crashes.")
        print("The issue is the total kernel IR complexity (2x matmul + LN + activation).")
        print()
        print("Workaround:")
        print("  Split MLP into two kernels at the activation boundary.")
        print("  Each kernel gets only 1x tl.dot → compiler handles it fine.")
    else:
        print("Some cases did not match expected behavior!")
        print("(The bug may be non-deterministic or already fixed in your driver)")
        for name, exp, rc in results:
            actual = "SIGSEGV" if rc == -11 else "PASS" if rc == 0 else f"rc={rc}"
            print(f"  {name}: expected={exp}, actual={actual}")


if __name__ == "__main__":
    main()
