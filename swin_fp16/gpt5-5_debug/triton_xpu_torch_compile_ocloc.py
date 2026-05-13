"""
Intel XPU torch.compile Failure Reproducer: ocloc / IGC Version Mismatch
========================================================================

Bug: torch.compile() fails on Intel XPU because PyTorch Inductor invokes
     `ocloc` to compile Triton-generated SPIR-V, but `ocloc` crashes with
     "Incompatible interface in IGC" error code 250.

Root cause: Version mismatch between installed packages:
  - intel-ocloc 26.18.038308 (newer)
  - libigc2 2.30.3-1249 (from 26.09 stack, older)
  - intel-opencl-icd 26.09.37435 (older)

  ocloc 26.18 expects IGC interfaces (IGC_OCL_TRAC) that don't exist in
  the older libigc2, so ALL ocloc compilations fail regardless of input.

  Note: Hand-written @triton.jit kernels work fine because they go through
  the Level Zero / ze_loader path, which uses libze-intel-gpu (not ocloc).
  Only torch.compile (Inductor) uses ocloc for its Triton kernel compilation.

Fix: Either align package versions, or bypass ocloc entirely:
  export TORCHINDUCTOR_XPU_KERNEL_FORMAT=spv

  This tells Inductor to emit SPIR-V instead of zebin, skipping ocloc
  and using Level Zero JIT compilation (same path @triton.jit uses).

Hardware: Intel Arc Pro B70 (BMG / Xe2)
To run:   python triton_xpu_torch_compile_ocloc.py
"""

import subprocess
import sys
import shutil
import struct
import tempfile
import os


def get_package_version(pkg):
    """Get installed dpkg package version."""
    try:
        r = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", pkg],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def test_ocloc_directly():
    """Test ocloc with a minimal valid SPIR-V binary (no PyTorch needed)."""
    # Minimal valid SPIR-V: header + OpCapability Kernel + OpMemoryModel
    header = struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)
    cap = struct.pack("<2I", (2 << 16) | 17, 6)       # OpCapability Kernel
    mem = struct.pack("<3I", (3 << 16) | 14, 0, 2)    # OpMemoryModel Logical OpenCL
    spv_bytes = header + cap + mem

    tmpf = tempfile.NamedTemporaryFile(suffix=".spv", delete=False)
    tmpf.write(spv_bytes)
    tmpf.close()
    out_path = tmpf.name + ".o"

    try:
        r = subprocess.run(
            ["ocloc", "compile", "-file", tmpf.name, "-o", out_path,
             "-spirv_input", "-device", "bmg"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode, r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except FileNotFoundError:
        return -1, "ocloc not found"
    finally:
        for p in (tmpf.name, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def test_torch_compile():
    """Test torch.compile with a trivial model (subprocess to catch crashes)."""
    script = """\
import torch

device = "xpu"
model = torch.nn.LayerNorm(32).half().to(device)
x = torch.randn(4, 32, dtype=torch.float16, device=device)

compiled = torch.compile(model)
with torch.no_grad():
    out = compiled(x)
print("PASS")
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0 and "PASS" in r.stdout:
            return "PASS", ""
        # Extract the key error
        stderr_lines = r.stderr.strip().splitlines()
        for line in stderr_lines:
            if "Incompatible interface" in line or "ocloc" in line:
                return f"FAIL(rc={r.returncode})", line.strip()
        last = stderr_lines[-1] if stderr_lines else ""
        return f"FAIL(rc={r.returncode})", last[:120]
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""


def test_torch_compile_spv():
    """Test torch.compile with TORCHINDUCTOR_XPU_KERNEL_FORMAT=spv (bypasses ocloc)."""
    script = """\
import torch

device = "xpu"
model = torch.nn.LayerNorm(32).half().to(device)
x = torch.randn(4, 32, dtype=torch.float16, device=device)

compiled = torch.compile(model)
with torch.no_grad():
    out = compiled(x)

# Verify correctness
with torch.no_grad():
    ref = model(x)
diff = (out - ref).abs().max().item()
print(f"PASS diff={diff:.2e}")
"""
    env = {**os.environ, "TORCHINDUCTOR_XPU_KERNEL_FORMAT": "spv"}
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if r.returncode == 0 and "PASS" in r.stdout:
            return "PASS", r.stdout.strip().split("PASS ")[-1]
        last = r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ""
        return f"FAIL(rc={r.returncode})", last
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""


def test_triton_direct():
    """Test a hand-written @triton.jit kernel (uses Level Zero, not ocloc)."""
    script = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _add_one(x_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, x + 1.0)

x = torch.zeros(16, device="xpu", dtype=torch.float16)
_add_one[(1,)](x, N=16)
torch.xpu.synchronize()
assert x.sum().item() == 16.0, f"Expected 16, got {x.sum().item()}"
print("PASS")
"""
    # @triton.jit needs source from a real file, not -c
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="triton_test_",
    )
    tmp.write(script)
    tmp.close()
    try:
        r = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and "PASS" in r.stdout:
            return "PASS", ""
        last = r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else ""
        return f"FAIL(rc={r.returncode})", last
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    finally:
        os.unlink(tmp.name)


def main():
    print("=" * 72)
    print("Intel XPU torch.compile reproducer: ocloc / IGC version mismatch")
    print("=" * 72)
    print()

    # Package versions
    packages = [
        ("intel-ocloc", "ocloc offline compiler"),
        ("libigc2", "Intel Graphics Compiler core lib"),
        ("intel-opencl-icd", "OpenCL ICD (compute runtime)"),
        ("libze-intel-gpu1", "Level Zero GPU driver"),
    ]
    print("Installed package versions:")
    versions = {}
    for pkg, desc in packages:
        ver = get_package_version(pkg)
        versions[pkg] = ver
        print(f"  {pkg:25s} {ver or 'NOT INSTALLED':40s}  ({desc})")
    print()

    # Check for version mismatch
    ocloc_ver = versions.get("intel-ocloc", "")
    igc_ver = versions.get("libigc2", "")
    if ocloc_ver and igc_ver:
        # Extract major release (26.XX)
        ocloc_major = ".".join(ocloc_ver.split(".")[:2]) if ocloc_ver else "?"
        # IGC uses different versioning; check if from same release
        print(f"  ocloc release: {ocloc_major}")
        ocl_ver = versions.get("intel-opencl-icd", "")
        ocl_major = ".".join(ocl_ver.split(".")[:2]) if ocl_ver else "?"
        print(f"  opencl-icd release: {ocl_major}")
        if ocloc_major != ocl_major:
            print(f"  *** VERSION MISMATCH: ocloc={ocloc_major} vs opencl-icd={ocl_major} ***")
        print()

    # Test 1: ocloc directly
    print("--- TEST 1: ocloc compile (minimal SPIR-V) ---")
    rc, err = test_ocloc_directly()
    if rc == 0:
        print(f"  PASS — ocloc works correctly")
    elif rc == 250:
        print(f"  FAIL (exit code 250) — IGC initialization failure")
        err_lines = err.splitlines()
        for line in err_lines:
            if "Incompatible" in line or "Error" in line:
                print(f"  {line}")
        print(f"  This means ocloc cannot compile ANY SPIR-V input.")
    else:
        print(f"  FAIL (exit code {rc})")
        print(f"  {err[:200]}")
    print()

    # Test 2: @triton.jit kernel (Level Zero path)
    print("--- TEST 2: @triton.jit kernel (Level Zero path, no ocloc) ---")
    status, err = test_triton_direct()
    print(f"  {status}")
    if err:
        print(f"  {err}")
    print()

    # Test 3: torch.compile (Inductor → ocloc path)
    print("--- TEST 3: torch.compile(nn.LayerNorm) (Inductor → ocloc path) ---")
    status, err = test_torch_compile()
    print(f"  {status}")
    if err:
        print(f"  {err}")
    print()

    # Test 4: torch.compile with SPIR-V format (bypasses ocloc)
    print("--- TEST 4: torch.compile + TORCHINDUCTOR_XPU_KERNEL_FORMAT=spv (fix) ---")
    status, detail = test_torch_compile_spv()
    print(f"  {status}")
    if detail:
        print(f"  {detail}")
    print()

    # Summary
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if rc == 250:
        print("ocloc 26.18 is incompatible with libigc2 from the 26.09 stack.")
        print("All torch.compile() calls fail because Inductor uses ocloc.")
        print("Hand-written @triton.jit kernels work because they use Level Zero.")
        print()
        print("Fix (no package changes needed):")
        print("  export TORCHINDUCTOR_XPU_KERNEL_FORMAT=spv")
        print()
        print("This tells Inductor to emit SPIR-V instead of zebin, bypassing")
        print("ocloc and using Level Zero JIT (same path @triton.jit uses).")
    else:
        print("ocloc appears to work. torch.compile may function correctly.")


if __name__ == "__main__":
    main()
