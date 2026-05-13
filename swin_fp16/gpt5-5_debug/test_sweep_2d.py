"""2D sweep: n_ptrs vs (n_reg + n_cex) with a fixed body that does tl.load."""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="sweep_2d_")

def make_test(n_ptrs, n_reg, n_cex):
    ptr_params = ", ".join(f"p{i}" for i in range(n_ptrs))
    reg_params = ", ".join(f"r{i}" for i in range(n_reg))
    cex_params = ", ".join(f"C{i}: tl.constexpr" for i in range(n_cex))
    all_params = ", ".join(x for x in [ptr_params, reg_params, cex_params] if x)
    body = "    offs = tl.arange(0, 16)\n    x = tl.load(p0 + offs)\n"
    ptr_args = ", ".join(["x"] * n_ptrs)
    reg_args = ", ".join(["1"] * n_reg)
    cex_args = ", ".join(["16"] * n_cex)
    all_args = ", ".join(x for x in [ptr_args, reg_args, cex_args] if x)
    return f"""\
import torch, triton, triton.language as tl
@triton.jit
def _k({all_params}):
{body}
x = torch.randn(16, dtype=torch.float16, device="xpu")
_k[(1,)]({all_args}, num_warps=4, num_stages=3)
torch.xpu.synchronize()
print("OK")
"""

# Sweep: n_ptrs from 1 to 10, n_other (split as reg=1,cex=rest or reg=0,cex=all) from 0 to 8
# Use: n_reg=0, n_cex=n_other  (all constexpr other)
# This tells us if constexpr count is the driver

print("=== All non-ptr params as constexpr (n_reg=0, n_cex=n_other) ===")
print(f"{'ptrs':>4s}", end="")
for n_other in range(9):
    print(f" {n_other:>3d}", end="")
print()

for n_ptrs in range(1, 11):
    print(f"{n_ptrs:>4d}", end="")
    for n_other in range(9):
        n_reg = 0
        n_cex = n_other
        code = make_test(n_ptrs, n_reg, n_cex)
        fname = f"p{n_ptrs}_r{n_reg}_c{n_cex}.py"
        fpath = os.path.join(tmpdir, fname)
        with open(fpath, "w") as f:
            f.write(code)
        r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=60)
        if r.returncode == 0: s = "  ."
        elif r.returncode == -11: s = "  X"
        else: s = f" !{r.returncode}"
        print(s, end="")
    print()

# Also test with n_reg=2 (matching round_010 which has total_tokens, eps)
print()
print("=== With n_reg=2 fixed, varying n_ptrs and n_cex ===")
print(f"{'ptrs':>4s}", end="")
for n_cex in range(9):
    print(f" {n_cex:>3d}", end="")
print("  (n_cex)")

for n_ptrs in range(1, 11):
    n_reg = 2
    print(f"{n_ptrs:>4d}", end="")
    for n_cex in range(9):
        code = make_test(n_ptrs, n_reg, n_cex)
        fname = f"p{n_ptrs}_r2_c{n_cex}.py"
        fpath = os.path.join(tmpdir, fname)
        with open(fpath, "w") as f:
            f.write(code)
        r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=60)
        if r.returncode == 0: s = "  ."
        elif r.returncode == -11: s = "  X"
        else: s = f" !{r.returncode}"
        print(s, end="")
    print()
