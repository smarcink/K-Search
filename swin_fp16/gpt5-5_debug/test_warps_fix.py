"""Test if num_warps=1/num_stages=1 fixes known crash combos."""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="warps_fix_")

def make_test(n_ptrs, n_reg, n_cex, num_warps, num_stages):
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
_k[(1,)]({all_args}, num_warps={num_warps}, num_stages={num_stages})
torch.xpu.synchronize()
print("OK")
"""

# Known crash combos from the 2D sweep
crash_combos = [
    (4, 0, 3),  # 4ptr + 3cex
    (4, 2, 3),  # 4ptr + 2reg + 3cex (round_010 shape)
    (6, 0, 3),  # 6ptr + 3cex
    (6, 2, 3),  # 6ptr + 2reg + 3cex
    (8, 0, 2),  # 8ptr + 2cex
    (3, 0, 1),  # 3ptr + 1cex
]

warp_configs = [(4, 3), (4, 1), (1, 3), (1, 1)]

print(f"{'combo':>20s}", end="")
for nw, ns in warp_configs:
    print(f"  w{nw}s{ns}", end="")
print()

for n_ptrs, n_reg, n_cex in crash_combos:
    label = f"{n_ptrs}p{n_reg}r{n_cex}c"
    print(f"{label:>20s}", end="")
    for nw, ns in warp_configs:
        code = make_test(n_ptrs, n_reg, n_cex, nw, ns)
        fpath = os.path.join(tmpdir, f"{label}_w{nw}s{ns}.py")
        with open(fpath, "w") as f:
            f.write(code)
        r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=60)
        if r.returncode == 0: s = "    ."
        elif r.returncode == -11: s = "    X"
        else: s = f"  !{r.returncode}"
        print(s, end="")
    print()
