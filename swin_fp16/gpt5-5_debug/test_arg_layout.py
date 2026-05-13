"""Deeper investigation: what combination of ptr/reg/constexpr triggers crash."""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="arg_layout_")

def make_test(n_ptrs, n_regular, n_constexpr, do_load=True, cex_vals=None):
    ptr_params = ", ".join(f"p{i}" for i in range(n_ptrs))
    reg_params = ", ".join(f"r{i}" for i in range(n_regular))
    cex_params = ", ".join(f"C{i}: tl.constexpr" for i in range(n_constexpr))
    all_params = ", ".join(x for x in [ptr_params, reg_params, cex_params] if x)
    
    body = "    offs = tl.arange(0, 16)\n    x = tl.load(p0 + offs)\n" if do_load else "    return\n"
    
    ptr_args = ", ".join(["x"] * n_ptrs)
    reg_args = ", ".join(["1"] * n_regular)
    if cex_vals is None:
        cex_vals = [16] * n_constexpr
    cex_args = ", ".join(str(v) for v in cex_vals)
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

configs = [
    # ptr, reg, cex, do_load, cex_vals, description
    # Test: is it about total non-constexpr arg count?
    (4, 0, 3, True, None, "4ptr_0reg_3cex"),
    (4, 1, 3, True, None, "4ptr_1reg_3cex"),
    (4, 2, 3, True, None, "4ptr_2reg_3cex"),
    (4, 3, 3, True, None, "4ptr_3reg_3cex"),
    (4, 4, 3, True, None, "4ptr_4reg_3cex"),
    # Test: is it about constexpr count?
    (4, 2, 0, True, None, "4ptr_2reg_0cex"),
    (4, 2, 1, True, None, "4ptr_2reg_1cex"),
    (4, 2, 2, True, None, "4ptr_2reg_2cex"),
    (4, 2, 3, True, None, "4ptr_2reg_3cex_2"),
    (4, 2, 4, True, None, "4ptr_2reg_4cex"),
    # Test: any regular args at all?
    (4, 0, 0, True, None, "4ptr_0reg_0cex"),
    (4, 1, 0, True, None, "4ptr_1reg_0cex"),
    (4, 2, 0, True, None, "4ptr_2reg_0cex_2"),
    # Test: no ptr but many reg args
    (1, 5, 3, True, None, "1ptr_5reg_3cex"),
    (1, 8, 3, True, None, "1ptr_8reg_3cex"),
    (1, 10, 3, True, None, "1ptr_10reg_3cex"),
    # Test: does it depend on constexpr VALUES?
    (4, 2, 3, True, [16, 32, 128], "4ptr_2reg_3cex_vals16_32_128"),
    (4, 2, 3, True, [32, 32, 32], "4ptr_2reg_3cex_vals32_32_32"),
    # Test: re-run for reliability
    (4, 2, 3, True, [16, 16, 16], "4ptr_2reg_3cex_rerun"),
    (7, 2, 3, True, [16, 16, 16], "7ptr_2reg_3cex_rerun"),
]

for n_ptrs, n_reg, n_cex, do_load, cex_vals, name in configs:
    code = make_test(n_ptrs, n_reg, n_cex, do_load, cex_vals)
    fpath = os.path.join(tmpdir, f"{name}.py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0: s = "PASS"
    elif r.returncode == -11: s = "SEGFAULT"
    else: s = f"FAIL({r.returncode})"
    total_rt = n_ptrs + n_reg  # runtime args (no constexpr)
    total_all = n_ptrs + n_reg + n_cex
    print(f"  [{s:>8s}]  {name:40s}  rt={total_rt} all={total_all}")
