"""Find the pointer arg count threshold for SIGSEGV."""
import subprocess, sys, os, tempfile
PYTHON = sys.executable
tmpdir = tempfile.mkdtemp(prefix="ptr_count_")

def make_test(n_ptrs, n_regular, n_constexpr, do_load=True, num_warps=4, num_stages=3):
    ptr_params = ", ".join(f"p{i}" for i in range(n_ptrs))
    reg_params = ", ".join(f"r{i}" for i in range(n_regular))
    cex_params = ", ".join(f"C{i}: tl.constexpr" for i in range(n_constexpr))
    all_params = ", ".join(x for x in [ptr_params, reg_params, cex_params] if x)
    
    body = ""
    if do_load:
        body = """\
    offs = tl.arange(0, 16)
    x = tl.load(p0 + offs)
"""
    else:
        body = "    return\n"
    
    ptr_args = ", ".join(["x"] * n_ptrs)
    reg_args = ", ".join(["1"] * n_regular)
    cex_args = ", ".join(["16"] * n_constexpr)
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

configs = [
    # (name, n_ptrs, n_regular, n_constexpr, do_load, num_warps, num_stages)
    ("2ptr_0reg_0cex_load", 2, 0, 0, True, 4, 3),
    ("3ptr_0reg_0cex_load", 3, 0, 0, True, 4, 3),
    ("4ptr_0reg_0cex_load", 4, 0, 0, True, 4, 3),
    ("5ptr_0reg_0cex_load", 5, 0, 0, True, 4, 3),
    ("6ptr_0reg_0cex_load", 6, 0, 0, True, 4, 3),
    ("7ptr_0reg_0cex_load", 7, 0, 0, True, 4, 3),
    ("8ptr_0reg_0cex_load", 8, 0, 0, True, 4, 3),
    ("8ptr_0reg_0cex_empty", 8, 0, 0, False, 4, 3),
    # Now with regular+constexpr args
    ("2ptr_2reg_3cex_load", 2, 2, 3, True, 4, 3),
    ("4ptr_2reg_3cex_load", 4, 2, 3, True, 4, 3),
    ("6ptr_2reg_3cex_load", 6, 2, 3, True, 4, 3),
    ("8ptr_2reg_3cex_load", 8, 2, 3, True, 4, 3),
    ("8ptr_2reg_3cex_empty", 8, 2, 3, False, 4, 3),
    # Smaller warps/stages
    ("8ptr_2reg_3cex_load_w1_s1", 8, 2, 3, True, 1, 1),
    # Extra tests - 5, 6, 7 with reg+constexpr
    ("5ptr_2reg_3cex_load", 5, 2, 3, True, 4, 3),
    ("7ptr_2reg_3cex_load", 7, 2, 3, True, 4, 3),
]

for name, n_ptrs, n_reg, n_cex, do_load, nw, ns in configs:
    code = make_test(n_ptrs, n_reg, n_cex, do_load, nw, ns)
    fpath = os.path.join(tmpdir, f"{name}.py")
    with open(fpath, "w") as f:
        f.write(code)
    r = subprocess.run([PYTHON, fpath], capture_output=True, text=True, timeout=120)
    if r.returncode == 0: s = "PASS"
    elif r.returncode == -11: s = "SEGFAULT"
    else: s = f"FAIL({r.returncode})"
    print(f"  [{s:>8s}]  {name}")
