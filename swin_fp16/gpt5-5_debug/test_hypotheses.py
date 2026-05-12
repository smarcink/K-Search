"""Bisect the SIGSEGV in _swin_ln_qkv_kernel.

Each test runs in a subprocess so a crash in one doesn't kill the rest.
"""
import subprocess, sys, textwrap

PYTHON = sys.executable

TESTS = {
    # ---------- baseline: reproduces the crash ----------
    "baseline (stage0, num_warps=1, big grid, constexpr)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _empty_kernel(
            x_ptr, out_ptr,
            H: tl.constexpr, W: tl.constexpr,
            NH: tl.constexpr, NW: tl.constexpr,
        ):
            pid = tl.program_id(0)
            return

        dev = "xpu"
        x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        grid = (1 * 180 * 320,)  # 57600
        _empty_kernel[grid](x, out, 720, 1280, 180, 320, num_warps=1)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 1: num_warps=2 ----------
    "num_warps=2 (stage0, big grid, constexpr)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _empty_kernel(
            x_ptr, out_ptr,
            H: tl.constexpr, W: tl.constexpr,
            NH: tl.constexpr, NW: tl.constexpr,
        ):
            pid = tl.program_id(0)
            return

        dev = "xpu"
        x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        grid = (1 * 180 * 320,)
        _empty_kernel[grid](x, out, 720, 1280, 180, 320, num_warps=2)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 2: small grid (1 window) ----------
    "small grid=1 (stage0, num_warps=1, constexpr)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _empty_kernel(
            x_ptr, out_ptr,
            H: tl.constexpr, W: tl.constexpr,
            NH: tl.constexpr, NW: tl.constexpr,
        ):
            pid = tl.program_id(0)
            return

        dev = "xpu"
        x = torch.randn(1, 4, 4, 32, dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        grid = (1,)
        _empty_kernel[grid](x, out, 4, 4, 1, 1, num_warps=1)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 3: no constexpr (regular args) ----------
    "no constexpr (stage0, num_warps=1, big grid)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _empty_kernel(
            x_ptr, out_ptr,
            H, W, NH, NW,
        ):
            pid = tl.program_id(0)
            return

        dev = "xpu"
        x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        grid = (1 * 180 * 320,)
        _empty_kernel[grid](x, out, 720, 1280, 180, 320, num_warps=1)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 4: num_warps=2 + no constexpr ----------
    "num_warps=2 + no constexpr (stage0, big grid)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _empty_kernel(
            x_ptr, out_ptr,
            H, W, NH, NW,
        ):
            pid = tl.program_id(0)
            return

        dev = "xpu"
        x = torch.randn(1, 720, 1280, 32, dtype=torch.float16, device=dev)
        out = torch.empty_like(x)
        grid = (1 * 180 * 320,)
        _empty_kernel[grid](x, out, 720, 1280, 180, 320, num_warps=2)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 5: truly minimal kernel (no args except pointers) ----------
    "minimal kernel (no extra args, grid=1, num_warps=1)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _noop_kernel(x_ptr):
            return

        dev = "xpu"
        x = torch.randn(16, dtype=torch.float16, device=dev)
        _noop_kernel[(1,)](x, num_warps=1)
        torch.xpu.synchronize()
        print("OK")
    """),

    # ---------- hypothesis 6: big grid + minimal kernel ----------
    "minimal kernel (no extra args, grid=57600, num_warps=1)": textwrap.dedent("""\
        import torch, triton, triton.language as tl

        @triton.jit
        def _noop_kernel(x_ptr):
            return

        dev = "xpu"
        x = torch.randn(16, dtype=torch.float16, device=dev)
        _noop_kernel[(57600,)](x, num_warps=1)
        torch.xpu.synchronize()
        print("OK")
    """),
}

if __name__ == "__main__":
    print(f"Python: {PYTHON}\n")
    for name, code in TESTS.items():
        print(f"--- {name} ---", flush=True)
        result = subprocess.run(
            [PYTHON, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0:
            print(f"  PASS  (exit 0) {stdout}")
        elif result.returncode == -11:
            print(f"  SEGFAULT (exit -11 / SIGSEGV)")
        else:
            print(f"  FAIL  (exit {result.returncode})")
        if stderr:
            # Show last 3 lines of stderr for context
            lines = stderr.splitlines()
            for line in lines[-3:]:
                print(f"  stderr: {line}")
        print(flush=True)
