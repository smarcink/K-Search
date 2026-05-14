"""Generic kernel comparison script.

Compares optimized kernel solutions (CUDA or Triton) against a reference
PyTorch model. Works with any task that follows the K-Search convention:
  - Reference .py file with: Model class, get_inputs(), get_init_inputs()
  - CUDA solutions: kernel.h / kernel.cu / main.cpp (or K-Search artifacts)
  - Triton solutions: model_new.py with ModelNew class (or K-Search artifacts)

Usage:
    # Compare extracted kernel dirs against a reference
    python compare_kernels.py path/to/reference.py path/to/cuda_dir path/to/triton_dir

    # Compare K-Search artifacts directories
    python compare_kernels.py path/to/reference.py path/to/ksearch-artifacts1 path/to/ksearch-artifacts2

    # With options
    python compare_kernels.py path/to/reference.py path/to/kernels \\
        --precision fp16 --compile-mode default --shape 64 1024 1024 \\
        --warmup 100 --iters 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.utils.cpp_extension
from torch import nn

from k_search.utils.device import get_device, get_device_name, synchronize, create_event

SEED = 42


def _copy_weights(dst: nn.Module, src: nn.Module) -> None:
    """Copy matching parameters/buffers from src to dst (non-strict)."""
    src_state = src.state_dict()
    if not src_state:
        return
    missing, unexpected = dst.load_state_dict(src_state, strict=False)
    if missing:
        print(f"    [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")


def load_reference(ref_path: Path, precision: str = "fp16", device: str = "cuda"):
    """Load and instantiate the reference model."""
    ref_dir = str(ref_path.parent)
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)

    spec_globals: dict = {}
    exec(compile(ref_path.read_text(), str(ref_path), "exec"), spec_globals)

    model_cls = spec_globals["Model"]
    get_inputs_fn = spec_globals["get_inputs"]
    get_init_inputs_fn = spec_globals["get_init_inputs"]

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(precision, torch.float16)

    model = model_cls(*get_init_inputs_fn()).to(dtype=dtype, device=device).eval()
    return model, get_inputs_fn, get_init_inputs_fn, dtype


def compile_kernel(kernel_h: str, kernel_cu: str, main_cpp: str, name: str = "kernel") -> Any:
    """Compile CUDA kernel from source strings, return loaded module."""
    tmp_dir = tempfile.mkdtemp(prefix=f"compare_{name}_")
    build_dir = os.path.join(tmp_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Write kernel.h for #include resolution
    Path(tmp_dir, "kernel.h").write_text(kernel_h)
    Path(build_dir, "kernel.h").write_text(kernel_h)

    module = torch.utils.cpp_extension.load_inline(
        name=f"compare_{name}",
        cpp_sources=[main_cpp],
        cuda_sources=[kernel_cu],
        extra_include_paths=[tmp_dir, build_dir],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
        build_directory=build_dir,
    )
    return module


def load_triton_model(code: str, init_inputs: list, dtype, name: str = "triton", device: str = "cuda"):
    """Load a triton/python ModelNew from source code string."""
    tmp_dir = tempfile.mkdtemp(prefix=f"compare_triton_{name}_")
    model_path = Path(tmp_dir) / "model_new.py"
    model_path.write_text(code)

    mod_globals: dict = {}
    exec(compile(code, str(model_path), "exec"), mod_globals)

    model_cls = mod_globals["ModelNew"]
    model = model_cls(*init_inputs).to(dtype=dtype, device=device).eval()
    return model


def collect_kernels(paths: list[str]) -> list[dict]:
    """Collect kernel sources from artifacts dirs or direct kernel dirs.

    Returns list of dicts with either:
      - type='cuda', kernel_h, kernel_cu, main_cpp
      - type='triton', model_new_code
    """
    kernels = []

    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"  [warn] path not found: {path}", file=sys.stderr)
            continue

        # Case 1a: directory with kernel.h/kernel.cu/main.cpp directly
        if (path / "kernel.cu").exists() and (path / "main.cpp").exists():
            kernels.append({
                "type": "cuda",
                "name": path.name,
                "kernel_h": (path / "kernel.h").read_text() if (path / "kernel.h").exists() else "",
                "kernel_cu": (path / "kernel.cu").read_text(),
                "main_cpp": (path / "main.cpp").read_text(),
                "source": str(path),
            })
            continue

        # Case 1b: directory with model_new.py directly
        if (path / "model_new.py").exists():
            kernels.append({
                "type": "triton",
                "name": path.name,
                "model_new_code": (path / "model_new.py").read_text(),
                "source": str(path),
            })
            continue

        # Case 2: K-Search artifacts directory — scan for solution JSONs
        for sol_file in sorted(path.rglob("*.json")):
            if sol_file.name == "solution.json" or "eval_report" in sol_file.name:
                continue
            try:
                sol = json.loads(sol_file.read_text())
                sources = sol.get("sources", [])
                if not isinstance(sources, list):
                    continue
                src_map = {s["path"]: s["content"] for s in sources if isinstance(s, dict)}
                if "kernel.cu" in src_map and "main.cpp" in src_map:
                    kernels.append({
                        "type": "cuda",
                        "name": sol.get("name", sol_file.stem)[:60],
                        "kernel_h": src_map.get("kernel.h", ""),
                        "kernel_cu": src_map["kernel.cu"],
                        "main_cpp": src_map["main.cpp"],
                        "source": str(sol_file),
                    })
                elif "model_new.py" in src_map:
                    kernels.append({
                        "type": "triton",
                        "name": sol.get("name", sol_file.stem)[:60],
                        "model_new_code": src_map["model_new.py"],
                        "source": str(sol_file),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

    return kernels


@torch.inference_mode()
def benchmark_model(model, inputs: list[torch.Tensor], warmup: int, iters: int, device: str = "cuda") -> tuple[float, torch.Tensor]:
    """Benchmark a model (reference or triton ModelNew), return (ms_per_iter, output)."""
    for _ in range(warmup):
        out = model(*inputs)
    synchronize(device)

    start = create_event(device)
    end = create_event(device)
    start.record()
    for _ in range(iters):
        out = model(*inputs)
    end.record()
    synchronize(device)
    return start.elapsed_time(end) / iters, out


@torch.inference_mode()
def benchmark_cuda_module(module, inputs: list[torch.Tensor], warmup: int, iters: int, device: str = "cuda") -> tuple[float, torch.Tensor]:
    """Benchmark a compiled CUDA module (with .run()), return (ms_per_iter, output)."""
    for _ in range(warmup):
        out = module.run(*inputs)
    synchronize(device)

    start = create_event(device)
    end = create_event(device)
    start.record()
    for _ in range(iters):
        out = module.run(*inputs)
    end.record()
    synchronize(device)

    if isinstance(out, (list, tuple)):
        out = out[0] if len(out) == 1 else torch.cat([o.flatten() for o in out])
    return start.elapsed_time(end) / iters, out


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Compute max_abs_error, mse, cosine_similarity."""
    a32 = a.detach().float().flatten()
    b32 = b.detach().float().flatten()
    abs_err = (a32 - b32).abs()
    max_abs = abs_err.max().item()
    mse = abs_err.pow(2).mean().item()
    denom = a32.norm() * b32.norm()
    cos_sim = (torch.dot(a32, b32) / denom.clamp_min(1e-12)).item() if denom > 0 else 1.0
    return max_abs, mse, cos_sim


def _try_set_params(module, ref_model: nn.Module) -> None:
    """Pass reference model's conv weights/bias to a CUDA module via set_params()."""
    for m in ref_model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            b = m.bias.data if m.bias is not None else torch.Tensor()
            module.set_params(w, b)
            return


def main():
    parser = argparse.ArgumentParser(
        description="Compare optimized kernels against a reference model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s ref.py cuda_dir/ triton_dir/
  %(prog)s ref.py ksearch-artifacts/ --compile-mode default
  %(prog)s ref.py dir1/ dir2/ --shape 64 1024 1024 --precision fp16
""")
    parser.add_argument("ref", help="Reference .py file (must define Model, get_inputs, get_init_inputs)")
    parser.add_argument("paths", nargs="+", help="Kernel dirs or K-Search artifacts dirs to compare")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for PASS/FAIL")
    parser.add_argument("--compile-mode", default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Apply torch.compile to the reference model with this mode")
    parser.add_argument("--shape", type=int, nargs="+", default=None,
                        help="Override input tensor shape (e.g. --shape 64 1024 1024)")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g. cuda:0, xpu:0). Auto-detected if omitted.")
    args = parser.parse_args()

    device = get_device(args.device)
    backend = device.split(":")[0]
    if backend not in ("cuda", "xpu"):
        sys.exit("CUDA or XPU GPU is required.")

    torch.manual_seed(SEED)

    print(f"Device: {device}  GPU: {get_device_name(device)}")
    print(f"Precision: {args.precision}  Warmup: {args.warmup}  Iters: {args.iters}")
    if args.compile_mode:
        print(f"torch.compile mode: {args.compile_mode}")
    print()

    # Load reference
    ref_path = Path(args.ref).resolve()
    if not ref_path.exists():
        sys.exit(f"Reference file not found: {ref_path}")
    print(f"Reference: {ref_path.name}")
    model, get_inputs_fn, get_init_inputs_fn, dtype = load_reference(ref_path, args.precision, device=device)

    # Get inputs
    if args.shape:
        orig_inputs = get_inputs_fn()
        num_tensors = sum(1 for x in orig_inputs if isinstance(x, torch.Tensor))
        inputs = [torch.randn(args.shape, dtype=dtype, device=device) for _ in range(num_tensors)]
    else:
        inputs = get_inputs_fn()
        inputs = [x.to(dtype=dtype, device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
    print(f"Input shapes: {[tuple(t.shape) for t in inputs if isinstance(t, torch.Tensor)]}")

    # Compute reference output from EAGER model (for correctness comparison)
    with torch.inference_mode():
        ref_out = model(*inputs)
    synchronize(device)

    # Optionally compile reference for timing (compile can change numerics via tf32 etc.)
    ref_model_for_timing = model
    if args.compile_mode:
        try:
            compiled = torch.compile(model, mode=args.compile_mode)
            # Warm-up to trigger compilation and catch errors early
            with torch.inference_mode():
                compiled(*inputs)
            synchronize(device)
            ref_model_for_timing = compiled
        except Exception as e:
            print(f"  [warn] torch.compile failed ({type(e).__name__}), using eager reference.")

    # Benchmark reference
    ref_ms, _ = benchmark_model(ref_model_for_timing, inputs, args.warmup, args.iters, device=device)
    print(f"Reference latency: {ref_ms:.4f} ms\n")

    # Collect kernels
    kernels = collect_kernels(args.paths)
    if not kernels:
        sys.exit("No kernel solutions found in the provided paths.")

    print(f"Found {len(kernels)} kernel(s) to compare\n")

    # Get init_inputs for triton models
    init_inputs = get_init_inputs_fn()

    # Collect model parameters for CUDA kernels that may need them
    model_params = [p.data for p in model.parameters()]

    # Benchmark each kernel
    rows: list[tuple[str, float, float, float, float, str]] = []
    for i, kernel in enumerate(kernels):
        name = kernel["name"]
        ktype = kernel.get("type", "cuda")
        print(f"  [{i+1}/{len(kernels)}] [{ktype}] Compiling: {name}...", end=" ", flush=True)
        try:
            if ktype == "cuda":
                if backend == "xpu":
                    raise RuntimeError("CUDA kernel compilation not supported on XPU device")
                module = compile_kernel(
                    kernel["kernel_h"], kernel["kernel_cu"], kernel["main_cpp"],
                    name=f"k{i}",
                )
                # If the module exposes set_params(), pass reference model weights
                if hasattr(module, "set_params"):
                    _try_set_params(module, model)
                # Try run with just inputs first; if it fails, try with model params appended
                try:
                    ms, out = benchmark_cuda_module(module, inputs, args.warmup, args.iters, device=device)
                except TypeError:
                    ms, out = benchmark_cuda_module(module, inputs + model_params, args.warmup, args.iters, device=device)
            else:  # triton
                triton_model = load_triton_model(
                    kernel["model_new_code"], init_inputs, dtype, name=f"k{i}", device=device
                )
                _copy_weights(triton_model, model)
                ms, out = benchmark_model(triton_model, inputs, args.warmup, args.iters, device=device)
            max_abs, mse, cos_sim = diff(out, ref_out)
            ok = max_abs <= args.atol
            status = "PASS" if ok else "FAIL"
            rows.append((name, ms, max_abs, mse, cos_sim, status))
            print(f"{ms:.4f}ms  speedup={ref_ms/ms:.2f}x  {status}")
        except Exception as e:
            err_msg = str(e).split("\n")[0][:80]
            rows.append((name, float("nan"), float("nan"), float("nan"), float("nan"), "ERROR"))
            print(f"ERROR: {err_msg}")

    # Summary table
    ref_label = f"[reference] {ref_path.name}"
    name_w = max(len(ref_label), *(len(r[0]) for r in rows)) if rows else len(ref_label)
    name_w = min(name_w, 60)
    print(f"\n{'='*100}")
    print(f"{'impl':<{name_w}}  {'ms/iter':>10}  {'speedup':>8}  {'max_abs':>11}  {'mse':>11}  {'cos_sim':>10}  status")
    print(f"{'-'*100}")

    # Reference row
    print(f"{ref_label:<{name_w}}  {ref_ms:>10.4f}  {'1.00x':>8}  {'0.000e+00':>11}  {'0.000e+00':>11}  {'1.000000':>10}  REF")

    for name, ms, ma, mse_v, cs, status in rows:
        display_name = name[:name_w]
        if ms != ms or ms <= 0:  # nan check
            print(f"{display_name:<{name_w}}  {'N/A':>10}  {'-':>8}  {'N/A':>11}  {'N/A':>11}  {'N/A':>10}  {status}")
        else:
            speed = f"{ref_ms / ms:.2f}x"
            print(f"{display_name:<{name_w}}  {ms:>10.4f}  {speed:>8}  {ma:>11.3e}  {mse_v:>11.3e}  {cs:>10.6f}  {status}")

    print(f"{'='*100}")
    print(f"(reference latency = {ref_ms:.4f} ms)")


if __name__ == "__main__":
    main()
