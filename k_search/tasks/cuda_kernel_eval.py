"""CUDA kernel evaluator for CudaKernelTask.

Compiles kernel.h + kernel.cu + main.cpp via torch.utils.cpp_extension,
runs correctness checks and performance measurements against a PyTorch reference.

Outputs a single JSON line to stdout with results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.utils.cpp_extension


def load_reference(ref_path: str, precision: str):
    """Load and instantiate the reference model."""
    ref_dir = str(Path(ref_path).parent)
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)

    spec_globals: dict = {}
    exec(compile(Path(ref_path).read_text(), ref_path, "exec"), spec_globals)

    # Get model class and input generators
    model_cls = spec_globals.get("Model")
    get_inputs_fn = spec_globals.get("get_inputs")
    get_init_inputs_fn = spec_globals.get("get_init_inputs")

    if model_cls is None:
        raise RuntimeError("Reference file must define a 'Model' class")
    if get_inputs_fn is None:
        raise RuntimeError("Reference file must define 'get_inputs()'")
    if get_init_inputs_fn is None:
        raise RuntimeError("Reference file must define 'get_init_inputs()'")

    # Instantiate model
    init_inputs = get_init_inputs_fn()
    model = model_cls(*init_inputs)

    # Move to GPU and set precision
    dtype = _precision_to_dtype(precision)
    model = model.to(dtype=dtype, device="cuda")
    model.eval()

    return model, get_inputs_fn, dtype


def _precision_to_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(precision.lower(), torch.float32)


def compile_cuda_kernel(kernel_dir: str):
    """Compile the CUDA kernel using torch.utils.cpp_extension.load."""
    kernel_h = Path(kernel_dir, "kernel.h").read_text()
    kernel_cu = Path(kernel_dir, "kernel.cu").read_text()
    main_cpp = Path(kernel_dir, "main.cpp").read_text()

    # Write combined sources for load_inline
    # kernel.h is included via the header
    build_dir = str(Path(kernel_dir) / "build")
    os.makedirs(build_dir, exist_ok=True)

    # Write kernel.h so it can be #included
    header_path = Path(build_dir, "kernel.h")
    header_path.write_text(kernel_h)

    # Compile using load_inline with cuda_sources and cpp_sources
    module = torch.utils.cpp_extension.load_inline(
        name="cuda_kernel_module",
        cpp_sources=[main_cpp],
        cuda_sources=[kernel_cu],
        extra_include_paths=[build_dir],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
        build_directory=build_dir,
    )
    return module


def check_correctness(module, model, get_inputs_fn, dtype, num_trials: int, rtol: float, atol: float):
    """Check if kernel output matches reference."""
    for trial in range(num_trials):
        inputs = get_inputs_fn()
        inputs = [x.to(dtype=dtype, device="cuda") if isinstance(x, torch.Tensor) else x for x in inputs]

        # Get reference output
        with torch.no_grad():
            ref_out = model(*inputs)

        # Get kernel output
        kernel_out = module.run(*inputs)

        # Normalize outputs to lists
        if isinstance(ref_out, torch.Tensor):
            ref_out = [ref_out]
        elif isinstance(ref_out, tuple):
            ref_out = list(ref_out)

        if isinstance(kernel_out, torch.Tensor):
            kernel_out = [kernel_out]
        elif isinstance(kernel_out, tuple):
            kernel_out = list(kernel_out)

        if len(ref_out) != len(kernel_out):
            return False, f"Output count mismatch: ref={len(ref_out)}, kernel={len(kernel_out)}"

        for i, (r, k) in enumerate(zip(ref_out, kernel_out)):
            if not isinstance(r, torch.Tensor) or not isinstance(k, torch.Tensor):
                continue
            if r.shape != k.shape:
                return False, f"Shape mismatch at output {i}: ref={r.shape}, kernel={k.shape}"
            if not torch.allclose(r.float(), k.float(), rtol=rtol, atol=atol):
                max_diff = (r.float() - k.float()).abs().max().item()
                return False, f"Values differ at output {i}: max_diff={max_diff:.6f} (rtol={rtol}, atol={atol})"

    return True, ""


def measure_latency(module, get_inputs_fn, dtype, num_trials: int) -> float:
    """Measure kernel latency using CUDA events."""
    inputs = get_inputs_fn()
    inputs = [x.to(dtype=dtype, device="cuda") if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    for _ in range(10):
        module.run(*inputs)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.run(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Return median
    times.sort()
    return times[len(times) // 2]


def measure_ref_latency(model, get_inputs_fn, dtype, num_trials: int) -> float:
    """Measure reference model latency."""
    inputs = get_inputs_fn()
    inputs = [x.to(dtype=dtype, device="cuda") if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(*inputs)
    torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _try_pass_params(module, model):
    """Try to pass reference model weights to the compiled kernel module.

    Strategies:
    1. If module has set_params(), find the first Conv2d in the model and pass its weight/bias.
    2. Store model on __main__ so try_autoload_from_python() in C++ code can find it.
    """
    import __main__ as _main_mod

    # Strategy 2: Make model discoverable by try_autoload_from_python()
    _main_mod.model_instance = model
    # Also try common attribute names the C++ code might scan for
    if not hasattr(_main_mod, "model"):
        _main_mod.model = model

    # Strategy 1: Explicitly call set_params if available
    if hasattr(module, "set_params"):
        # Find first Conv2d layer in reference model
        for m in model.modules():
            if hasattr(m, "weight") and hasattr(m.weight, "dim") and m.weight.dim() == 4:
                w = m.weight.data
                b = m.bias.data if m.bias is not None else torch.empty(0)
                try:
                    module.set_params(w, b)
                except Exception:
                    pass
                return
        # Fallback: try named parameters with 'weight' in the name
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 4:
                # Look for matching bias
                bias_name = name.replace("weight", "bias")
                bias = dict(model.named_parameters()).get(bias_name, torch.empty(0))
                try:
                    module.set_params(param.data, bias.data if hasattr(bias, "data") else bias)
                except Exception:
                    pass
                return


def _run_profile_only(args):
    """Profile-only mode: compile kernel, load ref model, run kernel once for ncu profiling.

    Uses CUDA profiler API markers (cudaProfilerStart/Stop) to delimit the
    region of interest.  ncu is invoked with --profile-from-start off so it
    only captures kernels launched between the markers.  This correctly handles
    LLM-generated code that launches multiple CUDA kernels per module.run().
    """
    model, get_inputs_fn, dtype = load_reference(args.ref_path, args.precision)
    module = compile_cuda_kernel(args.kernel_dir)
    _try_pass_params(module, model)

    # Generate inputs
    inputs = get_inputs_fn()
    inputs = [x.to(dtype=dtype, device="cuda") if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup: allocates GPU resources, triggers any lazy init.
    # NOT profiled because profiling hasn't started yet (--profile-from-start off).
    module.run(*inputs)
    torch.cuda.synchronize()

    # Start profiling region — ncu captures all kernel launches after this.
    torch.cuda.cudart().cudaProfilerStart()

    # Profiled run: ALL kernels launched here are captured by ncu.
    module.run(*inputs)
    torch.cuda.synchronize()

    # Stop profiling region.
    torch.cuda.cudart().cudaProfilerStop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-path", required=True)
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument(
        "--profile-only", action="store_true",
        help="Compile and run kernel once for ncu profiling. No correctness/timing."
    )
    args = parser.parse_args()

    # --profile-only mode: compile, load, run once, exit
    if args.profile_only:
        _run_profile_only(args)
        return

    result = {"compiled": False, "correct": False, "latency_ms": None, "ref_latency_ms": None, "error": ""}

    try:
        # Load reference
        model, get_inputs_fn, dtype = load_reference(args.ref_path, args.precision)

        # Compile kernel
        try:
            module = compile_cuda_kernel(args.kernel_dir)
        except Exception as e:
            result["error"] = f"Compilation error: {e}"
            print(json.dumps(result))
            return

        result["compiled"] = True

        # Pass model weights to kernel module (for kernels that need set_params)
        _try_pass_params(module, model)

        # Check correctness
        correct, error_msg = check_correctness(
            module, model, get_inputs_fn, dtype,
            num_trials=args.num_correct_trials,
            rtol=args.rtol, atol=args.atol,
        )
        if not correct:
            result["error"] = error_msg
            print(json.dumps(result))
            return

        result["correct"] = True

        # Measure performance
        latency_ms = measure_latency(module, get_inputs_fn, dtype, num_trials=args.num_perf_trials)
        ref_latency_ms = measure_ref_latency(model, get_inputs_fn, dtype, num_trials=args.num_perf_trials)

        result["latency_ms"] = latency_ms
        result["ref_latency_ms"] = ref_latency_ms

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    print(json.dumps(result))


if __name__ == "__main__":
    main()
