"""Standalone XPU evaluator script for K-Search.

Evaluates a ModelNew implementation against a reference Model on Intel XPU.
Called as a subprocess by XpuBenchTask for isolation.

Usage:
    python -m k_search.tasks.xpu_bench.run_and_eval \
        --ref-path path/to/reference.py \
        --kernel-src-path path/to/solution.py \
        --device xpu:0 \
        --precision fp16 \
        --num-correct-trials 5 \
        --num-perf-trials 100
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Device helpers (inlined to keep this script self-contained in subprocess)
# ---------------------------------------------------------------------------

def _synchronize(device: str) -> None:
    backend = device.split(":")[0]
    if backend == "xpu":
        torch.xpu.synchronize()
    elif backend == "cuda":
        torch.cuda.synchronize()


def _create_event(device: str, enable_timing: bool = True):
    backend = device.split(":")[0]
    if backend == "xpu":
        return torch.xpu.Event(enable_timing=enable_timing)
    if backend == "cuda":
        return torch.cuda.Event(enable_timing=enable_timing)
    raise ValueError(f"Timing events not supported on device: {device}")


def _get_device_name(device: str) -> str:
    backend = device.split(":")[0]
    idx = int(device.split(":")[1]) if ":" in device else 0
    if backend == "xpu":
        return torch.xpu.get_device_name(idx)
    if backend == "cuda":
        return torch.cuda.get_device_name(idx)
    return "cpu"


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_module_from_file(path: str, module_name: str):
    """Load a Python module from a file path."""
    abs_path = os.path.abspath(path)
    parent_dir = os.path.dirname(abs_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _copy_weights(dst: torch.nn.Module, src: torch.nn.Module, dst_mod: Any | None = None) -> None:
    """Copy reference parameters/buffers into the candidate model.

    Candidate modules may expose state_dict_remap(src_state) when their module
    layout differs from the reference implementation.
    """
    src_state = src.state_dict()
    if not src_state:
        return

    remap = getattr(dst_mod, "state_dict_remap", None) if dst_mod is not None else None
    if callable(remap):
        src_state = remap(src_state)

    dst_state = dst.state_dict()
    filtered_state = {}
    skipped_shape = []
    for key, value in src_state.items():
        dst_value = dst_state.get(key)
        if dst_value is None:
            continue
        if tuple(dst_value.shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(dst_value.shape)))
            continue
        filtered_state[key] = value

    missing, unexpected = dst.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when copying weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when copying weights: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if skipped_shape:
        preview = ", ".join(f"{k}: {src_shape}->{dst_shape}" for k, src_shape, dst_shape in skipped_shape[:5])
        suffix = "..." if len(skipped_shape) > 5 else ""
        print(f"[WARN] Skipped shape-mismatched weights: {preview}{suffix}")


def _get_precision_dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _get_tolerance(precision: str) -> tuple[float, float]:
    """Return (rtol, atol) appropriate for the precision."""
    if precision == "fp32":
        return (1e-5, 1e-5)
    return (1e-2, 1e-2)  # fp16 / bf16


# ---------------------------------------------------------------------------
# Device patching for inputs
# ---------------------------------------------------------------------------

def _move_inputs_to_device(inputs: list, device: str, dtype: torch.dtype | None = None) -> list:
    """Move a list of tensors (or nested structures) to the target device."""
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            t = x.to(device=device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype=dtype)
            result.append(t)
        else:
            result.append(x)
    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _check_correctness(
    ref_model: torch.nn.Module,
    new_model: torch.nn.Module,
    get_inputs_fn,
    device: str,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    num_trials: int,
) -> tuple[bool, str]:
    """Run correctness comparison over num_trials.

    Returns:
        (passed, message)
    """
    for trial in range(num_trials):
        try:
            inputs = get_inputs_fn()
            inputs = _move_inputs_to_device(inputs, device, dtype)

            with torch.no_grad():
                ref_out = ref_model(*inputs)
                new_out = new_model(*inputs)

            if isinstance(ref_out, torch.Tensor):
                ref_out = [ref_out]
                new_out = [new_out]
            elif not isinstance(ref_out, (list, tuple)):
                ref_out = [ref_out]
                new_out = [new_out]

            for i, (ro, no) in enumerate(zip(ref_out, new_out)):
                if not isinstance(ro, torch.Tensor) or not isinstance(no, torch.Tensor):
                    continue
                if ro.shape != no.shape:
                    return False, f"Trial {trial}: output {i} shape mismatch {ro.shape} vs {no.shape}"
                if not torch.allclose(ro, no, rtol=rtol, atol=atol):
                    max_diff = (ro - no).abs().max().item()
                    return False, f"Trial {trial}: output {i} mismatch, max diff={max_diff:.6e}"
        except Exception as e:
            return False, f"Trial {trial}: runtime error: {e}\n{traceback.format_exc()}"

    return True, f"Correctness PASSED ({num_trials} trials)"


def _measure_latency(
    model: torch.nn.Module,
    get_inputs_fn,
    device: str,
    dtype: torch.dtype,
    num_warmup: int,
    num_trials: int,
) -> float:
    """Measure mean latency in ms using device events."""
    # Warmup
    for _ in range(num_warmup):
        inputs = get_inputs_fn()
        inputs = _move_inputs_to_device(inputs, device, dtype)
        with torch.no_grad():
            model(*inputs)
    _synchronize(device)

    # Timed runs
    times: list[float] = []
    for _ in range(num_trials):
        inputs = get_inputs_fn()
        inputs = _move_inputs_to_device(inputs, device, dtype)

        start = _create_event(device)
        end = _create_event(device)

        _synchronize(device)
        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()
        _synchronize(device)
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------

def evaluate(
    ref_path: str,
    kernel_src_path: str,
    device: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    num_warmup: int = 10,
) -> None:
    """Run full evaluation and print structured results to stdout."""

    dtype = _get_precision_dtype(precision)
    rtol, atol = _get_tolerance(precision)

    print(f"[INFO] Device: {device} ({_get_device_name(device)})")
    print(f"[INFO] Precision: {precision} (dtype={dtype})")
    print(f"[INFO] Correctness trials: {num_correct_trials}")
    print(f"[INFO] Performance trials: {num_perf_trials}")
    print("=" * 60)

    # ---- Load reference module ----
    print("[INFO] Loading reference module...")
    ref_mod = _load_module_from_file(ref_path, "_xpu_bench_ref")

    if not hasattr(ref_mod, "Model"):
        raise RuntimeError(f"Reference file {ref_path} must define a 'Model' class")
    if not hasattr(ref_mod, "get_inputs"):
        raise RuntimeError(f"Reference file {ref_path} must define a 'get_inputs()' function")

    init_inputs = ref_mod.get_init_inputs() if hasattr(ref_mod, "get_init_inputs") else []
    ref_model = ref_mod.Model(*init_inputs)
    if hasattr(ref_model, "half") and dtype == torch.float16:
        ref_model = ref_model.half()
    elif hasattr(ref_model, "bfloat16") and dtype == torch.bfloat16:
        ref_model = ref_model.bfloat16()

    # Wrap get_inputs to always produce tensors on the right device
    def ref_get_inputs():
        inputs = ref_mod.get_inputs()
        return _move_inputs_to_device(inputs, device, dtype)

    # ---- Load solution module ----
    print("[INFO] Loading solution module...")
    sol_mod = _load_module_from_file(kernel_src_path, "_xpu_bench_sol")

    if not hasattr(sol_mod, "ModelNew"):
        raise RuntimeError(f"Solution file {kernel_src_path} must define a 'ModelNew' class")

    new_model = sol_mod.ModelNew(*init_inputs)
    if hasattr(new_model, "half") and dtype == torch.float16:
        new_model = new_model.half()
    elif hasattr(new_model, "bfloat16") and dtype == torch.bfloat16:
        new_model = new_model.bfloat16()

    _copy_weights(new_model, ref_model, sol_mod)

    ref_model = ref_model.to(device)
    new_model = new_model.to(device)
    ref_model.eval()
    new_model.eval()

    # ---- Correctness ----
    print("[INFO] Checking correctness...")
    passed, msg = _check_correctness(
        ref_model=ref_model,
        new_model=new_model,
        get_inputs_fn=ref_get_inputs,
        device=device,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
        num_trials=num_correct_trials,
    )
    print(f"[Eval] Correctness: {'PASS' if passed else 'FAIL'}")
    print(f"[Eval] Correctness detail: {msg}")

    if not passed:
        print("=" * 40)
        print(f"[Eval] Kernel eval result: FAILED (correctness)")
        print(f"Custom Kernel exec time: N/A")
        print(f"PyTorch Reference Eager exec time: N/A")
        print(f"Speedup over eager: N/A")
        return

    # ---- Performance ----
    print("[INFO] Measuring performance...")
    kernel_time = _measure_latency(
        model=new_model,
        get_inputs_fn=ref_get_inputs,
        device=device,
        dtype=dtype,
        num_warmup=num_warmup,
        num_trials=num_perf_trials,
    )

    ref_time = _measure_latency(
        model=ref_model,
        get_inputs_fn=ref_get_inputs,
        device=device,
        dtype=dtype,
        num_warmup=num_warmup,
        num_trials=num_perf_trials,
    )

    # torch.compile baseline
    compile_time: float | None = None
    try:
        compiled_model = torch.compile(ref_model)
        compile_time = _measure_latency(
            model=compiled_model,
            get_inputs_fn=ref_get_inputs,
            device=device,
            dtype=dtype,
            num_warmup=num_warmup,
            num_trials=num_perf_trials,
        )
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}")

    speedup_eager = ref_time / kernel_time if kernel_time > 0 else 0.0
    speedup_compile = (compile_time / kernel_time) if (compile_time and kernel_time > 0) else None

    print("=" * 40)
    print(f"[Eval] Kernel eval result: compiled=True, correctness=True")
    print(f"Custom Kernel exec time: {kernel_time:.4f} ms")
    print(f"PyTorch Reference Eager exec time: {ref_time:.4f} ms")
    print(f"Speedup over eager: {speedup_eager:.4f}x")
    if speedup_compile is not None:
        print(f"Speedup over torch.compile: {speedup_compile:.4f}x")
        print(f"torch.compile exec time: {compile_time:.4f} ms")
    else:
        print(f"Speedup over torch.compile: N/A")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XPU Bench evaluator for K-Search")
    parser.add_argument("--ref-path", required=True, help="Path to reference .py (must define Model, get_inputs)")
    parser.add_argument("--kernel-src-path", required=True, help="Path to solution .py (must define ModelNew)")
    parser.add_argument("--device", default="xpu:0", help="Device string (xpu:0, cuda:0, cpu)")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=10)
    args = parser.parse_args()

    evaluate(
        ref_path=args.ref_path,
        kernel_src_path=args.kernel_src_path,
        device=args.device,
        precision=args.precision,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        num_warmup=args.num_warmup,
    )


if __name__ == "__main__":
    main()
