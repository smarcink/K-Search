"""Generic kernel comparison script.

Compares optimized kernel solutions (CUDA, Triton, or HLSL) against a reference
PyTorch model. Works with any task that follows the K-Search convention:
  - Reference .py file with: Model class, get_inputs(), get_init_inputs()
  - CUDA solutions: kernel.h / kernel.cu / main.cpp (or K-Search artifacts)
  - Triton solutions: model_new.py with ModelNew class (or K-Search artifacts)
    - HLSL solutions: kernel.hlsl / launch.json (or K-Search artifacts)

Usage:
    # Compare extracted kernel dirs against a reference
    python compare_kernels.py path/to/reference.py path/to/cuda_dir path/to/triton_dir

    # Compare K-Search artifacts directories
    python compare_kernels.py path/to/reference.py path/to/ksearch-artifacts1 path/to/ksearch-artifacts2

    # Compare an HLSL K-Search artifact or extracted HLSL dir through hlsl_probe
    python compare_kernels.py elementwise_add_fp16/elementwise_add_fp16.py best_hlsl/ \
        --device cpu --warmup 1 --iters 100

    # With options
    python compare_kernels.py path/to/reference.py path/to/kernels \\
        --precision fp16 --compile-mode default --shape 64 1024 1024 \\
        --warmup 100 --iters 2000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
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


def _solution_from_json_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sources = data.get("sources")
    if isinstance(sources, list):
        return data
    return None


def _kernel_from_solution_json(sol_file: Path, sol: dict) -> dict | None:
    sources = sol.get("sources", [])
    if not isinstance(sources, list):
        return None
    src_map = {s["path"]: s["content"] for s in sources if isinstance(s, dict) and "path" in s and "content" in s}
    name = str(sol.get("name", sol_file.stem))[:60]
    if "kernel.cu" in src_map and "main.cpp" in src_map:
        return {
            "type": "cuda",
            "name": name,
            "kernel_h": src_map.get("kernel.h", ""),
            "kernel_cu": src_map["kernel.cu"],
            "main_cpp": src_map["main.cpp"],
            "source": str(sol_file),
        }
    if "model_new.py" in src_map:
        return {
            "type": "triton",
            "name": name,
            "model_new_code": src_map["model_new.py"],
            "source": str(sol_file),
        }
    if "kernel.hlsl" in src_map:
        return {
            "type": "hlsl",
            "name": name,
            "kernel_hlsl": src_map["kernel.hlsl"],
            "launch_json": src_map.get("launch.json", "{}"),
            "source": str(sol_file),
        }
    return None


def collect_kernels(paths: list[str]) -> list[dict]:
    """Collect kernel sources from artifacts dirs or direct kernel dirs.

    Returns list of dicts with either:
      - type='cuda', kernel_h, kernel_cu, main_cpp
      - type='triton', model_new_code
            - type='hlsl', kernel_hlsl, launch_json
    """
    kernels = []

    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"  [warn] path not found: {path}", file=sys.stderr)
            continue

        # Case 0: a solution JSON file directly
        if path.is_file() and path.suffix.lower() == ".json":
            sol = _solution_from_json_file(path)
            kernel = _kernel_from_solution_json(path, sol) if sol else None
            if kernel:
                kernels.append(kernel)
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

        # Case 1c: directory with kernel.hlsl directly
        if (path / "kernel.hlsl").exists():
            kernels.append({
                "type": "hlsl",
                "name": path.name,
                "kernel_hlsl": (path / "kernel.hlsl").read_text(),
                "launch_json": (path / "launch.json").read_text() if (path / "launch.json").exists() else "{}",
                "source": str(path),
            })
            continue

        # Case 2: K-Search artifacts directory — scan for solution JSONs
        for sol_file in sorted(path.rglob("*.json")):
            if sol_file.name == "solution.json" or "eval_report" in sol_file.name:
                continue
            try:
                sol = json.loads(sol_file.read_text())
                kernel = _kernel_from_solution_json(sol_file, sol)
                if kernel:
                    kernels.append(kernel)
            except (json.JSONDecodeError, KeyError):
                continue

    return kernels


def benchmark_hlsl_kernel(
    kernel_hlsl: str,
    launch_json: str,
    *,
    ref_path: Path,
    precision: str,
    warmup: int,
    iters: int,
    atol: float,
    hlsl_target: str,
    reference_device: str,
    hlsl_agility_sdk_path: str | None,
) -> dict[str, Any]:
    """Evaluate an HLSL kernel and return timing plus tensors for the common diff path."""
    from k_search.tasks import hlsl_kernel_eval as hlsl_eval

    hlsl_eval._ensure_hlsl_probe_on_path()
    from hlsl_probe import HlslProbe, compile_hlsl_source

    with tempfile.TemporaryDirectory(prefix="compare_hlsl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        hlsl_path = tmp_path / "kernel.hlsl"
        launch_path = tmp_path / "launch.json"
        hlsl_path.write_text(kernel_hlsl, encoding="utf-8")
        launch_path.write_text(launch_json or "{}", encoding="utf-8")

        launch = hlsl_eval._load_launch_config(launch_path, hlsl_target)
        dxil = compile_hlsl_source(kernel_hlsl, target=launch["target"], entry=launch["entry"])
        model, get_inputs_fn, _dtype, selected_reference_device = hlsl_eval._load_reference(
            str(ref_path),
            precision,
            reference_device,
        )
        state_tensors = hlsl_eval._state_tensors_for_probe(model)

        with HlslProbe(hlsl_agility_sdk_path) as probe:
            inputs = get_inputs_fn()
            expected_outputs = hlsl_eval._run_reference(model, inputs)
            hlsl_eval._synchronize(selected_reference_device)

            metadata, output_bytes = probe.run_dxil(
                dxil,
                inputs=hlsl_eval._make_probe_inputs(inputs, state_tensors),
                outputs=hlsl_eval._make_output_specs(expected_outputs),
                dispatch=tuple(launch["dispatch"]),
            )
            actual_outputs = [
                hlsl_eval._tensor_from_output_bytes(data, expected)
                for data, expected in zip(output_bytes, expected_outputs)
            ]
            ok, error = hlsl_eval._compare_outputs(actual_outputs, expected_outputs, atol, atol)
            if not ok:
                raise RuntimeError(error)

            perf_inputs = get_inputs_fn()
            perf_expected_outputs = hlsl_eval._run_reference(model, perf_inputs)
            hlsl_eval._synchronize(selected_reference_device)
            perf_probe_inputs = hlsl_eval._make_probe_inputs(perf_inputs, state_tensors)
            perf_output_specs = hlsl_eval._make_output_specs(perf_expected_outputs)

            for _ in range(max(0, int(warmup))):
                probe.run_dxil(
                    dxil,
                    inputs=perf_probe_inputs,
                    outputs=perf_output_specs,
                    dispatch=tuple(launch["dispatch"]),
                )
                hlsl_eval._run_reference(model, perf_inputs)
                hlsl_eval._synchronize(selected_reference_device)

            gpu_times: list[float] = []
            host_times: list[float] = []
            for _ in range(max(1, int(iters))):
                t0 = time.perf_counter()
                metadata, _ = probe.run_dxil(
                    dxil,
                    inputs=perf_probe_inputs,
                    outputs=perf_output_specs,
                    dispatch=tuple(launch["dispatch"]),
                )
                host_times.append((time.perf_counter() - t0) * 1000.0)
                gpu_time = metadata.get("gpu_time_ms") if isinstance(metadata, dict) else None
                if isinstance(gpu_time, (int, float)):
                    gpu_times.append(float(gpu_time))

            ref_times: list[float] = []
            for _ in range(max(1, int(iters))):
                t0 = time.perf_counter()
                hlsl_eval._run_reference(model, perf_inputs)
                hlsl_eval._synchronize(selected_reference_device)
                ref_times.append((time.perf_counter() - t0) * 1000.0)

        hlsl_gpu_ms = statistics.median(gpu_times) if gpu_times else None
        hlsl_host_ms = statistics.median(host_times) if host_times else None
        ref_ms = statistics.median(ref_times) if ref_times else None
        latency_ms = hlsl_gpu_ms or hlsl_host_ms
        speedup = (float(ref_ms) / float(latency_ms)) if ref_ms and latency_ms else None
        return {
            "compiled": True,
            "correct": True,
            "latency_ms": latency_ms,
            "hlsl_gpu_time_ms": hlsl_gpu_ms,
            "hlsl_host_wall_ms": hlsl_host_ms,
            "ref_latency_ms": ref_ms,
            "speedup_factor": speedup,
            "reference_device": selected_reference_device,
            "actual_output": _flatten_outputs(actual_outputs),
            "expected_output": _flatten_outputs([value.detach().cpu().contiguous() for value in expected_outputs]),
        }


def _flatten_outputs(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) == 1:
        return outputs[0].detach().cpu().contiguous()
    return torch.cat([value.detach().cpu().contiguous().flatten() for value in outputs])


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
    cos_raw = (torch.dot(a32, b32) / denom.clamp_min(1e-12)).item() if denom > 0 else 1.0
    cos_sim = max(-1.0, min(1.0, float(cos_raw)))
    return max_abs, mse, cos_sim


def _try_set_params(module, ref_model: nn.Module) -> None:
    """Pass reference model weights to a CUDA module via set_params()."""
    # Method 1: Conv2d shortcut (for conv-based tasks)
    for m in ref_model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            b = m.bias.data if m.bias is not None else torch.Tensor()
            module.set_params(w, b)
            return

    # Method 2: Try passing all state (multiple orderings)
    buffers = [b.data for b in ref_model.buffers()]
    params = [p.data for p in ref_model.parameters()]

    # Try buffers + params (common for kernels that expect bias tables first)
    try:
        module.set_params(*(buffers + params))
        return
    except TypeError:
        pass

    # Try params + buffers (state_dict default order)
    try:
        module.set_params(*(params + buffers))
        return
    except TypeError:
        pass

    # Try just params (no buffers)
    try:
        module.set_params(*params)
        return
    except TypeError:
        pass


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
                        help="PyTorch device for reference/CUDA/Triton timing (e.g. cpu, cuda:0, xpu:0). Auto-detected if omitted. HLSL dispatch still uses hlsl_probe/D3D12.")
    parser.add_argument("--hlsl-target", default="cs_6_8",
                        help="Shader model target used when an HLSL launch.json omits target")
    parser.add_argument("--hlsl-agility-sdk-path", default=None,
                        help="Optional Direct3D 12 Agility SDK path for hlsl_probe")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    ref_path = Path(args.ref).resolve()
    if not ref_path.exists():
        sys.exit(f"Reference file not found: {ref_path}")

    # Collect kernels first so HLSL-only comparisons do not require CUDA/XPU PyTorch events.
    kernels = collect_kernels(args.paths)
    if not kernels:
        sys.exit("No kernel solutions found in the provided paths.")

    has_hlsl = any(kernel.get("type") == "hlsl" for kernel in kernels)
    has_torch_gpu_kernel = any(kernel.get("type") != "hlsl" for kernel in kernels)
    if has_hlsl and args.precision == "bf16":
        sys.exit("HLSL evaluator currently supports fp16/fp32 only; use --precision fp16 or --precision fp32.")
    if has_hlsl and args.shape:
        sys.exit("--shape override is not supported for HLSL kernels yet; use the reference get_inputs() shape.")

    device = get_device(args.device)
    backend = device.split(":")[0]
    if has_torch_gpu_kernel and backend not in ("cuda", "xpu"):
        sys.exit("CUDA or XPU GPU is required for CUDA/Triton comparison. HLSL-only comparison can use hlsl_probe.")
    hlsl_reference_device = device

    if has_torch_gpu_kernel:
        print(f"Device: {device}  GPU: {get_device_name(device)}")
    else:
        print(f"Device: hlsl_probe  PyTorch reference device: {hlsl_reference_device}")
    print(f"Precision: {args.precision}  Warmup: {args.warmup}  Iters: {args.iters}")
    if args.compile_mode:
        print(f"torch.compile mode: {args.compile_mode}")
    print()

    print(f"Reference: {ref_path.name}")
    ref_ms: float | None = None
    ref_out = None
    model = None
    get_init_inputs_fn = None
    inputs: list[Any] = []
    dtype = None

    if has_torch_gpu_kernel:
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

    print(f"Found {len(kernels)} kernel(s) to compare\n")

    # Get init_inputs for triton models
    init_inputs = get_init_inputs_fn() if get_init_inputs_fn is not None else []

    # Collect model parameters for CUDA kernels that may need them
    model_params = [p.data for p in model.parameters()] if model is not None else []

    # Benchmark each kernel
    rows: list[tuple[str, float | None, float | None, float | None, float | None, float | None, str]] = []
    for kernel_index, kernel in enumerate(kernels):
        name = kernel["name"]
        ktype = kernel.get("type", "cuda")
        action = "Evaluating" if ktype == "hlsl" else "Compiling"
        print(f"  [{kernel_index+1}/{len(kernels)}] [{ktype}] {action}: {name}...", end=" ", flush=True)
        try:
            latency_extra_text = ""
            comparison_ref_out = ref_out
            if ktype == "cuda":
                if backend == "xpu":
                    raise RuntimeError("CUDA kernel compilation not supported on XPU device")
                module = compile_kernel(
                    kernel["kernel_h"], kernel["kernel_cu"], kernel["main_cpp"],
                    name=f"k{kernel_index}",
                )
                # If the module exposes set_params(), pass reference model weights
                if hasattr(module, "set_params"):
                    _try_set_params(module, model)
                # Try run with just inputs first; if it fails, try with model params appended
                try:
                    ms, out = benchmark_cuda_module(module, inputs, args.warmup, args.iters, device=device)
                except TypeError:
                    ms, out = benchmark_cuda_module(module, inputs + model_params, args.warmup, args.iters, device=device)
            elif ktype == "hlsl":
                result = benchmark_hlsl_kernel(
                    kernel["kernel_hlsl"],
                    kernel.get("launch_json", "{}"),
                    ref_path=ref_path,
                    precision=args.precision,
                    warmup=args.warmup,
                    iters=args.iters,
                    atol=args.atol,
                    hlsl_target=args.hlsl_target,
                    reference_device=hlsl_reference_device,
                    hlsl_agility_sdk_path=args.hlsl_agility_sdk_path,
                )
                if not result.get("compiled") or not result.get("correct"):
                    raise RuntimeError(str(result.get("error") or "HLSL evaluation failed"))
                ms = result.get("latency_ms")
                if not isinstance(ms, (int, float)):
                    raise RuntimeError("HLSL evaluation passed but did not report latency_ms")
                speedup = result.get("speedup_factor")
                hlsl_ref_ms = result.get("ref_latency_ms")
                if ref_ms is None and isinstance(hlsl_ref_ms, (int, float)):
                    ref_ms = float(hlsl_ref_ms)
                out = result["actual_output"]
                comparison_ref_out = result["expected_output"]
                host_ms = result.get("hlsl_host_wall_ms")
                latency_extra_text = f" host={host_ms:.4f}ms" if isinstance(host_ms, (int, float)) else ""
            else:  # triton
                triton_model = load_triton_model(
                    kernel["model_new_code"], init_inputs, dtype, name=f"k{kernel_index}", device=device
                )
                _copy_weights(triton_model, model)
                ms, out = benchmark_model(triton_model, inputs, args.warmup, args.iters, device=device)
            if comparison_ref_out is None:
                raise RuntimeError("Reference output unavailable for correctness comparison")
            max_abs, mse, cos_sim = diff(out, comparison_ref_out)
            ok = max_abs <= args.atol
            status = "PASS" if ok else "FAIL"
            speedup = (float(ref_ms) / float(ms)) if isinstance(ref_ms, (int, float)) and ms > 0 else None
            rows.append((name, ms, speedup, max_abs, mse, cos_sim, status))
            speed_text = f" speedup={speedup:.2f}x" if speedup is not None else ""
            print(f"{ms:.4f}ms{latency_extra_text}{speed_text}  {status}")
        except Exception as e:
            err_msg = str(e).split("\n")[0][:80]
            rows.append((name, None, None, None, None, None, "ERROR"))
            print(f"ERROR: {err_msg}")

    # Summary table
    ref_label = f"[reference] {ref_path.name}"
    name_w = max(len(ref_label), *(len(r[0]) for r in rows)) if rows else len(ref_label)
    name_w = min(name_w, 60)
    print(f"\n{'='*100}")
    print(f"{'impl':<{name_w}}  {'ms/iter':>10}  {'speedup':>8}  {'max_abs':>11}  {'mse':>11}  {'cos_sim':>10}  status")
    print(f"{'-'*100}")

    # Reference row
    if isinstance(ref_ms, (int, float)):
        print(f"{ref_label:<{name_w}}  {ref_ms:>10.4f}  {'1.00x':>8}  {'0.000e+00':>11}  {'0.000e+00':>11}  {'1.000000':>10}  REF")
    else:
        print(f"{ref_label:<{name_w}}  {'N/A':>10}  {'1.00x':>8}  {'0.000e+00':>11}  {'0.000e+00':>11}  {'1.000000':>10}  REF")

    for name, ms, speedup, ma, mse_v, cs, status in rows:
        display_name = name[:name_w]
        if not isinstance(ms, (int, float)) or ms <= 0:
            print(f"{display_name:<{name_w}}  {'N/A':>10}  {'-':>8}  {'N/A':>11}  {'N/A':>11}  {'N/A':>10}  {status}")
        else:
            speed = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "-"
            max_abs_text = f"{ma:>11.3e}" if isinstance(ma, (int, float)) else f"{'N/A':>11}"
            mse_text = f"{mse_v:>11.3e}" if isinstance(mse_v, (int, float)) else f"{'N/A':>11}"
            cos_text = f"{cs:>10.6f}" if isinstance(cs, (int, float)) else f"{'N/A':>10}"
            print(f"{display_name:<{name_w}}  {ms:>10.4f}  {speed:>8}  {max_abs_text}  {mse_text}  {cos_text}  {status}")

    print(f"{'='*100}")
    if isinstance(ref_ms, (int, float)):
        print(f"(reference latency = {ref_ms:.4f} ms)")
    else:
        print("(reference latency unavailable; all kernel evaluations failed before timing)")


if __name__ == "__main__":
    main()
