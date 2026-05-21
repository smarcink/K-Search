"""HLSL kernel evaluator for K-Search.

Compiles a generated ``kernel.hlsl`` with DXC, dispatches it through
``hlsl_probe``, and compares the readback buffers against a PyTorch reference
module. The evaluator prints one JSON object on stdout for the parent task.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_hlsl_probe_on_path() -> None:
    probe_python = _repo_root() / "hlsl_probe" / "python"
    if str(probe_python) not in sys.path:
        sys.path.insert(0, str(probe_python))


def _load_module_from_file(path: str, module_name: str):
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


def _precision_to_dtype(precision: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    key = str(precision or "").lower()
    if key not in mapping:
        raise ValueError(f"unsupported HLSL precision {precision!r}; supported: fp32, fp16")
    return mapping[key]


def _select_reference_device(requested: str) -> str:
    value = str(requested or "auto").lower()
    if value != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _synchronize(device: str) -> None:
    backend = str(device or "cpu").split(":")[0]
    if backend == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif backend == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _move_inputs_to_device(inputs: list[Any], device: str, dtype: torch.dtype) -> list[Any]:
    moved: list[Any] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            moved.append(tensor.contiguous())
        else:
            moved.append(value)
    return moved


def _load_reference(ref_path: str, precision: str, reference_device: str):
    mod = _load_module_from_file(ref_path, "_hlsl_eval_ref")
    if not hasattr(mod, "Model"):
        raise RuntimeError(f"Reference file {ref_path} must define a 'Model' class")
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError(f"Reference file {ref_path} must define a 'get_inputs()' function")

    init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = mod.Model(*init_inputs)
    dtype = _precision_to_dtype(precision)
    device = _select_reference_device(reference_device)

    try:
        model = model.to(device=device)
        model = model.to(dtype=dtype)
    except TypeError:
        model = model.to(device)
    model.eval()

    def get_inputs() -> list[Any]:
        return _move_inputs_to_device(list(mod.get_inputs()), device, dtype)

    return model, get_inputs, dtype, device


def _output_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _tensor_from_output_bytes(data: bytes, expected: torch.Tensor) -> torch.Tensor:
    cpu_expected = expected.detach().cpu().contiguous()
    return torch.frombuffer(bytearray(data), dtype=cpu_expected.dtype).clone().reshape(tuple(cpu_expected.shape))


def _make_probe_inputs(inputs: list[Any], state_tensors: list[torch.Tensor]):
    from hlsl_probe import BufferArg

    probe_inputs = []
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"HLSL evaluator currently supports tensor forward args only; input {index} is {type(value).__name__}"
            )
        probe_inputs.append(BufferArg.from_tensor(value))
    for value in state_tensors:
        probe_inputs.append(BufferArg.from_tensor(value))
    return probe_inputs


def _make_output_specs(expected_outputs: list[torch.Tensor]):
    from hlsl_probe import BufferSpec

    return [BufferSpec.from_tensor(tensor) for tensor in expected_outputs]


def _compare_outputs(
    actual_outputs: list[torch.Tensor],
    expected_outputs: list[torch.Tensor],
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    if len(actual_outputs) != len(expected_outputs):
        return False, f"Output count mismatch: got={len(actual_outputs)} expected={len(expected_outputs)}"

    for index, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
        expected_cpu = expected.detach().cpu().contiguous()
        if tuple(actual.shape) != tuple(expected_cpu.shape):
            return False, f"Output {index} shape mismatch: got={tuple(actual.shape)} expected={tuple(expected_cpu.shape)}"
        if actual.dtype != expected_cpu.dtype:
            return False, f"Output {index} dtype mismatch: got={actual.dtype} expected={expected_cpu.dtype}"
        if actual.is_floating_point():
            if not torch.allclose(actual.float(), expected_cpu.float(), rtol=rtol, atol=atol):
                max_diff = (actual.float() - expected_cpu.float()).abs().max().item()
                return False, f"Output {index} values differ: max_diff={max_diff:.6e} rtol={rtol} atol={atol}"
        else:
            if not torch.equal(actual, expected_cpu):
                mismatch = (actual != expected_cpu).sum().item()
                return False, f"Output {index} integer values differ: mismatched_elements={mismatch}"
    return True, ""


def _load_launch_config(path: str | Path | None, default_target: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "target": default_target,
        "entry": "main",
        "dispatch": [1, 1, 1],
    }
    if path:
        text = Path(path).read_text(encoding="utf-8")
        user_cfg = json.loads(text) if text.strip() else {}
        if not isinstance(user_cfg, dict):
            raise ValueError("launch.json must contain a JSON object")
        cfg.update(user_cfg)

    dispatch = cfg.get("dispatch", [1, 1, 1])
    if not isinstance(dispatch, list | tuple) or len(dispatch) != 3:
        raise ValueError("launch.json field 'dispatch' must be a 3-item integer array")
    cfg["dispatch"] = [max(1, int(v)) for v in dispatch]
    cfg["target"] = str(cfg.get("target") or default_target)
    cfg["entry"] = str(cfg.get("entry") or "main")
    return cfg


def _state_tensors_for_probe(model: torch.nn.Module) -> list[torch.Tensor]:
    tensors = []
    for value in model.state_dict().values():
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().contiguous())
    return tensors


def _run_reference(model: torch.nn.Module, inputs: list[Any]) -> list[torch.Tensor]:
    with torch.no_grad():
        outputs = _output_list(model(*inputs))
    tensor_outputs = []
    for index, value in enumerate(outputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"HLSL evaluator currently supports tensor outputs only; output {index} is {type(value).__name__}")
        tensor_outputs.append(value.contiguous())
    return tensor_outputs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_hlsl_probe_on_path()
    from hlsl_probe import HlslProbe, HlslProbeError, compile_hlsl_source

    result: dict[str, Any] = {
        "compiled": False,
        "correct": False,
        "latency_ms": None,
        "hlsl_gpu_time_ms": None,
        "hlsl_host_wall_ms": None,
        "ref_latency_ms": None,
        "speedup_factor": None,
        "target": None,
        "entry": None,
        "dispatch": None,
        "error": "",
    }

    launch = _load_launch_config(args.launch_path, args.hlsl_target)
    result["target"] = launch["target"]
    result["entry"] = launch["entry"]
    result["dispatch"] = launch["dispatch"]

    source = Path(args.hlsl_path).read_text(encoding="utf-8")
    try:
        dxil = compile_hlsl_source(source, target=launch["target"], entry=launch["entry"])
    except Exception as exc:
        result["error"] = f"HLSL compilation error: {exc}"
        return result
    result["compiled"] = True
    result["dxil_size_bytes"] = len(dxil)

    model, get_inputs_fn, _dtype, reference_device = _load_reference(
        args.ref_path,
        args.precision,
        args.reference_device,
    )
    result["reference_device"] = reference_device
    state_tensors = _state_tensors_for_probe(model)

    try:
        with HlslProbe(args.agility_sdk_path) as probe:
            for trial in range(max(1, int(args.num_correct_trials))):
                inputs = get_inputs_fn()
                expected_outputs = _run_reference(model, inputs)
                _synchronize(reference_device)

                metadata, output_bytes = probe.run_dxil(
                    dxil,
                    inputs=_make_probe_inputs(inputs, state_tensors),
                    outputs=_make_output_specs(expected_outputs),
                    dispatch=tuple(launch["dispatch"]),
                )
                actual_outputs = [
                    _tensor_from_output_bytes(data, expected)
                    for data, expected in zip(output_bytes, expected_outputs)
                ]
                ok, error = _compare_outputs(actual_outputs, expected_outputs, args.rtol, args.atol)
                if not ok:
                    result["error"] = f"Correctness failed at trial {trial}: {error}"
                    result["last_run"] = metadata
                    return result

            result["correct"] = True

            perf_inputs = get_inputs_fn()
            perf_expected_outputs = _run_reference(model, perf_inputs)
            _synchronize(reference_device)
            perf_output_specs = _make_output_specs(perf_expected_outputs)
            perf_probe_inputs = _make_probe_inputs(perf_inputs, state_tensors)

            for _ in range(max(0, int(args.num_warmup))):
                probe.run_dxil(
                    dxil,
                    inputs=perf_probe_inputs,
                    outputs=perf_output_specs,
                    dispatch=tuple(launch["dispatch"]),
                )
                _run_reference(model, perf_inputs)
                _synchronize(reference_device)

            gpu_times: list[float] = []
            host_times: list[float] = []
            for _ in range(max(1, int(args.num_perf_trials))):
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
            for _ in range(max(1, int(args.num_perf_trials))):
                t0 = time.perf_counter()
                _run_reference(model, perf_inputs)
                _synchronize(reference_device)
                ref_times.append((time.perf_counter() - t0) * 1000.0)

            hlsl_gpu_ms = statistics.median(gpu_times) if gpu_times else None
            hlsl_host_ms = statistics.median(host_times) if host_times else None
            ref_ms = statistics.median(ref_times) if ref_times else None
            result["hlsl_gpu_time_ms"] = hlsl_gpu_ms
            result["hlsl_host_wall_ms"] = hlsl_host_ms
            result["latency_ms"] = hlsl_gpu_ms or hlsl_host_ms
            result["ref_latency_ms"] = ref_ms
            if result["latency_ms"] and ref_ms:
                result["speedup_factor"] = float(ref_ms) / float(result["latency_ms"])
    except HlslProbeError as exc:
        result["error"] = f"HLSL probe error: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HLSL kernel through hlsl_probe")
    parser.add_argument("--ref-path", required=True)
    parser.add_argument("--hlsl-path", required=True)
    parser.add_argument("--launch-path", default=None)
    parser.add_argument("--hlsl-target", default="cs_6_8")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--reference-device", default="auto")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--agility-sdk-path", default=None)
    args = parser.parse_args()

    print(json.dumps(evaluate(args)))


if __name__ == "__main__":
    main()