"""HLSL Kernel Task for K-Search.

This backend is the first Direct3D 12/HLSL integration slice. Generated
solutions contain ``kernel.hlsl`` plus ``launch.json`` and are evaluated by the
minimal ``hlsl_probe`` runner from Python.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)


@dataclass(frozen=True)
class HlslKernelTaskConfig:
    gpu: str = "Direct3D 12 GPU"
    hlsl_target: str = "cs_6_8"
    precision: str = "fp16"
    reference_device: str = "auto"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    num_warmup: int = 5
    timeout: int = 300
    rtol: float = 1e-2
    atol: float = 1e-2
    agility_sdk_path: str | None = None
    max_failure_excerpt_chars: int = 4000


_HLSL_CODE_FORMAT = """IMPORTANT: Generate code in XML format with exactly 2 files:

<hlsl_file name="kernel.hlsl">
- A complete HLSL compute shader.
- Entry point must be `main` unless launch.json says otherwise.
- Use [numthreads(X, Y, Z)] and fixed shape constants from the task metadata.
- Declare input SRVs as t0, t1, ... and output UAVs as u0, u1, ... exactly as specified.
- Use typed buffers for tensor data: Buffer<float16_t>, Buffer<float>, Buffer<int>, RWBuffer<float16_t>, etc.
- For int8/uint8 typed descriptors, HLSL exposes elements as int/uint values even though storage is one byte.
- Do not use CUDA, Triton, PyTorch, root constants, CBVs, descriptor spaces, or undeclared resources.
</hlsl_file>

<json_file name="launch.json">
{
  "target": "cs_6_8",
  "entry": "main",
  "dispatch": [1, 1, 1]
}
</json_file>

Return only these XML blocks. No markdown or explanations."""


_HLSL_GENERATION_GUIDELINES = """## HLSL/DX12 Optimization Guidelines
- This first POC backend is single-dispatch and CPU-staged through hlsl_probe; optimize the shader's GPU timestamp.
- Hardcode the fixed shapes shown in the metadata. Dynamic/root constants are not wired up yet.
- Flatten tensors in row-major contiguous order. For shape [A, B, C], linear offset is ((a * B) + b) * C + c.
- Prefer coalesced adjacent loads/stores across SV_DispatchThreadID.x.
- For cs_6_8/cs_6_9, keep total groupshared memory under 32 KiB per threadgroup. Do not use CUDA per-SM shared-memory limits as the HLSL per-group limit.
- Prefer register-resident scalars/vectors for per-thread or single-owner intermediate values. Full fusion should minimize materialization, not store every phase's output in groupshared memory.
- Treat groupshared memory as an explicitly synchronized communication/cache resource. Good uses include compact read-only tiles/tables loaded cooperatively once and consumed many times, reusable input tiles, cross-thread or cross-wave exchange buffers, and reduction scratch.
- Do not justify groupshared use only because a tensor or working set fits under 32 KiB. For every groupshared array, classify it as read-only reusable tile, exchange buffer, or reduction scratch, and state the producer thread set, consumer thread set, reuse count/traffic saving, and required synchronization point.
- Leave safety margin below 32 KiB for groupshared layout/alignment. Do not stage a whole working set or whole phase tensors in groupshared merely to pass values between sequential phases when streaming from SRVs/L2, using wave communication, keeping owner-local registers, or recomputing a small value is cheaper.
- Prefer mappings where one fixed-size wave owns one independent work unit/tile when the tile fits wave-local ownership. Use WaveGetLaneIndex, WaveReadLaneAt, WaveReadLaneFirst, WaveActiveSum/Max/Min, and related wave intrinsics for lane exchange and reductions.
- Do not route wave-local intermediate values through groupshared memory. Keep them in registers and exchange through wave intrinsics; this avoids turning warp/wave-local dependencies into group-wide barriers.
- Avoid splitting one independent work unit across multiple waves unless the extra parallelism clearly pays for the required inter-wave handoff. If multiple waves share a threadgroup, prefer each wave owning an independent work unit, with at most compact read-only shared tiles loaded once for all waves.
- Minimize GroupMemoryBarrierWithGroupSync calls. They are usually needed only after cross-thread groupshared writes before cross-thread reads; avoid phase-by-phase barriers for values that can stay in registers or have a single producer/consumer.
- Use [WaveSize(32)] when the target accepts it and the mapping assumes 32 lanes. Prefer wave intrinsics for wave-local reductions, broadcasts, scans, or shuffles; use group-wide barriers only for true group-wide communication.
- For fp16 elementwise work, use float16_t where possible. Widen to float only when needed for numerical tolerance.
"""


class HlslKernelTask:
    """Task for optimizing a PyTorch reference with a generated HLSL compute shader."""

    def __init__(
        self,
        *,
        ref_path: str,
        gpu: str = "Direct3D 12 GPU",
        hlsl_target: str = "cs_6_8",
        precision: str = "fp16",
        reference_device: str = "auto",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 5,
        timeout: int = 300,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        agility_sdk_path: str | None = None,
        artifacts_dir: str | None = None,
        name: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._ref_path = str(Path(ref_path).resolve())
        if not Path(self._ref_path).exists():
            raise FileNotFoundError(f"Reference file not found: {self._ref_path}")

        self._cfg = HlslKernelTaskConfig(
            gpu=str(gpu),
            hlsl_target=str(hlsl_target),
            precision=str(precision),
            reference_device=str(reference_device),
            num_correct_trials=int(num_correct_trials),
            num_perf_trials=int(num_perf_trials),
            num_warmup=int(num_warmup),
            timeout=int(timeout),
            rtol=float(rtol),
            atol=float(atol),
            agility_sdk_path=str(agility_sdk_path) if agility_sdk_path else None,
        )
        self._name = str(name or Path(self._ref_path).stem)
        self._artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._verbose = bool(verbose)
        self._solutions: dict[str, Solution] = {}

        self._last_round_trace_logs: str = ""
        self._last_round_passed: bool = False
        self._last_round_summary: str = ""
        self._metadata_text_cache: str | None = None

        self._ref_code = Path(self._ref_path).read_text(encoding="utf-8")
        print(f"[{self._name}] Loaded reference: {self._ref_path}  (HLSL target={self._cfg.hlsl_target})")

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        metadata_text = self._reference_metadata_text()
        return f"""# HLSL Kernel Optimization Task

**Reference Module**: {Path(self._ref_path).name}
**Target GPU**: {self._cfg.gpu}
**HLSL Target**: {self._cfg.hlsl_target}
**Precision**: {self._cfg.precision}

## Objective
Optimize the following PyTorch reference implementation by writing a custom HLSL compute shader.
Your implementation must produce numerically equivalent outputs (rtol={self._cfg.rtol}, atol={self._cfg.atol}).

## Reference Implementation

```python
{self._ref_code}
```

## Fixed Evaluation Metadata
{metadata_text}

## Binding Contract
- Forward input tensors are bound as SRVs starting at register t0 in the order shown above.
- Model state_dict tensors are bound as additional SRVs immediately after the forward inputs, in the order shown above.
- Output tensors are bound as UAVs starting at register u0 in the order shown above.
- All tensors are contiguous, flattened, row-major buffers.
- The evaluator dispatches exactly the integer group counts from launch.json.
- The current hlsl_probe runner does not provide root constants, CBVs, temporary buffers, or multi-dispatch graphs.

## Format
{_HLSL_CODE_FORMAT}
"""

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return f"""You are a Direct3D 12 HLSL compute shader generator for {target_gpu}.

{self.get_definition_text(language)}

{_HLSL_GENERATION_GUIDELINES}

Generate the implementation:"""

    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        prompt = f"""You are optimizing an HLSL compute shader for {target_gpu}.

{self.get_definition_text(language)}

## Current Implementation
{str(current_code or '').strip()}
"""
        if previous_round_summary:
            prompt += "\n\n## Previous Round Summary\n" + str(previous_round_summary).strip()
        if trace_logs:
            prompt += "\n\n## Evaluation Feedback\n" + str(trace_logs).strip()
        if current_best:
            prompt += "\n\n## Current Best Performance\n" + str(current_best).strip()
        prompt += "\n\n" + _HLSL_GENERATION_GUIDELINES
        prompt += "\n\nReturn the full corrected XML blocks only."
        return prompt

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        return _HLSL_CODE_FORMAT

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        files = self._parse_generated_files(cleaned_code=cleaned_code, raw_code=raw_code)
        safe_model_name = str(model_name).replace("/", "_").replace("\\", "_")
        sol_name = f"{safe_model_name}_{self._name}_hlsl_r{round_num}"
        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.HLSL,
                target_hardware=[str(target_gpu or self._cfg.gpu)],
                entry_point="kernel.hlsl::main",
            ),
            sources=[
                SourceFile(path="kernel.hlsl", content=files["kernel.hlsl"]),
                SourceFile(path="launch.json", content=files["launch.json"]),
            ],
            description=f"HLSL kernel optimization for {self._name} round {round_num}",
        )

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions[name]
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=self.name,
                artifacts_dir=self._artifacts_dir,
                target_hardware=self._cfg.gpu,
            )
            sol = solution_from_json_dict(d)
            if sol.definition != self.name:
                return None
            self._solutions[sol.name] = sol
            return sol
        except Exception:
            return None

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        if isinstance(raw, dict):
            return str(raw.get("kernel.hlsl") or raw)
        text = str(raw or "")
        files = self._parse_xml_blocks(text)
        return files.get("kernel.hlsl", text)

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        sources = {sf.path: sf.content for sf in (solution.sources or [])}
        kernel_hlsl = sources.get("kernel.hlsl", "")
        launch_json = sources.get("launch.json", self._default_launch_json())
        if not kernel_hlsl.strip():
            return self._failed_eval("Missing kernel.hlsl in HLSL solution", round_num)

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="hlsl_kernel_eval_")
            hlsl_path = Path(tmp_dir, "kernel.hlsl")
            launch_path = Path(tmp_dir, "launch.json")
            hlsl_path.write_text(kernel_hlsl, encoding="utf-8")
            launch_path.write_text(launch_json, encoding="utf-8")

            evaluator_path = Path(__file__).parent / "hlsl_kernel_eval.py"
            cmd = [
                sys.executable,
                str(evaluator_path),
                "--ref-path", self._ref_path,
                "--hlsl-path", str(hlsl_path),
                "--launch-path", str(launch_path),
                "--hlsl-target", self._cfg.hlsl_target,
                "--precision", self._cfg.precision,
                "--reference-device", self._cfg.reference_device,
                "--num-correct-trials", str(self._cfg.num_correct_trials),
                "--num-perf-trials", str(self._cfg.num_perf_trials),
                "--num-warmup", str(self._cfg.num_warmup),
                "--rtol", str(self._cfg.rtol),
                "--atol", str(self._cfg.atol),
            ]
            if self._cfg.agility_sdk_path:
                cmd.extend(["--agility-sdk-path", self._cfg.agility_sdk_path])

            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[2]
            path_entries = [
                str(Path(self._ref_path).parent),
                str(repo_root),
                str(repo_root / "hlsl_probe" / "python"),
            ]
            env["PYTHONPATH"] = os.pathsep.join(path_entries + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout,
                env=env,
                cwd=str(repo_root),
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            if result.returncode != 0:
                return self._failed_eval(
                    f"HLSL evaluator failed with exit code {result.returncode}:\n{output[-self._cfg.max_failure_excerpt_chars:]}",
                    round_num,
                )
            eval_result = self._parse_eval_output(result.stdout)
            if self._verbose:
                self._print_debug_report(eval_result, kernel_hlsl, launch_json)
            return eval_result
        except subprocess.TimeoutExpired:
            return self._failed_eval(f"HLSL evaluation timed out after {self._cfg.timeout}s", round_num)
        except Exception as exc:
            return self._failed_eval(f"HLSL evaluation error: {type(exc).__name__}: {exc}", round_num)
        finally:
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = {}
        for sol in solutions:
            er = self.run_benchmark(solution=sol, config=config, dump_traces=dump_traces, round_num=None)
            results[sol.name] = er.to_dict(include_log_excerpt=True)
        return results

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "hlsl_kernel",
            "ref_path": self._ref_path,
            "gpu": self._cfg.gpu,
            "hlsl_target": self._cfg.hlsl_target,
            "precision": self._cfg.precision,
            "num_correct_trials": self._cfg.num_correct_trials,
            "num_perf_trials": self._cfg.num_perf_trials,
        }

    def has_last_round_feedback_trace(self) -> bool:
        return bool(self._last_round_trace_logs)

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs

    def get_last_round_passed_count(self) -> int:
        return 1 if self._last_round_passed else 0

    def get_last_round_total_workloads(self) -> int:
        return 1

    def _parse_eval_output(self, stdout: str) -> EvalResult:
        json_line = None
        for line in stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                json_line = line
        if not json_line:
            return self._failed_eval(f"No JSON output from HLSL evaluator. stdout:\n{stdout[-2000:]}", None)

        try:
            data = json.loads(json_line)
        except json.JSONDecodeError as exc:
            return self._failed_eval(f"Invalid JSON from HLSL evaluator: {exc}\n{json_line[:500]}", None)

        compiled = bool(data.get("compiled", False))
        correct = bool(data.get("correct", False))
        error = str(data.get("error", "") or "")
        if not compiled:
            return self._failed_eval(f"HLSL compilation failed: {error}", None)
        if not correct:
            return self._failed_eval(f"HLSL correctness failed: {error}", None)

        latency_ms = data.get("latency_ms")
        ref_latency_ms = data.get("ref_latency_ms")
        speedup = data.get("speedup_factor")
        gpu_ms = data.get("hlsl_gpu_time_ms")
        host_ms = data.get("hlsl_host_wall_ms")
        summary = (
            f"PASSED: hlsl_gpu={gpu_ms:.6f}ms host={host_ms:.6f}ms "
            f"ref={ref_latency_ms:.6f}ms speedup={speedup:.3f}x"
            if all(isinstance(v, (int, float)) for v in (gpu_ms, host_ms, ref_latency_ms, speedup))
            else "PASSED"
        )
        self._last_round_trace_logs = summary
        self._last_round_passed = True
        self._last_round_summary = summary
        return EvalResult(
            status="passed",
            latency_ms=float(latency_ms) if isinstance(latency_ms, (int, float)) else None,
            reference_latency_ms=float(ref_latency_ms) if isinstance(ref_latency_ms, (int, float)) else None,
            speedup_factor=float(speedup) if isinstance(speedup, (int, float)) else None,
            log_excerpt=summary,
            metrics={
                "score_name": "speedup_vs_reference",
                "score": float(speedup) if isinstance(speedup, (int, float)) else None,
                "hlsl_gpu_time_ms": gpu_ms,
                "hlsl_host_wall_ms": host_ms,
                "target": data.get("target"),
                "dispatch": data.get("dispatch"),
                "reference_device": data.get("reference_device"),
            },
        )

    def _failed_eval(self, message: str, round_num: int | None) -> EvalResult:
        self._last_round_trace_logs = message
        self._last_round_passed = False
        self._last_round_summary = f"FAILED: {message}"
        if round_num is not None:
            print(f"[{self._name}] Round {round_num}: status=failed | HLSL target={self._cfg.hlsl_target}")
        if message:
            print(f"[{self._name}] Failure excerpt:\n{message[-self._cfg.max_failure_excerpt_chars:]}", flush=True)
        return EvalResult(
            status="failed",
            log_excerpt=message[-self._cfg.max_failure_excerpt_chars:],
            metrics={"score_name": "speedup_vs_reference", "score": None},
        )

    def _print_debug_report(self, eval_result: EvalResult, kernel_hlsl: str, launch_json: str) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[DEBUG] [{self._name}] HLSL eval status: {eval_result.status}")
        print(f"{sep}\n--- kernel.hlsl ---")
        print(kernel_hlsl)
        print("\n--- launch.json ---")
        print(launch_json)
        print(f"{sep}\n")

    def _parse_generated_files(self, *, cleaned_code: Any, raw_code: Any) -> dict[str, str]:
        if isinstance(cleaned_code, dict) and "kernel.hlsl" in cleaned_code:
            launch = str(cleaned_code.get("launch.json") or self._default_launch_json())
            return {
                "kernel.hlsl": str(cleaned_code.get("kernel.hlsl") or ""),
                "launch.json": self._normalize_launch_json(launch),
            }

        for candidate in (raw_code, cleaned_code):
            text = str(candidate or "")
            files = self._parse_xml_blocks(text)
            if files.get("kernel.hlsl"):
                launch = files.get("launch.json") or self._default_launch_json()
                return {
                    "kernel.hlsl": files["kernel.hlsl"],
                    "launch.json": self._normalize_launch_json(launch),
                }

        kernel = self._strip_markdown_fence(str(cleaned_code or raw_code or ""))
        return {
            "kernel.hlsl": kernel,
            "launch.json": self._default_launch_json(),
        }

    def _parse_xml_blocks(self, text: str) -> dict[str, str]:
        files: dict[str, str] = {}
        patterns = {
            "kernel.hlsl": [
                r'<hlsl_file\s+name="kernel\.hlsl"\s*>([\s\S]*?)</hlsl_file>',
                r'<file\s+name="kernel\.hlsl"\s*>([\s\S]*?)</file>',
            ],
            "launch.json": [
                r'<json_file\s+name="launch\.json"\s*>([\s\S]*?)</json_file>',
                r'<launch_file\s+name="launch\.json"\s*>([\s\S]*?)</launch_file>',
                r'<file\s+name="launch\.json"\s*>([\s\S]*?)</file>',
            ],
        }
        for filename, filename_patterns in patterns.items():
            for pattern in filename_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    files[filename] = match.group(1).strip()
                    break
        return files

    def _normalize_launch_json(self, text: str) -> str:
        try:
            obj = json.loads(str(text or "{}"))
            if not isinstance(obj, dict):
                obj = {}
        except json.JSONDecodeError:
            obj = {}
        obj.setdefault("target", self._cfg.hlsl_target)
        obj.setdefault("entry", "main")
        dispatch = obj.get("dispatch")
        if not isinstance(dispatch, list) or len(dispatch) != 3:
            obj["dispatch"] = [1, 1, 1]
        else:
            obj["dispatch"] = [max(1, int(v)) for v in dispatch]
        return json.dumps(obj, indent=2)

    def _default_launch_json(self) -> str:
        return json.dumps({"target": self._cfg.hlsl_target, "entry": "main", "dispatch": [1, 1, 1]}, indent=2)

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        value = str(text or "").strip()
        match = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```", value)
        if match:
            return match.group(1).strip()
        return value.replace("```", "").strip()

    def _reference_metadata_text(self) -> str:
        if self._metadata_text_cache is not None:
            return self._metadata_text_cache
        try:
            self._metadata_text_cache = self._build_reference_metadata_text()
        except Exception as exc:
            self._metadata_text_cache = f"Metadata introspection failed: {type(exc).__name__}: {exc}"
        return self._metadata_text_cache

    def _build_reference_metadata_text(self) -> str:
        import importlib.util
        import torch

        ref_path = Path(self._ref_path)
        ref_dir = str(ref_path.parent)
        if ref_dir not in sys.path:
            sys.path.insert(0, ref_dir)
        spec = importlib.util.spec_from_file_location("_hlsl_prompt_ref", self._ref_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reference module from {self._ref_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_hlsl_prompt_ref"] = mod
        spec.loader.exec_module(mod)

        init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        model = mod.Model(*init_inputs)
        dtype = torch.float16 if self._cfg.precision == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = model.to(device=device).to(dtype=dtype).eval()
        except Exception:
            model = model.eval()

        raw_inputs = list(mod.get_inputs()) if hasattr(mod, "get_inputs") else []
        inputs: list[Any] = []
        for value in raw_inputs:
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=device)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                inputs.append(tensor.contiguous())
            else:
                inputs.append(value)

        with torch.no_grad():
            outputs_raw = model(*inputs)
        if isinstance(outputs_raw, torch.Tensor):
            outputs = [outputs_raw]
        elif isinstance(outputs_raw, tuple):
            outputs = list(outputs_raw)
        elif isinstance(outputs_raw, list):
            outputs = outputs_raw
        else:
            outputs = [outputs_raw]

        lines = ["### SRV Inputs"]
        srv_index = 0
        for index, value in enumerate(inputs):
            if isinstance(value, torch.Tensor):
                lines.append(
                    f"- input_{index}: register(t{srv_index}), dtype={str(value.dtype).replace('torch.', '')}, "
                    f"shape={tuple(value.shape)}, numel={value.numel()}"
                )
            else:
                lines.append(f"- input_{index}: non-tensor value {type(value).__name__}; HLSL backend does not support this yet")
            srv_index += 1

        state_items = [(name, tensor) for name, tensor in model.state_dict().items() if isinstance(tensor, torch.Tensor)]
        if state_items:
            lines.append("### SRV Model State")
            for state_index, (name, tensor) in enumerate(state_items):
                lines.append(
                    f"- state_{state_index} ({name}): register(t{srv_index}), dtype={str(tensor.dtype).replace('torch.', '')}, "
                    f"shape={tuple(tensor.shape)}, numel={tensor.numel()}"
                )
                srv_index += 1
        else:
            lines.append("### SRV Model State\n- none")

        lines.append("### UAV Outputs")
        for index, value in enumerate(outputs):
            if isinstance(value, torch.Tensor):
                lines.append(
                    f"- output_{index}: register(u{index}), dtype={str(value.dtype).replace('torch.', '')}, "
                    f"shape={tuple(value.shape)}, numel={value.numel()}"
                )
            else:
                lines.append(f"- output_{index}: non-tensor value {type(value).__name__}; HLSL backend does not support this yet")
        return "\n".join(lines)