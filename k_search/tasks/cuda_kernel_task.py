"""CUDA Kernel Task for K-Search.

This task generates multi-file CUDA kernels (kernel.h, kernel.cu, main.cpp)
compiled via torch.utils.cpp_extension, evaluated against a PyTorch reference model.

Unlike KernelBenchTask (which produces single Python files with inline CUDA),
this task uses the same multi-file XML format as FlashInferBenchTask.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

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
class CudaKernelTaskConfig:
    gpu: str = "H100"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    precision: str = "fp16"
    rtol: float = 1e-2
    atol: float = 1e-2
    max_failure_excerpt_chars: int = 4000


class CudaKernelTask:
    """Task for optimizing a PyTorch module with custom multi-file CUDA kernels.

    The reference is a Python file defining:
      - A class `Model` (or `ConvBlock`, etc. aliased to `Model`)
      - `get_inputs()` -> list of input tensors
      - `get_init_inputs()` -> list of constructor args

    Generated code uses the standard XML format:
      - kernel.h: declarations
      - kernel.cu: CUDA kernels
      - main.cpp: host code with PYBIND11_MODULE exposing a `run()` function

    The `run()` function receives the same tensors as `Model.forward()` and returns
    a list of output tensors matching the reference.
    """

    def __init__(
        self,
        *,
        ref_path: str,
        gpu: str = "H100",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timeout: int = 300,
        precision: str = "fp16",
        rtol: float = 1e-2,
        atol: float = 1e-2,
        artifacts_dir: str | None = None,
        name: str | None = None,
        enable_ncu_profiling: bool = False,
        verbose: bool = False,
    ) -> None:
        self._ref_path = str(Path(ref_path).resolve())
        if not Path(self._ref_path).exists():
            raise FileNotFoundError(f"Reference file not found: {self._ref_path}")

        self._cfg = CudaKernelTaskConfig(
            gpu=str(gpu),
            num_correct_trials=int(num_correct_trials),
            num_perf_trials=int(num_perf_trials),
            timeout=int(timeout),
            precision=str(precision),
            rtol=float(rtol),
            atol=float(atol),
        )
        self._name = str(name or Path(self._ref_path).stem)
        self._artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._enable_ncu_profiling = bool(enable_ncu_profiling)
        self._verbose = bool(verbose)
        self._solutions: dict[str, Solution] = {}

        # Feedback state for world-model prompts
        self._last_round_trace_logs: str = ""
        self._last_round_passed: bool = False
        self._last_round_summary: str = ""

        # Load reference code
        self._ref_code = Path(self._ref_path).read_text()
        print(f"[{self._name}] Loaded reference: {self._ref_path}")

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        return f"""# CUDA Kernel Optimization Task

**Reference Module**: {Path(self._ref_path).name}
**Target GPU**: {self._cfg.gpu}
**Precision**: {self._cfg.precision}

## Objective
Optimize the following PyTorch reference implementation by writing custom CUDA kernels.
Your implementation must produce numerically equivalent outputs (rtol={self._cfg.rtol}, atol={self._cfg.atol}).

## Reference Implementation

```python
{self._ref_code}
```

## Your Task
Write optimized CUDA code that replaces the `forward()` computation.
The `run()` function in main.cpp will receive the same input tensors as `Model.forward()`
and must return a list containing the same output tensor(s).

The model will be instantiated with `get_init_inputs()` args and run with `get_inputs()` tensors.
Your CUDA code should handle the specific shapes/dtypes from those functions.
"""

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        from k_search.tasks.prompts import _cuda_xml_and_guidelines_block
        return _cuda_xml_and_guidelines_block(target_gpu=str(target_gpu or self._cfg.gpu)).strip()

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
        """Extract kernel.cu from raw XML for world-model prompt brevity."""
        if not isinstance(raw, str):
            return str(raw or "")
        m = re.search(r'<cuda_file\s+name="kernel\.cu"\s*>([\s\S]*?)</cuda_file>', raw)
        if m:
            return m.group(1).strip()
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "cuda_kernel",
            "ref_path": self._ref_path,
            "gpu": self._cfg.gpu,
            "precision": self._cfg.precision,
            "num_correct_trials": self._cfg.num_correct_trials,
            "num_perf_trials": self._cfg.num_perf_trials,
        }

    def run_final_evaluation(
        self, *, solutions: list[Solution], config: Any = None,
        dump_traces: bool = False, workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = {}
        for sol in solutions:
            ev = self.run_benchmark(solution=sol, config=config, dump_traces=dump_traces, round_num=None)
            results[sol.name] = ev.to_dict(include_log_excerpt=True)
        return results

    # ---- World-model prompt hooks ----

    def has_last_round_feedback_trace(self) -> bool:
        return bool(self._last_round_trace_logs)

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs

    def get_last_round_passed_count(self) -> int:
        return 1 if self._last_round_passed else 0

    def get_last_round_total_workloads(self) -> int:
        return 1

    # ---- Evaluation ----

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        """Compile and evaluate the CUDA solution against the reference."""
        # Extract source files
        sources = {sf.path: sf.content for sf in (solution.sources or [])}
        kernel_h = sources.get("kernel.h", "")
        kernel_cu = sources.get("kernel.cu", "")
        main_cpp = sources.get("main.cpp", "")

        if not kernel_cu.strip() or not main_cpp.strip():
            msg = "Missing kernel.cu or main.cpp in solution sources"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            self._last_round_summary = f"FAILED: {msg}"
            return EvalResult(status="failed", log_excerpt=msg)

        # Write files to temp directory and run evaluator subprocess
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="cuda_kernel_eval_")
            # Write source files
            Path(tmp_dir, "kernel.h").write_text(kernel_h)
            Path(tmp_dir, "kernel.cu").write_text(kernel_cu)
            Path(tmp_dir, "main.cpp").write_text(main_cpp)

            # Run the evaluator script
            evaluator_path = Path(__file__).parent / "cuda_kernel_eval.py"
            cmd = [
                sys.executable, str(evaluator_path),
                "--ref-path", self._ref_path,
                "--kernel-dir", tmp_dir,
                "--num-correct-trials", str(self._cfg.num_correct_trials),
                "--num-perf-trials", str(self._cfg.num_perf_trials),
                "--precision", self._cfg.precision,
                "--rtol", str(self._cfg.rtol),
                "--atol", str(self._cfg.atol),
            ]

            env = os.environ.copy()
            # Add reference file's directory to PYTHONPATH for local imports
            ref_dir = str(Path(self._ref_path).parent)
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{ref_dir}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = ref_dir

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout,
                env=env,
            )

            stdout = result.stdout
            stderr = result.stderr
            output = (stdout + "\n" + stderr).strip()

            if result.returncode != 0:
                excerpt = output[-self._cfg.max_failure_excerpt_chars:]
                self._last_round_trace_logs = excerpt
                self._last_round_passed = False
                self._last_round_summary = "FAILED: compilation or runtime error"
                return EvalResult(status="failed", log_excerpt=excerpt)

            # Parse evaluator JSON output
            eval_result = self._parse_eval_output(stdout)

            # Run ncu profiling if enabled and kernel passed
            if self._enable_ncu_profiling and eval_result.is_passed():
                eval_result = self._run_ncu_profiling(eval_result, tmp_dir, env)

            # Verbose output
            if self._verbose:
                self._print_debug_report(eval_result, kernel_cu)

            return eval_result

        except subprocess.TimeoutExpired:
            msg = f"Evaluation timed out after {self._cfg.timeout}s"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            self._last_round_summary = f"FAILED: {msg}"
            return EvalResult(status="failed", log_excerpt=msg)
        except Exception as e:
            msg = f"Evaluation error: {type(e).__name__}: {e}"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            self._last_round_summary = f"FAILED: {msg}"
            return EvalResult(status="failed", log_excerpt=msg)
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_ncu_profiling(self, eval_result: EvalResult, tmp_dir: str, env: dict) -> EvalResult:
        """Run ncu profiling on a passed kernel and attach metrics to the EvalResult."""
        try:
            from k_search.utils.ncu_profiler import (
                NCU_AVAILABLE,
                metrics_to_dict,
                run_ncu_profile,
            )

            if not NCU_AVAILABLE:
                return eval_result

            evaluator_path = Path(__file__).parent / "cuda_kernel_eval.py"
            profile_cmd = [
                sys.executable, str(evaluator_path),
                "--ref-path", self._ref_path,
                "--kernel-dir", tmp_dir,
                "--precision", self._cfg.precision,
                "--profile-only",
            ]

            ncu_metrics = run_ncu_profile(profile_cmd, timeout=120, verbose=self._verbose)
            if ncu_metrics is not None:
                eval_result.profiler_metrics = metrics_to_dict(ncu_metrics)
            elif self._verbose:
                print("[ncu] No metrics parsed from ncu output")
        except Exception:
            # Profiling is best-effort; never fail the eval
            pass
        return eval_result

    def _print_debug_report(self, eval_result: EvalResult, kernel_cu: str) -> None:
        """Print kernel source and profiler report when debug mode is enabled."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[DEBUG] [{self._name}] Eval status: {eval_result.status}")
        print(f"{sep}")

        # Print kernel source
        print(f"\n--- kernel.cu ---")
        print(kernel_cu)

        # Print profiler report
        if eval_result.profiler_metrics:
            print(f"\n--- NCU Profiler Report ---")
            for line in eval_result.profiler_summary_lines():
                print(f"  {line}")
        elif self._enable_ncu_profiling and eval_result.is_passed():
            print(f"\n--- NCU Profiler Report ---")
            print("  (no profiler data collected)")

        print(f"{sep}\n")

    def _parse_eval_output(self, stdout: str) -> EvalResult:
        """Parse JSON output from cuda_kernel_eval.py."""
        import json

        # Find the last JSON line in output
        json_line = None
        for line in stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                json_line = line

        if not json_line:
            msg = f"No JSON output from evaluator. stdout:\n{stdout[-2000:]}"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            return EvalResult(status="failed", log_excerpt=msg)

        try:
            data = json.loads(json_line)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON from evaluator: {e}\n{json_line[:500]}"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            return EvalResult(status="failed", log_excerpt=msg)

        compiled = data.get("compiled", False)
        correct = data.get("correct", False)
        latency_ms = data.get("latency_ms")
        ref_latency_ms = data.get("ref_latency_ms")
        error = data.get("error", "")

        if not compiled:
            msg = f"Compilation failed: {error}"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            self._last_round_summary = f"FAILED: {msg}"
            return EvalResult(status="failed", log_excerpt=msg)

        if not correct:
            msg = f"Correctness check failed: {error}"
            self._last_round_trace_logs = msg
            self._last_round_passed = False
            self._last_round_summary = f"FAILED: {msg}"
            return EvalResult(status="failed", log_excerpt=msg)

        # Compute speedup
        speedup = None
        if latency_ms and ref_latency_ms and latency_ms > 0:
            speedup = ref_latency_ms / latency_ms

        summary = f"PASSED: latency={latency_ms:.3f}ms ref={ref_latency_ms:.3f}ms speedup={speedup:.2f}x" if speedup else "PASSED"
        self._last_round_trace_logs = summary
        self._last_round_passed = True
        self._last_round_summary = summary

        return EvalResult(
            status="passed",
            latency_ms=latency_ms,
            reference_latency_ms=ref_latency_ms,
            speedup_factor=speedup,
            log_excerpt=summary,
        )
