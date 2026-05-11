"""XPU Bench Task implementation for K-Search.

Evaluates Triton kernels on Intel XPU GPUs using a local reference .py file
that follows the Model/ModelNew + get_inputs()/get_init_inputs() convention.
"""

from __future__ import annotations

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
class XpuBenchTaskConfig:
    """Configuration for XPU Bench task evaluation."""
    device: str = "xpu:0"
    precision: str = "fp16"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    num_warmup: int = 10
    timeout: int = 300
    max_failure_excerpt_chars: int = 4000


class XpuBenchTask:
    """Task wrapper for optimizing PyTorch modules on Intel XPU with K-Search.

    Accepts any local .py file defining Model, get_inputs(), and get_init_inputs().
    The LLM generates a ModelNew class with Triton kernels targeting Intel XPU.
    Evaluation runs in a subprocess via k_search/tasks/xpu_bench/run_and_eval.py.
    """

    def __init__(
        self,
        *,
        ref_path: str,
        device: str = "xpu:0",
        precision: str = "fp16",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 10,
        timeout: int = 300,
        artifacts_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        self._ref_path = str(Path(ref_path).resolve())
        self._cfg = XpuBenchTaskConfig(
            device=str(device),
            precision=str(precision),
            num_correct_trials=int(num_correct_trials),
            num_perf_trials=int(num_perf_trials),
            num_warmup=int(num_warmup),
            timeout=int(timeout),
        )
        self._name = str(name or Path(ref_path).stem)
        self._ksearch_artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._solutions: dict[str, Solution] = {}

        # Feedback state (updated after each eval round)
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_summary_line: str = ""

        # Load reference code
        self._ref_code: str = ""
        self._problem_name: str = ""
        self._load_reference()

    def _load_reference(self) -> None:
        ref_path = Path(self._ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self._ref_path}")
        self._ref_code = ref_path.read_text()
        self._problem_name = ref_path.stem
        print(f"[{self._name}] Loaded reference: {self._ref_path}")

    # ------------------------------------------------------------------
    # Task protocol: identity & definition
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        backend_display = "Triton"
        format_text = (
            "Your code should define a `ModelNew` class with the same interface as `Model`.\n"
            "You can include Triton kernels using the `@triton.jit` decorator and launch them appropriately.\n"
            "The kernel will run on an Intel XPU device — use standard Triton primitives (tl.load, tl.store, tl.dot, etc.)."
        )

        return f"""# XPU Bench Optimization Task

**Problem**: {self._problem_name}
**Target Device**: Intel XPU ({self._cfg.device})
**Backend**: {backend_display}
**Precision**: {self._cfg.precision}

## Objective
Optimize the following PyTorch reference implementation by writing a custom {backend_display} kernel
targeting Intel XPU hardware.
Your implementation should:
1. Match the reference implementation's correctness
2. Achieve better performance (lower latency)

## Reference Implementation (class Model)

```python
{self._ref_code}
```

## Your Task
Create an optimized implementation in a class called `ModelNew` that:
- Inherits from the same base class as Model
- Implements the same forward() method signature
- Uses custom {backend_display} kernels for better performance
- Maintains numerical correctness (within tolerance)

## Important Notes
- The code runs on Intel XPU, NOT NVIDIA CUDA. Do NOT use any CUDA-specific APIs.
- Use `torch.xpu` for device operations if needed.
- Triton kernels should use standard Triton primitives — they will be compiled for the Intel XPU backend.
- Do NOT reference cuDNN, cuBLAS, CUTLASS, or any NVIDIA-specific libraries.

## Format
{format_text}

## Evaluation
Your implementation will be evaluated for:
1. **Correctness**: Output must match reference within tolerance
2. **Performance**: Speedup over PyTorch eager mode and torch.compile
"""

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return f"{self.get_definition_text(language)}\n\nTarget GPU: {target_gpu}\n"

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
        prompt = self.get_definition_text(language)
        prompt += f"\n\nTarget GPU: {target_gpu}\n"

        current_code_str = str(current_code or "").strip()
        prompt += f"\n\n## Current Implementation\n```python\n{current_code_str}\n```"

        if previous_round_summary:
            prompt += "\n\n## Previous Round Summary\n" + str(previous_round_summary).strip()

        if trace_logs:
            prompt += "\n\n## Evaluation Feedback\n" + str(trace_logs).strip()

        if current_best:
            prompt += "\n\n## Current Best Performance\n" + str(current_best).strip()

        prompt += """

## Optimization Guidelines
Before modifying the code, analyze:
1. **Performance bottlenecks**: Identify memory access patterns, kernel launch overhead, unnecessary operations
2. **Correctness risks**: Check numerical stability, edge cases, data type handling
3. **XPU utilization**: Consider sub-group sizes (16/32), Xe Matrix Extensions (XMX), memory coalescing

Then implement your optimized version.
"""
        return prompt

    # ------------------------------------------------------------------
    # Task protocol: solution management
    # ------------------------------------------------------------------

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
        code_text = str(cleaned_code or raw_code or "")
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        sol_name = f"{safe_model_name}_{self._name}_r{round_num}"

        sources = [SourceFile(path="model_new.py", content=code_text)]
        spec = BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=[str(target_gpu)],
            entry_point="model_new.py::ModelNew",
        )

        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=spec,
            sources=sources,
            description=f"XPU Bench {self._problem_name} optimization",
        )

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions[name]
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=self.name,
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            if sol.definition != self.name:
                return None
            self._solutions[sol.name] = sol
            return sol
        except (FileNotFoundError, Exception):
            return None

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    # ------------------------------------------------------------------
    # Task protocol: evaluation
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        entry_src = solution.get_entry_source()
        if not entry_src:
            return self._failed_eval("No entry source found in solution", round_num)

        code = entry_src.content

        # Ensure ModelNew class exists
        if "class ModelNew" not in code:
            code = code.replace("class Model(", "class ModelNew(")
            code = code.replace("class Model:", "class ModelNew:")
            code = re.sub(r"super\(Model,\s*self\)", "super()", code)
            code = re.sub(r"super\(Model,\s*cls\)", "super()", code)

        kernel_src_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                kernel_src_path = tmp.name

            repo_root = Path(__file__).parent.parent.parent
            if not (repo_root / "k_search").exists():
                repo_root = Path.cwd()
                while repo_root.parent != repo_root:
                    if (repo_root / "k_search").exists():
                        break
                    repo_root = repo_root.parent

            cmd = [
                sys.executable,
                "-m", "k_search.tasks.xpu_bench.run_and_eval",
                f"--ref-path={self._ref_path}",
                f"--kernel-src-path={kernel_src_path}",
                f"--device={self._cfg.device}",
                f"--precision={self._cfg.precision}",
                f"--num-correct-trials={self._cfg.num_correct_trials}",
                f"--num-perf-trials={self._cfg.num_perf_trials}",
                f"--num-warmup={self._cfg.num_warmup}",
            ]

            env = os.environ.copy()
            src_path = str(repo_root)
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = src_path

            print(f"[{self._name}] Running evaluation: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout + 60,
                cwd=str(repo_root),
                env=env,
            )

        except subprocess.TimeoutExpired:
            return self._failed_eval(f"Evaluation timed out after {self._cfg.timeout} seconds", round_num)
        except Exception as e:
            return self._failed_eval(f"Evaluation error: {type(e).__name__}: {e}", round_num)
        finally:
            if kernel_src_path:
                try:
                    os.unlink(kernel_src_path)
                except Exception:
                    pass

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            excerpt = self._extract_error_excerpt(stdout, stderr)
            return self._failed_eval(f"Evaluation failed:\n{excerpt}", round_num)

        # Parse structured output
        speedup_eager = self._extract_metric(stdout, "Speedup over eager:", r"([0-9.]+)x")
        speedup_compile = self._extract_metric(stdout, "Speedup over torch.compile:", r"([0-9.]+)x")
        kernel_time = self._extract_metric(stdout, "Custom Kernel exec time:", r"([0-9.]+) ms")
        ref_eager_time = self._extract_metric(stdout, "PyTorch Reference Eager exec time:", r"([0-9.]+) ms")

        if speedup_eager and speedup_eager > 0:
            self._last_round_trace_logs_for_prompt = f"Speedup: {speedup_eager:.2f}x over eager"
            self._last_round_passed_count = 1
            self._last_round_total_workloads = 1

            er = EvalResult(
                status="passed",
                latency_ms=kernel_time,
                reference_latency_ms=ref_eager_time,
                mean_vs_baseline_factor=None,
                speedup_factor=speedup_eager,
                log_excerpt=f"Speedup: {speedup_eager:.2f}x",
                metrics={
                    "score_name": "speedup_over_eager",
                    "score": float(speedup_eager),
                    "speedup_over_eager": speedup_eager,
                    "speedup_over_compile": speedup_compile,
                    "kernel_time_ms": kernel_time,
                    "ref_eager_time_ms": ref_eager_time,
                },
            )
            self._print_summary(round_num, er, passed=True)
            return er
        else:
            excerpt = self._extract_eval_excerpt(stdout)
            self._last_round_trace_logs_for_prompt = excerpt
            self._last_round_passed_count = 0
            self._last_round_total_workloads = 1

            er = EvalResult(
                status="failed",
                latency_ms=None,
                reference_latency_ms=None,
                mean_vs_baseline_factor=None,
                speedup_factor=None,
                log_excerpt=excerpt,
                metrics={
                    "score_name": "speedup_over_eager",
                    "score": None,
                },
            )
            self._print_summary(round_num, er, passed=False)
            return er

    # ------------------------------------------------------------------
    # Task protocol: feedback for prompts
    # ------------------------------------------------------------------

    def has_last_round_feedback_trace(self) -> bool:
        return bool(self._last_round_trace_logs_for_prompt)

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs_for_prompt

    def get_last_round_passed_count(self) -> int:
        return self._last_round_passed_count

    def get_last_round_total_workloads(self) -> int:
        return self._last_round_total_workloads

    # ------------------------------------------------------------------
    # Task protocol: final evaluation & logging
    # ------------------------------------------------------------------

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = []
        for sol in solutions:
            if not sol:
                continue
            er = self.run_benchmark(solution=sol, dump_traces=False, round_num=None)
            results.append({
                "solution": sol.name,
                "status": er.status,
                "speedup_over_eager": er.speedup_factor,
                "latency_ms": er.latency_ms,
                "score_name": er.metrics.get("score_name") if isinstance(er.metrics, dict) else None,
                "score": er.metrics.get("score") if isinstance(er.metrics, dict) else None,
            })

        return {
            "task": self._name,
            "problem": self._problem_name,
            "device": self._cfg.device,
            "precision": self._cfg.precision,
            "solutions": results,
        }

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "xpu_bench",
            "task_name": self._name,
            "problem_name": self._problem_name,
            "ref_path": self._ref_path,
            "device": self._cfg.device,
            "precision": self._cfg.precision,
            "num_correct_trials": self._cfg.num_correct_trials,
            "num_perf_trials": self._cfg.num_perf_trials,
            "timeout": self._cfg.timeout,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metric(self, text: str, line_pattern: str, value_regex: str) -> float | None:
        for line in text.split("\n"):
            if line_pattern in line:
                match = re.search(value_regex, line)
                if match:
                    return float(match.group(1))
        return None

    def _extract_error_excerpt(self, stdout: str, stderr: str) -> str:
        max_chars = self._cfg.max_failure_excerpt_chars
        if stderr.strip():
            excerpt = stderr[-max_chars:] if len(stderr) > max_chars else stderr
            return f"[stderr]\n{excerpt}"
        excerpt = stdout[-max_chars:] if len(stdout) > max_chars else stdout
        return f"[stdout]\n{excerpt}"

    def _extract_eval_excerpt(self, stdout: str) -> str:
        max_chars = self._cfg.max_failure_excerpt_chars
        if "[Eval]" in stdout:
            eval_start = stdout.find("[Eval]")
            excerpt = stdout[eval_start:]
        else:
            excerpt = stdout
        if len(excerpt) > max_chars:
            excerpt = excerpt[-max_chars:]
        return excerpt

    def _failed_eval(self, message: str, round_num: int | None) -> EvalResult:
        self._last_round_trace_logs_for_prompt = message
        self._last_round_passed_count = 0
        self._last_round_total_workloads = 1

        er = EvalResult(
            status="failed",
            latency_ms=None,
            reference_latency_ms=None,
            mean_vs_baseline_factor=None,
            speedup_factor=None,
            log_excerpt=message,
            metrics={
                "score_name": "speedup_over_eager",
                "score": None,
            },
        )
        self._print_summary(round_num, er, passed=False)
        return er

    def _print_summary(self, round_num: int | None, er: EvalResult, passed: bool) -> None:
        rn = str(round_num) if round_num is not None else "?"
        status = "passed" if passed else "failed"

        if passed:
            speedup = er.speedup_factor or 0
            latency = er.latency_ms or 0
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: status={status} | "
                f"speedup={speedup:.2f}x | latency={latency:.4f}ms | "
                f"device={self._cfg.device}"
            )
        else:
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: status={status} | "
                f"device={self._cfg.device}"
            )

        print(self._last_round_summary_line, flush=True)

        if not passed:
            excerpt = er.log_excerpt or ""
            if excerpt:
                max_chars = self._cfg.max_failure_excerpt_chars
                if len(excerpt) > max_chars:
                    excerpt = excerpt[:max_chars] + "...<truncated>..."
                print(f"[{self._name}] Failure excerpt:\n{excerpt}", flush=True)
