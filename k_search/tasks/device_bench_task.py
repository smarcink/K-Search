"""Unified Device Bench Task for K-Search.

Single task class that handles both NVIDIA CUDA and Intel XPU (and any
future PyTorch accelerator backend).  Device-specific behaviour is driven
by the ``--device`` argument:

  - ``cuda:N``  → NVIDIA-specific prompts / optimization hints
  - ``xpu:N``   → Intel XPU-specific prompts / optimization hints

Old ``xpu_bench`` and ``cuda_bench`` task names are kept as CLI aliases that
simply resolve to this class.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backend_of(device: str) -> str:
    """Extract backend name from device string: 'cuda:0' → 'cuda'."""
    return device.split(":")[0]


def _is_xpu(device: str) -> bool:
    return _backend_of(device) == "xpu"


def _is_cuda(device: str) -> bool:
    return _backend_of(device) == "cuda"


# ---------------------------------------------------------------------------
# Device-specific prompt snippets
# ---------------------------------------------------------------------------

_XPU_DEVICE_NOTES = """\
## Important Notes — Intel XPU
- The code runs on Intel XPU, NOT NVIDIA CUDA. Do NOT use any CUDA-specific APIs.
- Use `torch.xpu` for device operations if needed.
- Triton kernels should use standard Triton primitives — they will be compiled for the Intel XPU backend.
- Do NOT reference cuDNN, cuBLAS, CUTLASS, or any NVIDIA-specific libraries.
"""

_XPU_OPT_GUIDELINES = """\
## Optimization Guidelines — Intel XPU
Before modifying the code, analyze:
1. **Performance bottlenecks**: Identify memory access patterns, kernel launch overhead, unnecessary operations
2. **Correctness risks**: Check numerical stability, edge cases, data type handling
3. **XPU utilization**: Consider sub-group sizes (16/32), Xe Matrix Extensions (XMX), memory coalescing

Then implement your optimized version.
"""

_CUDA_DEVICE_NOTES = """\
## Important Notes — NVIDIA CUDA
- Target NVIDIA GPU with tensor cores (H100/RTX 5090 class).
- Use tl.dot() for matrix operations to leverage tensor cores.
- Standard Triton primitives work on CUDA: tl.load, tl.store, tl.dot, tl.program_id, etc.
- You may use torch.cuda for device operations.
"""

_CUDA_OPT_GUIDELINES = """\
## Optimization Guidelines — NVIDIA CUDA
Before modifying the code, analyze:
1. **Performance bottlenecks**: Identify memory access patterns, kernel launch overhead, unnecessary operations
2. **Correctness risks**: Check numerical stability, edge cases, data type handling
3. **GPU utilization**: Consider warp sizes (32), tensor core shapes (16x16x16), memory coalescing, shared memory usage

Then implement your optimized version.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceBenchTaskConfig:
    """Configuration for Device Bench task evaluation."""
    device: str = "cuda:0"
    precision: str = "fp16"
    num_correct_trials: int = 5
    num_perf_trials: int = 1000
    num_warmup: int = 100
    timeout: int = 300
    max_failure_excerpt_chars: int = 4000


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class DeviceBenchTask:
    """Unified task wrapper for optimizing PyTorch modules on any accelerator.

    Accepts any local .py file defining Model, get_inputs(), and optionally
    get_init_inputs().  The LLM generates a ModelNew class with Triton
    kernels.  Evaluation runs in a subprocess via
    ``k_search/tasks/device_bench/run_and_eval.py``.
    """

    def __init__(
        self,
        *,
        ref_path: str,
        device: str = "cuda:0",
        precision: str = "fp16",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 10,
        timeout: int = 300,
        artifacts_dir: str | None = None,
        name: str | None = None,
        enable_profiling: bool = False,
        verbose: bool = False,
    ) -> None:
        self._ref_path = str(Path(ref_path).resolve())
        self._cfg = DeviceBenchTaskConfig(
            device=str(device),
            precision=str(precision),
            num_correct_trials=int(num_correct_trials),
            num_perf_trials=int(num_perf_trials),
            num_warmup=int(num_warmup),
            timeout=int(timeout),
        )
        self._name = str(name or Path(ref_path).stem)
        self._ksearch_artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._enable_profiling = bool(enable_profiling)
        self._verbose = bool(verbose)
        self._solutions: dict[str, Solution] = {}

        # Hardware profiler selected by device backend (NCU on CUDA; no-op on XPU).
        from k_search.utils.profiler import select_profiler
        self._profiler = select_profiler(self._cfg.device)

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
        print(f"[{self._name}] Loaded reference: {self._ref_path}  (device={self._cfg.device})")

    # ------------------------------------------------------------------
    # Device-conditional text helpers
    # ------------------------------------------------------------------

    def _device_display(self) -> str:
        if _is_xpu(self._cfg.device):
            return f"Intel XPU ({self._cfg.device})"
        if _is_cuda(self._cfg.device):
            return f"NVIDIA CUDA ({self._cfg.device})"
        return self._cfg.device

    def _device_notes(self) -> str:
        if _is_xpu(self._cfg.device):
            return _XPU_DEVICE_NOTES
        return _CUDA_DEVICE_NOTES

    def _device_optimization_guidelines(self) -> str:
        if _is_xpu(self._cfg.device):
            return _XPU_OPT_GUIDELINES
        return _CUDA_OPT_GUIDELINES

    def _kernel_description_line(self) -> str:
        if _is_xpu(self._cfg.device):
            return "The kernel will run on an Intel XPU device — use standard Triton primitives (tl.load, tl.store, tl.dot, etc.)."
        return "The kernel will run on an NVIDIA CUDA GPU — use standard Triton primitives (tl.load, tl.store, tl.dot, etc.)."

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
            f"{self._kernel_description_line()}"
        )

        return f"""# Device Bench Optimization Task

**Problem**: {self._problem_name}
**Target Device**: {self._device_display()}
**Backend**: {backend_display}
**Precision**: {self._cfg.precision}

## Objective
Optimize the following PyTorch reference implementation by writing a custom {backend_display} kernel
targeting the device hardware.
Your implementation should:
1. Match the reference implementation's correctness
2. Achieve better performance (lower latency) than torch.compile

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

{self._device_notes()}

## CRITICAL: No Silent Fallbacks
- Your `ModelNew.forward()` MUST always execute your custom kernel code path.
- Do NOT wrap your kernel call in `try/except` that falls back to the reference PyTorch
  implementation (e.g. `nn.Conv2d`, `nn.Linear`, etc.) when the kernel fails.
  If your kernel crashes or produces errors, the evaluation MUST see that failure —
  hiding it behind a fallback means the kernel is never actually tested.
- Do NOT copy the reference Model's forward() as a "fallback" path that gets used
  when your kernel has issues. If you need a fallback for unsupported configurations
  (e.g. integer_forward mode), that is fine, but the PRIMARY code path that is being
  benchmarked must always run your optimized kernel.

## Format
{format_text}

## Evaluation
Your implementation will be evaluated for:
1. **Correctness**: Output must match reference within tolerance
2. **Performance**: Speedup over torch.compile (must beat it to be useful)
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

        prompt += "\n\n" + self._device_optimization_guidelines()
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

        backend_tag = _backend_of(self._cfg.device).upper()
        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=spec,
            sources=sources,
            description=f"{backend_tag} Device Bench {self._problem_name} optimization",
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

        # Detect silent-fallback antipattern
        if re.search(
            r'except\s.*:\s*\n\s*return\s+self\._fallback',
            code,
        ):
            print(f"[{self._name}] WARNING: Generated code contains try/except fallback to reference impl — "
                  f"kernel errors will be silently hidden. Consider removing the fallback.")

        # Auto-fix missing triton imports
        if "@triton.jit" in code or "tl." in code:
            if "import triton" not in code:
                code = "import triton\nimport triton.language as tl\n" + code

        kernel_src_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                kernel_src_path = tmp.name

            # Save debug kernel
            try:
                if self._ksearch_artifacts_dir:
                    _dbg_dir = Path(self._ksearch_artifacts_dir) / self._name / "debug_kernels"
                else:
                    _dbg_dir = Path("debug_kernels") / self._name
                _dbg_dir.mkdir(parents=True, exist_ok=True)
                _dbg_path = _dbg_dir / f"round_{round_num or 0:03d}.py"
                _dbg_path.write_text(code)
                print(f"[{self._name}] Saved debug kernel: {_dbg_path}")
            except Exception:
                pass

            # -----------------------------------------------------------
            # Pre-flight check: syntax + import validation
            # -----------------------------------------------------------
            _ref_dir = str(Path(self._ref_path).resolve().parent)
            preflight_code = (
                "import sys\n"
                f"sys.path.insert(0, '{_ref_dir}')\n"
                "try:\n"
                f"    compile(open('{kernel_src_path}').read(), '{kernel_src_path}', 'exec')\n"
                "except SyntaxError as e:\n"
                "    print(f'SyntaxError: {e}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "import importlib.util\n"
                f"spec = importlib.util.spec_from_file_location('_preflight', '{kernel_src_path}')\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "try:\n"
                "    spec.loader.exec_module(mod)\n"
                "except Exception as e:\n"
                "    print(f'Import error: {type(e).__name__}: {e}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "if not hasattr(mod, 'ModelNew'):\n"
                "    print('Error: ModelNew class not defined', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "print('preflight OK')\n"
            )
            preflight_result = subprocess.run(
                [sys.executable, "-c", preflight_code],
                capture_output=True, text=True, timeout=120,
                cwd=str(Path(__file__).parent.parent.parent),
                env=os.environ.copy(),
            )
            if preflight_result.returncode != 0:
                preflight_err = (preflight_result.stderr or preflight_result.stdout or "unknown error").strip()
                return self._failed_eval(
                    f"Pre-flight check failed (code does not import cleanly):\n{preflight_err[-2000:]}",
                    round_num,
                )

            repo_root = Path(__file__).parent.parent.parent
            if not (repo_root / "k_search").exists():
                repo_root = Path.cwd()
                while repo_root.parent != repo_root:
                    if (repo_root / "k_search").exists():
                        break
                    repo_root = repo_root.parent

            cmd = [
                sys.executable,
                "-u",
                "-m", "k_search.tasks.device_bench.run_and_eval",
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
            _ref_dir = str(Path(self._ref_path).resolve().parent)
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{_ref_dir}:{src_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = f"{_ref_dir}:{src_path}"

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
            if kernel_src_path:
                try: os.unlink(kernel_src_path)
                except Exception: pass
            return self._failed_eval(f"Evaluation timed out after {self._cfg.timeout} seconds", round_num)
        except Exception as e:
            if kernel_src_path:
                try: os.unlink(kernel_src_path)
                except Exception: pass
            return self._failed_eval(f"Evaluation error: {type(e).__name__}: {e}", round_num)

        # NB: kernel_src_path is intentionally kept alive past this point so that
        # the optional profiling pass can re-import the same source. It is unlinked
        # in the try/finally wrapping the parsing + profiling block below.
        try:

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                excerpt = self._extract_error_excerpt(stdout, stderr)
                rc_info = f"(exit code {result.returncode})"
                if result.returncode < 0:
                    import signal as _sig
                    try:
                        sig_name = _sig.Signals(-result.returncode).name
                        rc_info = f"(killed by {sig_name}, exit code {result.returncode})"
                    except (ValueError, AttributeError):
                        rc_info = f"(killed by signal {-result.returncode})"
                return self._failed_eval(f"Evaluation failed {rc_info}:\n{excerpt}", round_num)

            # Parse structured output
            speedup_eager = self._extract_metric(stdout, "Speedup over eager:", r"([0-9.]+)x")
            speedup_compile = self._extract_metric(stdout, "Speedup over torch.compile:", r"([0-9.]+)x")
            kernel_time = self._extract_metric(stdout, "Custom Kernel exec time:", r"([0-9.]+) ms")
            ref_eager_time = self._extract_metric(stdout, "PyTorch Reference Eager exec time:", r"([0-9.]+) ms")
            compile_time = self._extract_metric(stdout, "torch.compile exec time:", r"([0-9.]+) ms")

            # Use speedup over torch.compile as the primary score
            primary_speedup = speedup_compile if speedup_compile and speedup_compile > 0 else speedup_eager
            primary_score_name = "speedup_over_compile" if (speedup_compile and speedup_compile > 0) else "speedup_over_eager"

            if speedup_eager and speedup_eager > 0:
                # Build informative trace log for LLM prompts
                trace_parts = [f"Custom kernel: {kernel_time:.4f} ms"]
                if ref_eager_time:
                    trace_parts.append(f"PyTorch eager: {ref_eager_time:.4f} ms (speedup: {speedup_eager:.2f}x)")
                if compile_time and speedup_compile:
                    trace_parts.append(f"torch.compile: {compile_time:.4f} ms (speedup: {speedup_compile:.2f}x)")
                    if speedup_compile < 1.0:
                        trace_parts.append(f"*** Your kernel is SLOWER than torch.compile ({speedup_compile:.2f}x) — you need to beat {compile_time:.4f} ms")
                self._last_round_trace_logs_for_prompt = "\n".join(trace_parts)
                self._last_round_passed_count = 1
                self._last_round_total_workloads = 1

                log_line = f"Speedup: {speedup_eager:.2f}x over eager"
                if speedup_compile:
                    log_line += f", {speedup_compile:.2f}x over torch.compile"

                er = EvalResult(
                    status="passed",
                    latency_ms=kernel_time,
                    reference_latency_ms=ref_eager_time,
                    mean_vs_baseline_factor=None,
                    speedup_factor=primary_speedup,
                    log_excerpt=log_line,
                    metrics={
                        "score_name": primary_score_name,
                        "score": float(primary_speedup) if primary_speedup else None,
                        "speedup_over_eager": speedup_eager,
                        "speedup_over_compile": speedup_compile,
                        "kernel_time_ms": kernel_time,
                        "ref_eager_time_ms": ref_eager_time,
                    },
                )

                # Optional hardware profiling pass (best-effort, never fails the eval).
                if self._enable_profiling:
                    er = self._run_profiling(er, kernel_src_path, env, repo_root)

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
                        "score_name": "speedup_over_compile",
                        "score": None,
                    },
                )
                self._print_summary(round_num, er, passed=False)
                return er
        finally:
            if kernel_src_path:
                try: os.unlink(kernel_src_path)
                except Exception: pass

    def _run_profiling(self, eval_result: "EvalResult", kernel_src_path: str, env: dict, repo_root: Path) -> "EvalResult":
        """Run hardware profiling on a passed kernel and attach metrics to EvalResult."""
        try:
            if not self._profiler.available():
                if self._verbose:
                    print(f"[{self._name}] [profiler] backend '{self._profiler.name}' not available; skipping")
                return eval_result

            profile_cmd = [
                sys.executable,
                "-u",
                "-m", "k_search.tasks.device_bench.run_and_eval",
                f"--ref-path={self._ref_path}",
                f"--kernel-src-path={kernel_src_path}",
                f"--device={self._cfg.device}",
                f"--precision={self._cfg.precision}",
                "--profile-only",
            ]

            metrics_dict = self._profiler.run(profile_cmd, timeout=120, verbose=self._verbose)
            if metrics_dict is not None:
                metrics_dict["profiler_backend"] = self._profiler.name
                eval_result.profiler_metrics = metrics_dict
                if self._verbose:
                    print(f"[{self._name}] [profiler] attached {self._profiler.name} metrics "
                          f"({len(metrics_dict.get('kernels', []))} kernels)")
            elif self._verbose:
                print(f"[{self._name}] [profiler] no metrics parsed from {self._profiler.name} output")
        except Exception as e:
            if self._verbose:
                print(f"[{self._name}] [profiler] error (non-fatal): {type(e).__name__}: {e}")
        return eval_result

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
                "speedup_over_eager": er.metrics.get("speedup_over_eager") if isinstance(er.metrics, dict) else None,
                "speedup_over_compile": er.metrics.get("speedup_over_compile") if isinstance(er.metrics, dict) else None,
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
            "task_type": "device_bench",
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
                "score_name": "speedup_over_compile",
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
            metrics = er.metrics or {}
            sp_eager = metrics.get("speedup_over_eager")
            sp_compile = metrics.get("speedup_over_compile")
            parts = [
                f"[{self._name}] Round {rn}: status={status}",
                f"speedup={speedup:.2f}x",
                f"latency={latency:.4f}ms",
            ]
            if sp_eager:
                parts.append(f"vs_eager={sp_eager:.2f}x")
            if sp_compile:
                parts.append(f"vs_compile={sp_compile:.2f}x")
            parts.append(f"device={self._cfg.device}")
            self._last_round_summary_line = " | ".join(parts)
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


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

XpuBenchTask = DeviceBenchTask
CudaBenchTask = DeviceBenchTask
