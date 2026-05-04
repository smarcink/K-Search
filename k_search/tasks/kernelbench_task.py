"""KernelBench Task implementation for K-Search.

This task integrates KernelBench problems with K-Search optimization framework.
It uses the KernelBench evaluation logic from run_and_check.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import subprocess
import sys
import tempfile
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
class KernelBenchTaskConfig:
    """Configuration for KernelBench task evaluation."""
    level: int = 1
    problem_id: int = 1
    eval_mode: str = "local"  # local or modal
    gpu: str = "H100"
    num_correct_trials: int = 5
    num_perf_trials: int = 1000
    timeout: int = 300
    dataset_src: str = "huggingface"
    dataset_name: str = "ScalingIntelligence/KernelBench"
    max_failure_excerpt_chars: int = 4000
    backend: str = "cuda"  # Backend for kernelbench evaluation (triton, cuda, python)
    precision: str = "fp32"  # fp32, fp16, bf16 -- controls the dtype the eval framework casts model + inputs to and the allclose tolerance


class KernelBenchTask:
    """Task wrapper for KernelBench problem optimization with K-Search."""

    def __init__(
        self,
        *,
        level: int = 1,
        problem_id: int = 1,
        eval_mode: str = "local",
        gpu: str = "H100",
        num_correct_trials: int = 5,
        num_perf_trials: int = 1000,
        timeout: int = 300,
        dataset_src: str = "huggingface",
        dataset_name: str = "ScalingIntelligence/KernelBench",
        artifacts_dir: str | None = None,
        name: str | None = None,
        backend: str = "cuda",
        precision: str = "fp32",
        local_ref_path: str | None = None,
    ) -> None:
        """Initialize KernelBench task.
        
        Args:
            level: KernelBench level (1, 2, or 3)
            problem_id: Problem ID within the level
            eval_mode: Evaluation mode (local or modal)
            gpu: Target GPU type (e.g., H100, A100-80GB)
            num_correct_trials: Number of correctness check trials
            num_perf_trials: Number of performance measurement trials
            timeout: Timeout per evaluation in seconds
            dataset_src: Dataset source (huggingface or local)
            dataset_name: Dataset name for huggingface source
            artifacts_dir: Directory for K-Search artifacts
            name: Task name (defaults to kernelbench_l{level}_p{problem_id})
            backend: Backend for kernel evaluation (triton, cuda, or python)
            local_ref_path: Path to a local .py file with a Model class to optimize
                (bypasses HuggingFace dataset; must define Model, get_inputs, get_init_inputs)
        """
        self._local_ref_path = str(Path(local_ref_path).resolve()) if local_ref_path else None
        self._cfg = KernelBenchTaskConfig(
            level=int(level),
            problem_id=int(problem_id),
            eval_mode=str(eval_mode),
            gpu=str(gpu),
            num_correct_trials=int(num_correct_trials),
            num_perf_trials=int(num_perf_trials),
            timeout=int(timeout),
            dataset_src=str(dataset_src),
            dataset_name=str(dataset_name),
            backend=str(backend),
            precision=str(precision),
        )
        self._name = str(name or f"kernelbench_l{level}_p{problem_id}")
        self._ksearch_artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._solutions: dict[str, Solution] = {}
        
        # Cache for prompt feedback
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0
        self._last_round_summary_line: str = ""
        
        # Load the reference problem
        self._ref_code: str = ""
        self._problem_name: str = ""
        self._load_reference_problem()

    def _load_reference_problem(self) -> None:
        """Load the reference problem from KernelBench dataset or local file."""
        if self._local_ref_path:
            # Load from local file
            ref_path = Path(self._local_ref_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"Local reference file not found: {self._local_ref_path}")
            self._ref_code = ref_path.read_text()
            self._problem_name = ref_path.stem
            print(f"[{self._name}] Loaded local reference: {self._local_ref_path}")
            return

        try:
            # Add src to path for imports
            repo_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(repo_root / "src"))
            
            from kernelbench.dataset import construct_kernelbench_dataset
            
            dataset = construct_kernelbench_dataset(
                level=self._cfg.level,
                source=self._cfg.dataset_src,
                dataset_name=self._cfg.dataset_name,
            )
            problem = dataset.get_problem_by_id(self._cfg.problem_id)
            self._ref_code = problem.code
            self._problem_name = problem.name
            
            print(f"[{self._name}] Loaded KernelBench problem: {self._problem_name}")
            print(f"[{self._name}] Level: {self._cfg.level}, Problem ID: {self._cfg.problem_id}")
            
        except Exception as e:
            print(f"[{self._name}] Error loading reference problem: {e}")
            import traceback
            traceback.print_exc()
            raise

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        """Return the task specification text.
        
        For KernelBench, this includes the reference PyTorch implementation
        and instructions for optimization.
        
        Args:
            language: Target language/backend ('cuda', 'triton', or None to use config default)
        """
        # Use provided language or fall back to backend config
        backend = language.lower() if language else self._cfg.backend.lower()
        
        # Proper capitalization for display
        if backend == "triton":
            backend_display = "Triton"
            format_text = """Your code should define a `ModelNew` class with the same interface as `Model`.
You can include Triton kernels using the `@triton.jit` decorator and launch them appropriately."""
        else:  # cuda or default
            backend_display = "CUDA"
            format_text = """Your code should define a `ModelNew` class with the same interface as `Model`.
You can include inline CUDA code using PyTorch's custom CUDA extensions."""
        
        spec = f"""# KernelBench Optimization Task

**Problem**: {self._problem_name}
**Level**: {self._cfg.level}
**Problem ID**: {self._cfg.problem_id}
**Target GPU**: {self._cfg.gpu}
**Backend**: {backend_display}

## Objective
Optimize the following PyTorch reference implementation by writing a custom {backend_display} kernel.
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

## Format
{format_text}

## Evaluation
Your implementation will be evaluated for:
1. **Correctness**: Output must match reference within tolerance
2. **Performance**: Speedup over PyTorch eager mode and torch.compile

Target: Achieve >1.0x speedup while maintaining correctness.
"""
        return spec

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        """Get the initial generation prompt."""
        return f"{self.get_definition_text()}\n\nTarget GPU: {target_gpu}\n"

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
        """Get the optimization prompt for iterative improvement."""
        prompt = self.get_definition_text()
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
3. **GPU utilization**: Consider thread block sizes, shared memory usage, memory coalescing

Then implement your optimized version.
"""     
        return prompt

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
        """Create a Solution object from generated code."""
        code_text = str(cleaned_code or raw_code or "")
        uid = f"r{round_num}"
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        sol_name = f"{safe_model_name}_{self._name}_{uid}"
        
        sources = [SourceFile(path="model_new.py", content=code_text)]
        spec = BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[str(target_gpu)],
            entry_point="model_new.py::ModelNew",
        )
        
        return Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=spec,
            sources=sources,
            description=f"KernelBench {self._problem_name} optimization",
        )

    def get_solution(self, solution_name: str) -> Solution | None:
        """Get a solution by name."""
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions[name]
        
        # Try loading from artifacts
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
        """Extract code for world model from raw generation."""
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        """Run initial evaluation for base solution."""
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=None)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        """Run benchmark evaluation using run_and_check.py."""
        entry_src = solution.get_entry_source()
        if not entry_src:
            return self._failed_eval("No entry source found in solution", round_num)
        
        code = entry_src.content
        
        if 'class ModelNew' not in code:
            code = code.replace('class Model(', 'class ModelNew(')
            code = code.replace('class Model:', 'class ModelNew:')
            code = re.sub(r'super\(Model,\s*self\)', 'super()', code)
            code = re.sub(r'super\(Model,\s*cls\)', 'super()', code)
        
        kernel_src_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                kernel_src_path = tmp_file.name
            
            repo_root = Path(__file__).parent.parent.parent
            # Verify it looks like the repo root (has k_search directory)
            if not (repo_root / "k_search").exists():
                # Fallback: search from cwd
                repo_root = Path.cwd()
                while repo_root.parent != repo_root:
                    if (repo_root / "k_search").exists():
                        break
                    repo_root = repo_root.parent
            
            if self._local_ref_path:
                cmd = [
                    sys.executable,
                    "k_search/tasks/kernelbench/run_and_check.py",
                    "ref_origin=local",
                    f"ref_arch_src_path={self._local_ref_path}",
                    f"kernel_src_path={kernel_src_path}",
                    f"eval_mode={self._cfg.eval_mode}",
                    f"gpu={self._cfg.gpu}",
                    f"num_correct_trials={self._cfg.num_correct_trials}",
                    f"num_perf_trials={self._cfg.num_perf_trials}",
                    f"timeout={self._cfg.timeout}",
                    f"backend={self._cfg.backend}",
                    f"precision={self._cfg.precision}",
                    "check_kernel=False",
                ]
            else:
                cmd = [
                    sys.executable,
                    "k_search/tasks/kernelbench/run_and_check.py",
                    "ref_origin=kernelbench",
                    f"level={self._cfg.level}",
                    f"problem_id={self._cfg.problem_id}",
                    f"kernel_src_path={kernel_src_path}",
                    f"eval_mode={self._cfg.eval_mode}",
                    f"gpu={self._cfg.gpu}",
                    f"num_correct_trials={self._cfg.num_correct_trials}",
                    f"num_perf_trials={self._cfg.num_perf_trials}",
                    f"timeout={self._cfg.timeout}",
                    f"backend={self._cfg.backend}",
                    f"precision={self._cfg.precision}",
                    "check_kernel=False",
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
            return self._failed_eval(f"Evaluation failed: {excerpt}", round_num)
        
        speedup_eager = self._extract_metric(stdout, "Speedup over eager:", r'([0-9.]+)x')
        speedup_compile = self._extract_metric(stdout, "Speedup over torch.compile:", r'([0-9.]+)x')
        kernel_time = self._extract_metric(stdout, "Custom Kernel exec time:", r'([0-9.]+) ms')
        ref_eager_time = self._extract_metric(stdout, "PyTorch Reference Eager exec time:", r'([0-9.]+) ms')
        
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
            # Failed correctness or compilation
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

    def _extract_metric(self, text: str, line_pattern: str, value_regex: str) -> float | None:
        for line in text.split('\n'):
            if line_pattern in line:
                match = re.search(value_regex, line)
                if match:
                    return float(match.group(1))
        return None

    def _extract_error_excerpt(self, stdout: str, stderr: str) -> str:
        """Extract relevant error information."""
        max_chars = self._cfg.max_failure_excerpt_chars
        
        if stderr.strip():
            excerpt = stderr[-max_chars:] if len(stderr) > max_chars else stderr
            return f"[stderr]\n{excerpt}"
        
        excerpt = stdout[-max_chars:] if len(stdout) > max_chars else stdout
        return f"[stdout]\n{excerpt}"

    def _extract_eval_excerpt(self, stdout: str) -> str:
        """Extract evaluation excerpt starting from [Eval] marker."""
        max_chars = self._cfg.max_failure_excerpt_chars
        
        if '[Eval]' in stdout:
            eval_start = stdout.find('[Eval]')
            excerpt = stdout[eval_start:]
        else:
            excerpt = stdout
        
        if len(excerpt) > max_chars:
            excerpt = excerpt[-max_chars:]
        
        return excerpt

    def _failed_eval(self, message: str, round_num: int | None) -> EvalResult:
        """Create a failed evaluation result."""
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
        """Print evaluation summary."""
        rn = str(round_num) if round_num is not None else "?"
        status = "passed" if passed else "failed"
        
        if passed:
            speedup = er.speedup_factor or 0
            latency = er.latency_ms or 0
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: status={status} | "
                f"speedup={speedup:.2f}x | latency={latency:.4f}ms | "
                f"eval_mode={self._cfg.eval_mode}"
            )
        else:
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: status={status} | "
                f"eval_mode={self._cfg.eval_mode}"
            )
        
        print(self._last_round_summary_line, flush=True)
        
        if not passed:
            excerpt = er.log_excerpt or ""
            if excerpt:
                max_chars = self._cfg.max_failure_excerpt_chars
                if len(excerpt) > max_chars:
                    excerpt = excerpt[:max_chars] + "...<truncated>..."
                print(f"[{self._name}] Failure excerpt:\n{excerpt}", flush=True)

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        """Run final evaluation on solutions.
        
        Args:
            solutions: List of solutions to evaluate
            config: Optional evaluation config (unused for KernelBench)
            dump_traces: Whether to dump evaluation traces (unused for KernelBench)
            workload_limit: Limit on number of workloads (unused for KernelBench)
            
        Returns:
            Dictionary with task metadata and evaluation results for each solution
        """
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
            "level": self._cfg.level,
            "problem_id": self._cfg.problem_id,
            "eval_mode": self._cfg.eval_mode,
            "gpu": self._cfg.gpu,
            "solutions": results,
        }

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs_for_prompt

    def get_last_round_passed_count(self) -> int:
        return self._last_round_passed_count

    def get_last_round_total_workloads(self) -> int:
        return self._last_round_total_workloads

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "task_type": "kernelbench",
            "task_name": self._name,
            "problem_name": self._problem_name,
            "level": self._cfg.level,
            "problem_id": self._cfg.problem_id,
            "eval_mode": self._cfg.eval_mode,
            "gpu": self._cfg.gpu,
            "num_correct_trials": self._cfg.num_correct_trials,
            "num_perf_trials": self._cfg.num_perf_trials,
            "timeout": self._cfg.timeout,
        }
