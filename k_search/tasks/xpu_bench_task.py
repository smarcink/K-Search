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
    num_perf_trials: int = 1000
    num_warmup: int = 100
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

        # ------------------------------------------------------------------
        # XPU Triton workaround: strip tl.constexpr from non-BLOCK params.
        # The Intel XPU Triton backend (triton-xpu 3.7.x) segfaults with
        # certain pointer-count + constexpr-count combinations.  Safe rule:
        # only keep tl.constexpr on params whose names look like compile-time
        # constants (BLOCK*, TILE*, NUM_WARPS, etc.); strip it from runtime
        # shape/stride args (H, W, N, stride_*, etc.).
        # ------------------------------------------------------------------
        if "xpu" in self._cfg.device:
            _SAFE_CONSTEXPR_RE = re.compile(
                r'^(BLOCK|TILE|GROUP|UNROLL|NUM_|DEPTH|STAGES)',
                re.IGNORECASE,
            )
            def _strip_unsafe_constexpr(m):
                param_name = m.group(1)
                if _SAFE_CONSTEXPR_RE.match(param_name):
                    return m.group(0)  # keep it
                return param_name  # strip ": tl.constexpr"
            code_before = code
            code = re.sub(
                r'(\w+)\s*:\s*tl\.constexpr',
                _strip_unsafe_constexpr,
                code,
            )
            if code != code_before:
                _n_stripped = code_before.count('tl.constexpr') - code.count('tl.constexpr')
                print(f"[{self._name}] XPU workaround: stripped tl.constexpr from {_n_stripped} non-BLOCK params")

        kernel_src_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                kernel_src_path = tmp.name

            # Save a persistent copy of every kernel for post-mortem debugging
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
            # Pre-flight check: syntax + import validation in a subprocess
            # to surface Triton compilation errors before they become SIGSEGV.
            # -----------------------------------------------------------
            preflight_code = (
                "import sys\n"
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

            # ----------------------------------------------------------
            # XPU SIGSEGV retry: if the kernel crashed due to the Intel
            # XPU Triton backend bug (chaotic crash pattern based on
            # n_ptrs × n_constexpr × body complexity), try progressively
            # more aggressive workarounds.
            # ----------------------------------------------------------
            if (
                result.returncode == -11
                and "xpu" in self._cfg.device
                and "@triton.jit" in code
            ):
                _xpu_retries = [
                    ("constexpr_pad_7", lambda c: self._xpu_pad_triton_constexpr(c, 7)),
                    ("constexpr_pad_2", lambda c: self._xpu_pad_triton_constexpr(c, 2)),
                    ("constexpr_pad_4", lambda c: self._xpu_pad_triton_constexpr(c, 4)),
                    ("constexpr_pad_1", lambda c: self._xpu_pad_triton_constexpr(c, 1)),
                    ("constexpr_pad_6", lambda c: self._xpu_pad_triton_constexpr(c, 6)),
                    ("strip_all_constexpr", lambda c: self._xpu_strip_all_constexpr(c)),
                ]
                for _retry_name, _retry_fn in _xpu_retries:
                    _retry_code = _retry_fn(code)
                    if _retry_code == code:
                        continue
                    print(f"[{self._name}] XPU SIGSEGV: retrying with {_retry_name}")
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".py", delete=False
                        ) as _rtmp:
                            _rtmp.write(_retry_code)
                            _retry_path = _rtmp.name
                        _retry_cmd = [
                            c.replace(kernel_src_path, _retry_path) if kernel_src_path in c else c
                            for c in cmd
                        ]
                        _retry_result = subprocess.run(
                            _retry_cmd,
                            capture_output=True,
                            text=True,
                            timeout=self._cfg.timeout + 60,
                            cwd=str(repo_root),
                            env=env,
                        )
                        os.unlink(_retry_path)
                        if _retry_result.returncode != -11:
                            print(f"[{self._name}] XPU SIGSEGV workaround '{_retry_name}' succeeded (rc={_retry_result.returncode})")
                            result = _retry_result
                            break
                        print(f"[{self._name}] XPU SIGSEGV persists with '{_retry_name}' (rc={_retry_result.returncode})")
                    except Exception as _e:
                        print(f"[{self._name}] XPU retry '{_retry_name}' error: {_e}")
                        try:
                            os.unlink(_retry_path)
                        except Exception:
                            pass

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
    # XPU SIGSEGV workarounds
    # ------------------------------------------------------------------

    @staticmethod
    def _xpu_pad_triton_constexpr(code: str, n_pad: int) -> str:
        """Add dummy tl.constexpr params to all @triton.jit kernel functions.

        The Intel XPU Triton backend (triton-xpu 3.7.x on BMG/Xe2) has a
        SPIR-V codegen bug that causes SIGSEGV for certain combinations of
        (n_ptrs, n_regular, n_constexpr) parameters when the kernel body
        is non-trivial.  Padding the constexpr count can shift the parameter
        layout out of the crash zone.

        This modifies both the function signature AND the kernel launch call
        sites by inserting ``_XPU_PAD_N: tl.constexpr`` / ``_XPU_PAD_N=0``.
        """
        if "triton.jit" not in code and "triton.jit" not in code:
            return code

        # 1. Find @triton.jit function names
        jit_func_pattern = re.compile(
            r'@triton\.jit\s*\n\s*def\s+(\w+)\s*\(', re.MULTILINE
        )
        func_names = jit_func_pattern.findall(code)
        if not func_names:
            return code

        pad_params = ", ".join(f"_XPU_PAD_{i}: tl.constexpr" for i in range(n_pad))
        pad_kwargs = ", ".join(f"_XPU_PAD_{i}=0" for i in range(n_pad))

        result = code

        for fname in func_names:
            # 2. Insert padding params at end of function signature.
            # Find the closing ')' of the def statement.  Handle multi-line
            # signatures by counting parentheses from the 'def fname(' match.
            def_pattern = re.compile(
                rf'(def\s+{re.escape(fname)}\s*\()', re.MULTILINE
            )
            m = def_pattern.search(result)
            if not m:
                continue

            start = m.end()  # right after the opening '('
            depth = 1
            pos = start
            while pos < len(result) and depth > 0:
                if result[pos] == '(':
                    depth += 1
                elif result[pos] == ')':
                    depth -= 1
                pos += 1
            if depth != 0:
                continue  # unbalanced parens, skip

            close_paren = pos - 1  # index of the matching ')'

            # Insert padding before the closing ')'
            # Handle trailing comma / whitespace
            before = result[start:close_paren].rstrip()
            if before and not before.endswith(','):
                sep = ",\n        "
            else:
                sep = "\n        "
            result = result[:close_paren] + sep + pad_params + ",\n    " + result[close_paren:]

            # 3. Insert padding kwargs at kernel launch call sites.
            # Pattern: funcname[grid](..., num_warps=...) or funcname[grid](..., )
            # We insert _XPU_PAD_N=0 before num_warps= or num_stages= or at
            # the end of the arg list.
            call_pattern = re.compile(
                rf'({re.escape(fname)}\s*\[.*?\]\s*\()', re.DOTALL
            )
            search_start = 0
            while True:
                cm = call_pattern.search(result, search_start)
                if not cm:
                    break

                # Find closing ')' of the call
                call_start = cm.end()
                depth = 1
                pos = call_start
                while pos < len(result) and depth > 0:
                    ch = result[pos]
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    elif ch in ('"', "'"):
                        # skip strings
                        q = ch
                        pos += 1
                        while pos < len(result) and result[pos] != q:
                            if result[pos] == '\\':
                                pos += 1
                            pos += 1
                    pos += 1
                if depth != 0:
                    search_start = pos
                    continue

                call_close = pos - 1

                # Try to insert before num_warps= or num_stages=
                call_body = result[call_start:call_close]
                nw_match = re.search(r'(\s*num_warps\s*=)', call_body)
                ns_match = re.search(r'(\s*num_stages\s*=)', call_body)

                if nw_match:
                    insert_pos = call_start + nw_match.start()
                elif ns_match:
                    insert_pos = call_start + ns_match.start()
                else:
                    # Insert before closing ')'
                    insert_pos = call_close

                # Build the insertion text
                before_insert = result[:insert_pos].rstrip()
                if before_insert and not before_insert.endswith(','):
                    insert_text = ",\n                " + pad_kwargs + ",\n                "
                else:
                    insert_text = "\n                " + pad_kwargs + ",\n                "

                result = result[:insert_pos] + insert_text + result[insert_pos:]
                # Advance past this call to avoid re-matching
                search_start = insert_pos + len(insert_text) + 100

        return result

    @staticmethod
    def _xpu_strip_all_constexpr(code: str) -> str:
        """Strip ALL tl.constexpr annotations, inlining literal values
        throughout kernel bodies so that tl.arange, tl.zeros, tl.full,
        tl.static_range, etc. still receive compile-time constants.

        Strategy: for each constexpr param, find its launch-site value,
        then replace EVERY occurrence of the param name (word-boundary)
        inside the kernel function body with the literal value.  The param
        is kept in the signature as a regular (unused) arg so the call
        site doesn't need changes.
        """
        if "tl.constexpr" not in code:
            return code

        # 1. Collect constexpr param names per @triton.jit function
        constexpr_names = re.findall(r'(\w+)\s*:\s*tl\.constexpr', code)
        if not constexpr_names:
            return code

        # 2. Build name → literal value mapping from call sites (named kwargs)
        name_to_val: dict[str, str] = {}
        for name in constexpr_names:
            val_match = re.search(rf'\b{re.escape(name)}\s*=\s*(\d+)', code)
            if val_match:
                name_to_val[name] = val_match.group(1)

        # 3. Strip the ": tl.constexpr" annotations from signatures
        result = re.sub(r'(\w+)\s*:\s*tl\.constexpr', r'\1', code)

        # 4. For each @triton.jit function, inline values in the body
        #    We locate each 'def fname(...):' and replace param names
        #    with literal values inside the body (up to the next top-level
        #    def/class or end of file).
        jit_func_re = re.compile(
            r'@triton\.jit\s*\n\s*def\s+(\w+)\s*\(', re.MULTILINE
        )
        for jit_match in jit_func_re.finditer(result):
            # Find the end of the function signature (the ':' after ')')
            sig_start = jit_match.end()
            depth = 1
            pos = sig_start
            while pos < len(result) and depth > 0:
                if result[pos] == '(':
                    depth += 1
                elif result[pos] == ')':
                    depth -= 1
                pos += 1
            # pos is now just after the closing ')' of the signature
            # Find the ':' that ends 'def f(...):'
            colon_pos = result.find(':', pos)
            if colon_pos == -1:
                continue
            body_start = colon_pos + 1

            # Find end of function body: next top-level def/class/@ or EOF
            body_end_match = re.search(
                r'\n(?=\S)',  # next line starting at column 0
                result[body_start:],
            )
            body_end = body_start + body_end_match.start() if body_end_match else len(result)

            # Extract body, do replacements, put it back
            body = result[body_start:body_end]
            for name, val in name_to_val.items():
                body = re.sub(rf'\b{re.escape(name)}\b', val, body)
            result = result[:body_start] + body + result[body_end:]

        return result

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
