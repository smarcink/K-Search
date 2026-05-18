import argparse
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Optional
import json

def _persist_ksearch_solution(
    solution: Any, *, definition_name: str, artifacts_dir: Optional[str]
) -> Optional[Path]:
    """
    Persist a k-search task_base.Solution JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir
    except Exception:
        return None
    try:
        from k_search.tasks.task_base import Solution as KSearchSolution
    except Exception:
        KSearchSolution = None  # type: ignore

    try:
        # Note: base_dir is provided by caller; default remains ./ .ksearch
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or "")
        ).resolve()
        out_dir = root / "solutions" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(solution, "name", "") or "solution")
        dest = out_dir / f"{name}.json"
        if KSearchSolution is not None and isinstance(solution, KSearchSolution):
            obj = solution.to_dict()
        else:
            obj = solution.__dict__ if hasattr(solution, "__dict__") else {"solution": str(solution)}
        dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving k-search solution: {e}")
        import traceback
        traceback.print_exc()
        return None


def _persist_ksearch_eval_report(
    report: dict[str, Any],
    *,
    definition_name: str,
    solution_name: Optional[str],
    artifacts_dir: Optional[str],
) -> Optional[Path]:
    """
    Persist a final-eval report JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir
    except Exception:
        return None
    try:
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or "")
        ).resolve()
        out_dir = root / "eval" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sn = str(solution_name or "").strip()
        safe_sn = "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in sn]) if sn else ""
        suffix = f"_{safe_sn}" if safe_sn else ""
        dest = out_dir / f"eval_report_{ts}{suffix}.json"
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving eval report: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_and_evaluate(
    task: Any,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
    target_gpu: str,
    max_opt_rounds: int,
    save_solutions: bool,
    save_results: bool,
    continue_from_solution: Optional[str] = None,
    continue_from_world_model: Optional[str] = None,
    num_eval_workload: Optional[int] = None,
    # W&B options
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    # World model prompting
    enable_world_model: bool = False,
    wm_stagnation_window: int = 5,
    wm_max_difficulty: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    hw_spec_path: Optional[str] = None,
) -> None:
    """
    Generate exactly one solution for the task, then run final evaluation.
    """
    
    # Optional Weights & Biases support
    try:
        import wandb  # type: ignore
    except Exception:  # pragma: no cover
        wandb = None

    # Initialize wandb if enabled
    wb_run = None
    if enable_wandb and wandb is not None:
        print(f"Initializing wandb with project: {wandb_project} and name: {run_name}")
        try:
            task_cfg = task.get_config_for_logging()
        except Exception:
            task_cfg = {}
        wb_run = wandb.init(
            project=wandb_project or os.getenv("WANDB_PROJECT", "flashinfer-bench"),
            name=run_name or os.getenv("RUN_NAME"),
            config={
                "task": task_cfg,
                "generator": {
                    "model_name": model_name,
                    "language": language,
                    "target_gpu": target_gpu,
                },
                "max_opt_rounds": int(max_opt_rounds),
                "continue_from_solution": continue_from_solution,
                "continue_from_world_model": continue_from_world_model,
                "enable_world_model": bool(enable_world_model),
                "wm_stagnation_window": int(wm_stagnation_window),
                "wm_max_difficulty": wm_max_difficulty,
                "save_results": bool(save_results),
                "save_solutions": bool(save_solutions),
                "num_eval_workload": num_eval_workload,
                "artifacts_dir": artifacts_dir,
            },
            reinit=True,
        )

    def _eval_and_report_one(*, sol: Any) -> None:
        def_name = str(getattr(task, "name", "") or "")
        sol_name = str(getattr(sol, "name", "") or "")

        report = task.run_final_evaluation(
            solutions=[sol],
            config=None,
            dump_traces=bool(save_results),
            workload_limit=num_eval_workload,
        )
        if save_results:
            saved = _persist_ksearch_eval_report(
                report,
                definition_name=def_name,
                solution_name=sol_name,
                artifacts_dir=artifacts_dir,
            )
            if saved:
                print(f"[{def_name}] Saved eval report to: {saved}")

    if enable_world_model:
        # World-model mode uses the WM generator (task-driven).
        from k_search.kernel_generators.kernel_generator_world_model import WorldModelKernelGeneratorWithBaseline

        generator = WorldModelKernelGeneratorWithBaseline(
            model_name=model_name,
            language=language,
            target_gpu=target_gpu,
            api_key=api_key,
            base_url=base_url,
            artifacts_dir=artifacts_dir,
            wm_max_difficulty=wm_max_difficulty,
            hw_spec_path=hw_spec_path,
        )
    else:
        # Non-world-model mode: baseline-style generator (task-driven).
        from k_search.kernel_generators.kernel_generator import KernelGenerator

        generator = KernelGenerator(
            model_name=model_name,
            language=language,
            target_gpu=target_gpu,
            api_key=api_key,
            base_url=base_url,
            hw_spec_path=hw_spec_path,
        )

    # Generate exactly one solution.
    if enable_world_model:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=int(wm_stagnation_window),
            continue_from_solution=continue_from_solution,
            continue_from_world_model=continue_from_world_model,
        )
    else:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            continue_from_solution=continue_from_solution,
        )

    # Append timestamp and uid to ensure uniqueness and traceability
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    solution.name = f"{solution.name}_{ts}_{uid}"
    # Optional: reflect in description
    try:
        solution.description = (solution.description or "") + f" (generated {ts} uid={uid})"
    except Exception:
        pass

    # Optionally persist to disk (k-search solution type)
    if save_solutions:
        saved_path = _persist_ksearch_solution(
            solution, definition_name=str(getattr(task, "name", "") or ""), artifacts_dir=artifacts_dir
        )
        if saved_path:
            print(f"  ✓ Saved solution to: {saved_path}")
        else:
            print(f"  ✗ Failed to save solution")

    def_name = str(getattr(task, "name", "") or "")
    print(f"[{def_name}] Generated solution: {solution.name}")

    # Final eval: evaluate ONLY the solution(s) returned by the generator, one at a time.
    # This keeps the logic simple and avoids comparing multiple generated solutions in one report.
    _eval_and_report_one(sol=solution)

    # Cleanly close W&B run if it was opened (prevents BrokenPipe in Ray workers)
    if wb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate kernels with GPT/Gemini (OpenAI-compatible) and evaluate via task backends.")
    parser.add_argument(
        "--task-path",
        default=None,
        help="Path to reference .py file with Model class to optimize.",
    )
    parser.add_argument("--definition", default=None, help="Single definition name to target (required)")
    parser.add_argument("--model-name", required=True, help="LLM model name (e.g., gpt-4.1, gpt-5, gemini-2.5-pro via OpenAI-compatible endpoint, or claude-opus-4-6/claude-4-6-opus via Anthropic-compatible endpoint)")
    parser.add_argument("--base-url", default=None, help="Provider base URL. For Claude/Anthropic models a GNAI OpenAI URL is auto-rewritten to the /providers/anthropic path; or pass it directly (e.g. https://gnai.intel.com/api/providers/anthropic)")
    parser.add_argument("--api-key", default=None, help="API key; if omitted, uses LLM_API_KEY env var")
    parser.add_argument("--language", default="triton", choices=["triton", "python", "cuda"], help="Target language for generated kernel. 'cuda' uses the CUDA kernel task; 'triton'/'python' uses the Triton kernel task.")
    parser.add_argument("--target-gpu", default=None, help="Target GPU architecture hint (e.g. 'H100', 'RTX 5090'). Mutually exclusive with --hw-spec.")
    parser.add_argument("--hw-spec", default=None, help="Path to a HW spec JSON file. Mutually exclusive with --target-gpu.")
    parser.add_argument("--max-opt-rounds", type=int, default=5, help="Max optimization rounds for each solution generation")

    # Benchmark configuration
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--no-save-results", action="store_true", help="Do not write traces to dataset")
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Persist generated solutions JSON into the k-search artifacts dir (see --artifacts-dir)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=".ksearch",
        help="Base directory for k-search artifacts (solutions, world model snapshots, eval reports).",
    )
    parser.add_argument("--num-eval-workload", type=int, default=None, help="If set, evaluate only this many workloads per definition; default uses all workloads")
    # Continue optimization options
    parser.add_argument("--continue-from-solution", default=None,
                        help="Resume optimization from an existing solution (name, .json path, or directory with kernel sources)")
    parser.add_argument(
        "--continue-from-world-model",
        default=None,
        help=(
            "Resume world-model prompting state from a JSON file path. "
            "Use 'auto' to load <artifacts>/<task>/world_model/world_model.json if present."
        ),
    )

    parser.add_argument(
        "--world-model",
        action="store_true",
        help="Enable world-model prompting (maintain a persistent world model across rounds and inject it into prompts).",
    )
    parser.add_argument(
        "--wm-stagnation-window",
        type=int,
        default=5,
        help="World-model mode: end an action cycle after this many consecutive non-improving rounds (>=1).",
    )
    parser.add_argument(
        "--wm-max-difficulty",
        type=int,
        default=None,
        help="World-model mode: max difficulty (1-5) for action selection. Actions above this are deferred. Default: use policy default (4).",
    )
    # W&B options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"), help="W&B project")
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME"), help="W&B run name")

    # CUDA Kernel options
    parser.add_argument("--cuda-kernel-num-correct-trials", type=int, default=5, help="Number of correctness trials (cuda_kernel task)")
    parser.add_argument("--cuda-kernel-num-perf-trials", type=int, default=100, help="Number of performance trials (cuda_kernel task)")
    parser.add_argument(
        "--cuda-kernel-precision",
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="dtype for cuda_kernel eval",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Run hardware profiler on passed kernels to collect metrics for the LLM "
             "(NCU on CUDA; planned VTune for XPU). No-op on backends without a wired-up profiler.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=2,
        help="Number of unprofiled warmup runs before profiler capture when --enable-profiling is set.",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=1,
        help="Number of profiled candidate runs captured by the profiler when --enable-profiling is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output: print kernel source and profiler report after each evaluation.",
    )

    # Triton Kernel options (unified for CUDA / XPU / any accelerator)
    parser.add_argument("--triton-device", default=None,
                        help="Device string for Triton kernel task (e.g. cuda:0, xpu:0). Required for --language=triton.")
    parser.add_argument(
        "--triton-precision",
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="dtype for Triton kernel eval",
    )
    parser.add_argument("--triton-num-correct-trials", type=int, default=5, help="Number of correctness trials (triton kernel task)")
    parser.add_argument("--triton-num-perf-trials", type=int, default=100, help="Number of performance trials (triton kernel task)")

    args = parser.parse_args()

    if args.hw_spec and args.target_gpu:
        parser.error("--hw-spec and --target-gpu are mutually exclusive; provide one or the other, not both.")
    if not args.target_gpu and not args.hw_spec:
        parser.error("Either --target-gpu or --hw-spec must be provided.")

    # When --hw-spec is provided, extract target_gpu from the spec's "name" field.
    if args.hw_spec and not args.target_gpu:
        from k_search.hw_specs import HWSpec
        args.target_gpu = HWSpec.from_file(args.hw_spec).name

    if not args.target_gpu:
        parser.error("Could not determine target GPU. Provide --target-gpu or a valid --hw-spec with a 'name' field.")

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("API key is required (pass --api-key or set LLM_API_KEY)")

    task_path = str(args.task_path or "")
    language = args.language

    if language == "cuda":
        from k_search.tasks.cuda_kernel_task import CudaKernelTask

        if not task_path:
            raise ValueError("--task-path is required for --language=cuda (path to reference .py file)")
        task = CudaKernelTask(
            ref_path=task_path,
            gpu=args.target_gpu,
            num_correct_trials=args.cuda_kernel_num_correct_trials,
            num_perf_trials=args.cuda_kernel_num_perf_trials,
            precision=args.cuda_kernel_precision,
            rtol=args.rtol,
            atol=args.atol,
            profile_warmup=args.profile_warmup,
            profile_repeats=args.profile_repeats,
            artifacts_dir=args.artifacts_dir,
            enable_ncu_profiling=args.enable_profiling,
            verbose=args.verbose,
        )
    elif language in ("triton", "python"):
        from k_search.tasks.triton_kernel_task import TritonKernelTask

        if not task_path:
            raise ValueError("--task-path is required for --language=triton (path to reference .py file with Model class)")

        if not args.triton_device:
            raise ValueError("--triton-device is required for --language=triton (e.g. cuda:0, xpu:0)")

        task = TritonKernelTask(
            ref_path=task_path,
            device=args.triton_device,
            precision=args.triton_precision,
            num_correct_trials=args.triton_num_correct_trials,
            num_perf_trials=args.triton_num_perf_trials,
            profile_warmup=args.profile_warmup,
            profile_repeats=args.profile_repeats,
            artifacts_dir=args.artifacts_dir,
            enable_profiling=args.enable_profiling,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unsupported language: {language}")

    generate_and_evaluate(
        task=task,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=api_key,
        language=args.language,
        target_gpu=args.target_gpu,
        max_opt_rounds=args.max_opt_rounds,
        save_solutions=args.save_solutions,
        save_results=not args.no_save_results,
        num_eval_workload=args.num_eval_workload,
        continue_from_solution=args.continue_from_solution,
        continue_from_world_model=args.continue_from_world_model,
        enable_world_model=args.world_model,
        wm_stagnation_window=args.wm_stagnation_window,
        wm_max_difficulty=args.wm_max_difficulty,
        artifacts_dir=args.artifacts_dir,
        hw_spec_path=args.hw_spec,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()


