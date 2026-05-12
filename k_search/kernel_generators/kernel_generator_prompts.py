"""
This file contains the prompts for baseline agent generation.
"""

from __future__ import annotations

# CUDA-specific hints
# Note: keep these hints generic (avoid naming specific low-level instructions).
CUDA_OPTIMIZATION_HINTS = "** You MUST use MMA to utilize the tensor cores on H100! ** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step."

# Intel XPU Triton hints
XPU_TRITON_OPTIMIZATION_HINTS = """
** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step.

Key Triton performance principles for Intel XPU:
- This kernel targets an Intel GPU (Xe2 / Battlemage architecture), NOT an NVIDIA GPU.
- Use tl.dot() for matrix operations to leverage Intel Xe Matrix Extensions (XMX).
  Sub-group sizes are 16 or 32; choose tile sizes that are multiples of these.
- Stage data through shared memory (SRAM): load input tiles with tl.load into a block,
  then use tl.dot for the compute. This maximizes data reuse and hides memory latency.
- The device has 256 Execution Units (EUs), 32 subslices, 32 GB device memory, 192-bit bus.
- Do NOT reference cuDNN, cuBLAS, CUTLASS, MMA instructions, or any NVIDIA-specific APIs.
- Do NOT use torch.cuda — use torch.xpu for device operations.
- Triton kernels use the same standard primitives (tl.load, tl.store, tl.dot, tl.program_id, etc.)
  regardless of backend — the Intel XPU Triton backend compiles them automatically.
- Fusing elementwise ops (bias, activation, pooling) into a matmul epilogue is where
  custom Triton kernels can beat torch.compile.

CRITICAL Intel XPU Triton backend constraints (violations cause SIGSEGV or compilation failure):
- tl.arange(start, end): the range (end - start) MUST be a power of 2 (e.g., 16, 32, 64, 128).
  Non-power-of-2 ranges (e.g., 96, 48, 3) will cause a compilation error or crash.
  If you need to work with non-power-of-2 dimensions (e.g., dim=96), round UP to the next
  power of 2 (e.g., 128) and use a mask to guard out-of-bounds elements.
- BLOCK sizes and tile dimensions used as constexpr MUST be powers of 2.
- tl.dot(a, b) operands must have inner dimension that is a multiple of 16.
  Minimum supported shapes: (M, 16) x (16, N). Smaller inner dims will crash.
- Avoid complex control flow (nested if/else, dynamic loops) inside Triton kernels —
  the Intel backend has limited support and may produce invalid SPIR-V, causing SIGSEGV.
- Keep kernels simple and well-structured. If a kernel causes SIGSEGV, simplify it —
  break fused operations into separate smaller kernels rather than one monolithic kernel.
- torch.xpu.synchronize() instead of torch.cuda.synchronize().
"""

# Triton-appropriate subset
TRITON_OPTIMIZATION_HINTS = """
** For each round, you can see your current best solution and the previous round's summary, therefore you can implement the kernel step by step.

Key Triton performance principles:
- Use tl.dot() for any reduction/inner-product dimension >= 16 to leverage GPU tensor cores.
  Scalar FMA loops (wv * xv in tl.static_range) use only CUDA cores and are much slower.
- For convolutions, consider the implicit-GEMM approach: reshape the convolution as a
  matrix multiplication (im2col-style or direct) and use tl.dot() on [K, N] x [N, M] tiles.
- Stage data through shared memory (SRAM): load input tiles with tl.load into a block,
  then use tl.dot for the compute. This maximizes data reuse and hides memory latency.
- If your custom kernel cannot beat torch.compile, you likely need a fundamentally
  different algorithm (not just tile-size tuning). torch.compile uses cuDNN/tensor-cores
  internally for standard ops like conv2d.
- Fusing elementwise ops (bias, activation, pooling) into the matmul epilogue is where
  custom Triton kernels can beat torch.compile — but only if the matmul itself uses
  tensor cores via tl.dot.
"""


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

{per_task_requirement}

{hints}

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed or its performance can be improved.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{per_task_requirement}

{hints}

{extra_context}
Generate the corrected and optimized implementation:"""


# CUDA prompt
CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

{per_task_requirement}

{hints}

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

{per_task_requirement}

{hints}

{extra_context}

Generate the corrected and optimized implementation:"""


def _is_intel_gpu(target_gpu: str) -> bool:
    """Check if the target GPU string refers to an Intel device."""
    tg = (target_gpu or "").lower()
    return any(kw in tg for kw in ("intel", "arc", "xpu", "xe", "battlemage", "ponte vecchio", "bmg"))


def _select_triton_hints(target_gpu: str) -> str:
    """Return the appropriate Triton optimization hints for the target GPU."""
    if _is_intel_gpu(target_gpu):
        return XPU_TRITON_OPTIMIZATION_HINTS
    return TRITON_OPTIMIZATION_HINTS


def get_prompt_from_definition_text(
    language: str,
    definition_text: str,
    target_gpu: str = "H100",
    *,
    per_task_requirement: str = "",
) -> str:
    """
    Task-agnostic prompt builder: takes a fully-rendered definition text.
    """
    prompts = {"triton": TRITON_PROMPT, "cuda": CUDA_PROMPT}

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    # Only Triton/CUDA prompts include advanced hints
    if language == "triton":
        hints = _select_triton_hints(target_gpu)
        return prompts[language].format(
            definition=str(definition_text or "").strip(),
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=hints,
        )
    if language == "cuda":
        return prompts[language].format(
            definition=str(definition_text or "").strip(),
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
        )
    return prompts[language].format(definition=str(definition_text or "").strip(), target_gpu=target_gpu)


def get_optimization_prompt_from_definition_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    target_gpu: str = "H100",
    current_best: str | None = None,
    previous_round_summary: str | None = None,
    per_task_requirement: str = "",
) -> str:
    """
    Task-agnostic optimization prompt builder: takes rendered definition + rendered trace logs.
    """
    optimization_prompts = {"triton": TRITON_OPTIMIZATION_PROMPT, "cuda": CUDA_OPTIMIZATION_PROMPT}

    if language not in optimization_prompts:
        raise ValueError(f"Unsupported language for optimization: {language}")

    extra_context = _build_extra_context(
        current_best=current_best,
        previous_round_summary=previous_round_summary,
    )

    if language == "triton":
        hints = _select_triton_hints(target_gpu)
        return optimization_prompts[language].format(
            definition=str(definition_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            current_code=current_code,
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=hints,
            extra_context=extra_context,
        )
    if language == "cuda":
        return optimization_prompts[language].format(
            definition=str(definition_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            current_code=current_code,
            target_gpu=target_gpu,
            per_task_requirement=str(per_task_requirement or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
            extra_context=extra_context,
        )
    # Python doesn't use this path
    raise ValueError(f"No optimization prompt available for language: {language}")


def _build_extra_context(
    *,
    current_best: str | None,
    previous_round_summary: str | None,
) -> str:
    """
    Optional context that can be injected into optimization prompts.
    Returns an empty string when no context is available.
    """
    parts: list[str] = []

    # Put "best so far" last to avoid interrupting the flow of "what just happened" (prev round + profiling).
    if current_best and current_best.strip():
        parts.append("Current Best Solution So Far (best so far across rounds):\n" + current_best.strip())
      
    if previous_round_summary and previous_round_summary.strip():
        parts.append("Last Round Summary::\n" + previous_round_summary.strip())


    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
