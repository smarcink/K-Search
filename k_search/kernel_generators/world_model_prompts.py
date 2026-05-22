"""
World-model codegen prompts (action application + debugging).

These prompts are WM-specific and intentionally kept separate from the baseline
kernel generator prompts in `kernel_generator_prompts.py`.
"""

from __future__ import annotations

from .kernel_generator_prompts import (
    CUDA_OPTIMIZATION_HINTS,
    TRITON_OPTIMIZATION_HINTS,
    XPU_TRITON_OPTIMIZATION_HINTS,
    _is_intel_gpu,
)


TRITON_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good Triton baseline for {target_gpu}.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (start from this; do not include any other previous code):
{base_code}

Chosen Next Action (apply this action):
{action_text}

{code_format}

Rules:
- Implement ONLY the chosen action; keep everything else as close as possible to the base implementation.
- Keep changes small and single-iteration implementable.
- Preserve correctness and the function signature / wrapper behavior.
- Return only the full updated code (no explanations, no markdown).

{hints}

Generate the updated implementation:"""


CUDA_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good CUDA baseline for {target_gpu}.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (start from this; do not include any other previous code):
{base_code}

Chosen Next Action (apply this action):
{action_text}

Rules:
- Implement ONLY the chosen action; keep everything else as close as possible to the base implementation.
- Keep changes small and single-iteration implementable.
- Preserve correctness and the same PyTorch binding / entry points.
- Return only the full updated XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the updated implementation:"""


HLSL_OPTIMIZATION_HINTS = """Key HLSL/DX12 POC constraints:
- Generate a single HLSL compute shader plus launch.json XML blocks.
- The current runner is single-dispatch and CPU-staged through hlsl_probe; optimize the shader GPU timestamp.
- Use fixed shape constants from the task metadata. Root constants, CBVs, temporary buffers, and multi-dispatch graphs are not wired up yet.
- Bind forward inputs and model state as SRVs t0, t1, ... and outputs as UAVs u0, u1, ... exactly as specified.
- Target cs_6_8/cs_6_9 style HLSL unless the task explicitly requests a newer target.
- For cs_6_8/cs_6_9, total groupshared memory is limited to 32 KiB per threadgroup in this D3D/DXIL path; hardware CUDA shared-memory-per-SM figures do not raise that limit.
- Prefer register-resident scalars/vectors for per-thread or single-owner intermediate values. Full fusion should minimize materialization, not store every phase's output in groupshared memory.
- Treat groupshared memory as an explicitly synchronized communication/cache resource. Good uses include compact read-only tiles/tables loaded cooperatively once and consumed many times, reusable input tiles, cross-thread or cross-wave exchange buffers, and reduction scratch.
- Do not justify groupshared use only because a tensor or working set fits under 32 KiB. For every groupshared array, classify it as read-only reusable tile, exchange buffer, or reduction scratch, and state the producer thread set, consumer thread set, reuse count/traffic saving, and required synchronization point.
- Keep groupshared allocations comfortably below 32 KiB after type/layout alignment. Do not stage a whole working set or whole phase tensors in groupshared merely to pass values between sequential phases when streaming from SRVs/L2, using wave communication, keeping owner-local registers, or recomputing a small value is cheaper.
- Prefer mappings where one fixed-size wave owns one independent work unit/tile when the tile fits wave-local ownership. Use WaveGetLaneIndex, WaveReadLaneAt, WaveReadLaneFirst, WaveActiveSum/Max/Min, and related wave intrinsics for lane exchange and reductions.
- Do not route wave-local intermediate values through groupshared memory. Keep them in registers and exchange through wave intrinsics; this avoids turning warp/wave-local dependencies into group-wide barriers.
- Avoid splitting one independent work unit across multiple waves unless the extra parallelism clearly pays for the required inter-wave handoff. If multiple waves share a threadgroup, prefer each wave owning an independent work unit, with at most compact read-only shared tiles loaded once for all waves.
- Minimize GroupMemoryBarrierWithGroupSync calls. They are usually needed only after cross-thread groupshared writes before cross-thread reads; avoid phase-by-phase barriers for values that can stay in registers or have a single producer/consumer.
- Use [WaveSize(32)] when the target accepts it and the mapping assumes 32 lanes. Prefer wave intrinsics for wave-local reductions, broadcasts, scans, or shuffles; use group-wide barriers only for true group-wide communication.
"""


HLSL_ACTION_PROMPT = """You are implementing a SPECIFIC NEXT ACTION on top of a known-good HLSL baseline for {target_gpu}.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (start from this; do not include any other previous code):
{base_code}

Chosen Next Action (apply this action):
{action_text}

Rules:
- Implement ONLY the chosen action; keep everything else as close as possible to the base implementation.
- Keep changes small and single-iteration implementable.
- Preserve correctness, resource bindings, entry point, and dispatch metadata.
- Return only the full updated XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the updated implementation:"""


TRITON_DEBUG_PROMPT = """You are in a debug-and-improve loop for a Triton kernel on {target_gpu}.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (reference):
{base_code}

Current Implementation (fix or improve THIS code; keep it aligned with the base and the chosen action):
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action (still targeting; do not expand scope):
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

{code_format}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Keep the implementation aligned with the base and the chosen action intent.
- Return only the full corrected code (no explanations, no markdown).

{hints}

Generate the corrected implementation:"""


CUDA_DEBUG_PROMPT = """You are in a debug-and-improve loop for a CUDA kernel on {target_gpu}.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (reference):
{base_code}

Current Implementation (fix or improve THIS code; keep it aligned with the base and the chosen action):
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action (still targeting; do not expand scope):
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Keep the implementation aligned with the base and the chosen action intent.
- Return only the full corrected XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the corrected implementation:"""


HLSL_DEBUG_PROMPT = """You are in a debug-and-improve loop for an HLSL compute shader on {target_gpu}.
The current implementation may be buggy OR already correct-but-slower-than-desired.

Original Specification:
{definition}

{hw_spec}

Known-Good Base Implementation (reference):
{base_code}

Current Implementation (fix or improve THIS code; keep it aligned with the base and the chosen action):
{buggy_code}

Performance Summary:
{perf_summary}

Failure Logs:
{trace_logs}

Chosen Next Action (still targeting; do not expand scope):
{action_text}

Debug-and-improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix HLSL compile, DX12 probe, metadata, dispatch, or correctness issues FIRST.
- If the current implementation PASSED: improve GPU timestamp performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Keep the implementation aligned with the base and the chosen action intent.
- Return only the full corrected XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the corrected implementation:"""


TRITON_IMPROVE_PROMPT = """You are improving a Triton kernel on {target_gpu}.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

{hw_spec}

Cycle-Best Base Implementation (reference):
{base_code}

Current Implementation (improve THIS code; keep it aligned with the base):
{current_code}

Performance Summary:
{perf_summary}

Recent Logs (only if FAILED):
{trace_logs}

Improve round: {debug_round}/{max_rounds}

{code_format}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Return only the full corrected code (no explanations, no markdown).

{hints}

Generate the improved implementation:"""


CUDA_IMPROVE_PROMPT = """You are improving a CUDA kernel on {target_gpu}.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

{hw_spec}

Cycle-Best Base Implementation (reference):
{base_code}

Current Implementation (improve THIS code; keep it aligned with the base):
{current_code}

Performance Summary:
{perf_summary}

Recent Logs (only if FAILED):
{trace_logs}

Improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix correctness/compile/runtime issues FIRST.
- If the current implementation PASSED: improve performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Return only the full corrected XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the improved implementation:"""


HLSL_IMPROVE_PROMPT = """You are improving an HLSL compute shader on {target_gpu}.
The current implementation may be correct-but-slower-than-desired, or it may have regressed.

Original Specification:
{definition}

{hw_spec}

Cycle-Best Base Implementation (reference):
{base_code}

Current Implementation (improve THIS code; keep it aligned with the base):
{current_code}

Performance Summary:
{perf_summary}

Recent Logs (only if FAILED):
{trace_logs}

Improve round: {debug_round}/{max_rounds}

Rules:
- If the current implementation FAILED: fix HLSL compile, DX12 probe, metadata, dispatch, or correctness issues FIRST.
- If the current implementation PASSED: improve GPU timestamp performance while preserving correctness.
- Keep changes minimal; do not introduce extra unrelated optimizations.
- Return only the full corrected XML blocks (no explanations, no markdown).

{code_format}

{hints}

Generate the improved implementation:"""

def get_generate_code_from_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    base_code: str,
    action_text: str,
    code_format: str = "",
    target_gpu: str = "H100",
    hw_spec: str = "",
) -> str:
    """Task-agnostic variant: accepts rendered definition text."""
    lang = (language or "").lower()
    hw_spec_text = str(hw_spec or "").strip()
    if lang == "triton":
        hints = XPU_TRITON_OPTIMIZATION_HINTS if _is_intel_gpu(target_gpu) else TRITON_OPTIMIZATION_HINTS
        return TRITON_ACTION_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            action_text=action_text,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=hints,
            hw_spec=hw_spec_text,
        )
    if lang == "cuda":
        return CUDA_ACTION_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            action_text=action_text,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    if lang == "hlsl":
        return HLSL_ACTION_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            action_text=action_text,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=HLSL_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    raise ValueError(f"Unsupported language for action prompt: {language}")


def get_generate_code_from_spec_with_action_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    action_text: str,
    code_format: str = "",
    target_gpu: str = "H100",
    hw_spec: str = "",
) -> str:
    """
    Task-agnostic variant: accepts rendered definition text.
    Used when the chosen action's parent is the WM root: start from spec + action only.
    """
    lang = (language or "").lower()
    hw_spec_text = str(hw_spec or "").strip()
    if lang == "triton":
        hints = XPU_TRITON_OPTIMIZATION_HINTS if _is_intel_gpu(target_gpu) else TRITON_OPTIMIZATION_HINTS
        return (
            "You are implementing a SPECIFIC NEXT ACTION starting from the specification.\n\n"
            + TRITON_ACTION_PROMPT.format(
                definition=str(definition_text or "").strip(),
                base_code="(no base code; start from spec)",
                action_text=action_text,
                target_gpu=target_gpu,
                code_format=str(code_format or "").strip(),
                hints=hints,
                hw_spec=hw_spec_text,
            )
        )
    if lang == "cuda":
        return (
            "You are implementing a SPECIFIC NEXT ACTION starting from the specification.\n\n"
            + CUDA_ACTION_PROMPT.format(
                definition=str(definition_text or "").strip(),
                base_code="(no base code; start from spec)",
                action_text=action_text,
                target_gpu=target_gpu,
                code_format=str(code_format or "").strip(),
                hints=CUDA_OPTIMIZATION_HINTS,
                hw_spec=hw_spec_text,
            )
        )
    if lang == "hlsl":
        return (
            "You are implementing a SPECIFIC NEXT ACTION starting from the specification.\n\n"
            + HLSL_ACTION_PROMPT.format(
                definition=str(definition_text or "").strip(),
                base_code="(no base code; start from spec)",
                action_text=action_text,
                target_gpu=target_gpu,
                code_format=str(code_format or "").strip(),
                hints=HLSL_OPTIMIZATION_HINTS,
                hw_spec=hw_spec_text,
            )
        )
    raise ValueError(f"Unsupported language for spec+action prompt: {language}")


def get_debug_and_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    action_text: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
    hw_spec: str = "",
) -> str:
    return get_debug_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        buggy_code=current_code,
        action_text=action_text,
        code_format=code_format,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
        hw_spec=hw_spec,
    )


def get_debug_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    buggy_code: str,
    action_text: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    hw_spec: str = "",
) -> str:
    """Task-agnostic variant: accepts rendered definition + rendered trace logs."""
    lang = (language or "").lower()
    dr = int(debug_round)
    if dr < 1:
        dr = 1
    mr = int(max_rounds)
    if mr < 1:
        mr = 1
    if dr > mr:
        dr = mr
    hw_spec_text = str(hw_spec or "").strip()
    if lang == "triton":
        hints = XPU_TRITON_OPTIMIZATION_HINTS if _is_intel_gpu(target_gpu) else TRITON_OPTIMIZATION_HINTS
        return TRITON_DEBUG_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            buggy_code=buggy_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            action_text=action_text,
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=hints,
            hw_spec=hw_spec_text,
        )
    if lang == "cuda":
        return CUDA_DEBUG_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            buggy_code=buggy_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            action_text=action_text,
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    if lang == "hlsl":
        return HLSL_DEBUG_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            buggy_code=buggy_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            action_text=action_text,
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=HLSL_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    raise ValueError(f"Unsupported language for debug prompt: {language}")


def get_improve_from_spec_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    current_code: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    base_code: str = "(no base code; start from spec)",
    hw_spec: str = "",
) -> str:
    return get_improve_generated_code_prompt_from_text(
        language,
        definition_text=definition_text,
        trace_logs=trace_logs,
        base_code=base_code,
        current_code=current_code,
        code_format=code_format,
        debug_round=debug_round,
        max_rounds=max_rounds,
        target_gpu=target_gpu,
        perf_summary=perf_summary,
        hw_spec=hw_spec,
    )


def get_improve_generated_code_prompt_from_text(
    language: str,
    *,
    definition_text: str,
    trace_logs: str,
    base_code: str,
    current_code: str,
    code_format: str = "",
    debug_round: int,
    max_rounds: int = 5,
    target_gpu: str = "H100",
    perf_summary: str = "",
    hw_spec: str = "",
) -> str:
    """Task-agnostic variant: accepts rendered definition + rendered trace logs."""
    lang = (language or "").lower()
    dr = int(debug_round)
    if dr < 1:
        dr = 1
    mr = int(max_rounds)
    if mr < 1:
        mr = 1
    if dr > mr:
        dr = mr
    hw_spec_text = str(hw_spec or "").strip()
    if lang == "triton":
        hints = XPU_TRITON_OPTIMIZATION_HINTS if _is_intel_gpu(target_gpu) else TRITON_OPTIMIZATION_HINTS
        return TRITON_IMPROVE_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            current_code=current_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=hints,
            hw_spec=hw_spec_text,
        )
    if lang == "cuda":
        return CUDA_IMPROVE_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            current_code=current_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=CUDA_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    if lang == "hlsl":
        return HLSL_IMPROVE_PROMPT.format(
            definition=str(definition_text or "").strip(),
            base_code=base_code,
            current_code=current_code,
            perf_summary=str(perf_summary or "").strip() or "(none)",
            trace_logs=str(trace_logs or "").strip() or "(no logs)",
            debug_round=dr,
            max_rounds=mr,
            target_gpu=target_gpu,
            code_format=str(code_format or "").strip(),
            hints=HLSL_OPTIMIZATION_HINTS,
            hw_spec=hw_spec_text,
        )
    raise ValueError(f"Unsupported language for improve prompt: {language}")


