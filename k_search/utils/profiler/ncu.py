"""NVIDIA Nsight Compute (ncu) profiler integration for K-Search.

Provides structured profiling data from compiled CUDA kernels (hand-written
or Triton-generated). Returns None on any failure (ncu not found, timeout,
parse error) so callers can treat profiling as best-effort.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from k_search.utils.profiler.base import Profiler


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _find_ncu() -> Optional[str]:
    """Find the best available ncu binary, preferring newer versions."""
    ncu_search_paths = sorted(
        glob.glob("/opt/nvidia/nsight-compute/*/ncu"),
        reverse=True,  # newest version first
    )
    for path in ncu_search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("ncu")


NCU_PATH: Optional[str] = _find_ncu()
NCU_AVAILABLE: bool = NCU_PATH is not None


# ---------------------------------------------------------------------------
# Metrics to collect (stable ncu metric names)
# ---------------------------------------------------------------------------

# Selected for maximum diagnostic value with minimum profiling overhead.
# Using selective --metrics instead of --set full reduces profiling from ~60s to ~15s.
# Keep a small core set so we can retry if optional diagnostics are unsupported
# by a particular GPU / Nsight Compute version.
NCU_CORE_METRICS = [
    # Kernel duration. Used to rank multi-kernel reports by runtime contribution.
    "gpu__time_duration.sum",
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Compute throughput (% of peak)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory (DRAM) throughput (% of peak)
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # Achieved DRAM bandwidth (bytes/sec)
    "dram__bytes.sum.per_second",
    # Registers per thread
    "launch__registers_per_thread",
    # Shared memory per block (dynamic + static)
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static",
]

NCU_OPTIONAL_METRICS = [
    # L1 cache hit rate
    "l1tex__t_sector_hit_rate.pct",
    # L2 cache hit rate
    "lts__t_sector_hit_rate.pct",
    # Tensor core utilization
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # Stall reasons (top contributors to warp stalls)
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
]

NCU_METRICS = NCU_CORE_METRICS + NCU_OPTIONAL_METRICS

NCU_CORE_METRICS_STR = ",".join(NCU_CORE_METRICS)
NCU_METRICS_STR = ",".join(NCU_METRICS)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NcuMetrics:
    """Parsed profiler metrics from a single ncu profiling run."""

    sm_occupancy_pct: float | None = None
    compute_throughput_pct: float | None = None
    memory_throughput_pct: float | None = None
    l1_hit_rate_pct: float | None = None
    l2_hit_rate_pct: float | None = None
    achieved_bandwidth_gb_s: float | None = None
    tensor_core_utilization_pct: float | None = None
    registers_per_thread: int | None = None
    shared_mem_per_block_bytes: int | None = None
    duration_us: float | None = None
    top_stall_reasons: list[tuple[str, float]] = field(default_factory=list)
    raw_csv: str = ""
    kernel_name: str = ""


# ---------------------------------------------------------------------------
# Low-level subprocess + parsing
# ---------------------------------------------------------------------------

def _run_ncu_subprocess(
    cmd: list[str],
    *,
    timeout: int = 120,
    kernel_name_filter: str | None = None,
    verbose: bool = False,
) -> list[NcuMetrics] | None:
    """Run ncu on the given command and return parsed metrics, or None on failure."""
    if not NCU_AVAILABLE:
        if verbose:
            print("[ncu] SKIPPED: ncu binary not found on PATH")
        return None

    def _build_cmd(metrics_str: str) -> list[str]:
        ncu_cmd = [
            NCU_PATH,
            "--metrics", metrics_str,
            "--csv",
            "--target-processes", "all",
            # Lock clocks for reproducible measurements
            "--clock-control", "base",
            # Only profile kernels between cudaProfilerStart/Stop markers
            "--profile-from-start", "off",
        ]
        if kernel_name_filter:
            ncu_cmd.extend(["--kernel-name", kernel_name_filter])
        ncu_cmd.append("--")
        ncu_cmd.extend(cmd)
        return ncu_cmd

    attempts = [("full", NCU_METRICS_STR)]
    if NCU_CORE_METRICS_STR != NCU_METRICS_STR:
        attempts.append(("core", NCU_CORE_METRICS_STR))

    for label, metrics_str in attempts:
        ncu_cmd = _build_cmd(metrics_str)

        if verbose:
            print(f"[ncu] Running ({label} metrics): {' '.join(ncu_cmd)}")

        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"[ncu] FAILED ({label} metrics): timed out after {timeout}s")
            return None
        except Exception as e:
            if verbose:
                print(f"[ncu] FAILED ({label} metrics): exception: {e}")
            return None

        if verbose:
            print(f"[ncu] Return code ({label} metrics): {result.returncode}")
            if result.stderr.strip():
                print(f"[ncu] stderr ({label}, last 2000 chars):\n{result.stderr[-2000:]}")
            if not result.stdout.strip():
                print(f"[ncu] stdout ({label}): (empty)")
            else:
                print(f"[ncu] stdout length ({label}): {len(result.stdout)} chars")

        if result.returncode != 0:
            if label == "full" and len(attempts) > 1 and verbose:
                print("[ncu] Full metric collection failed; retrying with core metrics only")
            continue

        csv_output = result.stdout
        if not csv_output.strip():
            if label == "full" and len(attempts) > 1 and verbose:
                print("[ncu] Full metric collection produced no CSV; retrying with core metrics only")
            continue

        parsed = _parse_ncu_csv(csv_output)
        if parsed:
            return parsed
        if label == "full" and len(attempts) > 1 and verbose:
            print("[ncu] Full metric CSV was not parseable; retrying with core metrics only")

    return None


# Substrings (case-insensitive) that identify PyTorch / cuBLAS / cuDNN
# infrastructure kernels we want to hide from the profiler report. The point
# of profiling is to give the LLM feedback on the kernel IT generated, not on
# the surrounding torch glue (residual adds, casts, layernorm helpers, etc.).
_INFRA_KERNEL_PATTERNS: tuple[str, ...] = (
    "at::native::",
    "at::elementwise_kernel",
    "at::vectorized_elementwise_kernel",
    "at::gpu_kernel",
    "at::reduce_kernel",
    "aten::",
    "cudnn::",
    "cudnn_",
    "cublas",
    "cutlass::",
    "thrust::",
    "void at::",          # most templated at:: launches
    "scalar_tensor",
    "_kernel_impl_nocast",
)


def _is_infrastructure_kernel(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(p.lower() in n for p in _INFRA_KERNEL_PATTERNS)


def _duration_to_us(value: float, unit: str | None) -> float | None:
    """Convert an NCU duration metric value to microseconds."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    u = str(unit or "").strip().lower()
    if u in ("ns", "nsecond", "nseconds", "nanosecond", "nanoseconds"):
        return val / 1000.0
    if u in ("us", "usecond", "useconds", "microsecond", "microseconds"):
        return val
    if u in ("ms", "msecond", "mseconds", "millisecond", "milliseconds"):
        return val * 1000.0
    if u in ("s", "sec", "second", "seconds"):
        return val * 1_000_000.0
    # Nsight Compute commonly reports gpu__time_duration.sum in nanoseconds.
    return val / 1000.0


def _parse_ncu_csv(csv_text: str) -> list[NcuMetrics] | None:
    """Parse ncu --csv output into a list of NcuMetrics (one per kernel launch)."""
    try:
        lines = csv_text.strip().splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"') or line.startswith("ID"):
                header_idx = i
                break
        if header_idx is None:
            header_idx = 0

        csv_content = "\n".join(lines[header_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        kernels: dict[str, dict[str, float]] = {}
        kernel_names: dict[str, str] = {}
        kernel_units: dict[str, dict[str, str]] = {}

        for row in reader:
            launch_id = row.get("ID", "").strip()
            metric_name = row.get("Metric Name", "").strip()
            metric_value_str = row.get("Metric Value", "").strip()
            metric_unit = (row.get("Metric Unit", "") or row.get("Unit", "") or "").strip()
            if not launch_id or not metric_name or not metric_value_str:
                continue

            kernel_names.setdefault(launch_id, row.get("Kernel Name", "").strip())

            try:
                clean_val = metric_value_str.replace(",", "")
                val = float(clean_val)
            except (ValueError, TypeError):
                continue

            kernels.setdefault(launch_id, {})[metric_name] = val
            kernel_units.setdefault(launch_id, {})[metric_name] = metric_unit

        if not kernels:
            return None

        throughput_metric = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        result: list[NcuMetrics] = []

        for kid in kernels:
            metrics_by_name = kernels[kid]
            kernel_name = kernel_names.get(kid, "")

            def _get(name: str, _m=metrics_by_name) -> float | None:
                val = _m.get(name)
                return val if val is not None else None

            def _get_int(name: str, _m=metrics_by_name) -> int | None:
                val = _m.get(name)
                return int(val) if val is not None else None

            m = NcuMetrics()
            m.kernel_name = kernel_name

            duration = _get("gpu__time_duration.sum")
            if duration is not None:
                m.duration_us = _duration_to_us(duration, kernel_units.get(kid, {}).get("gpu__time_duration.sum"))

            m.sm_occupancy_pct = _get("sm__warps_active.avg.pct_of_peak_sustained_active")
            m.compute_throughput_pct = _get(throughput_metric)
            m.memory_throughput_pct = _get("dram__throughput.avg.pct_of_peak_sustained_elapsed")
            m.l1_hit_rate_pct = _get("l1tex__t_sector_hit_rate.pct")
            m.l2_hit_rate_pct = _get("lts__t_sector_hit_rate.pct")
            m.tensor_core_utilization_pct = _get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")

            bw = _get("dram__bytes.sum.per_second")
            if bw is not None:
                m.achieved_bandwidth_gb_s = bw / 1e9

            m.registers_per_thread = _get_int("launch__registers_per_thread")

            dyn = _get("launch__shared_mem_per_block_dynamic")
            static = _get("launch__shared_mem_per_block_static")
            if dyn is not None or static is not None:
                m.shared_mem_per_block_bytes = int((dyn or 0) + (static or 0))

            stall_metrics = [
                ("long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
                ("short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
                ("wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
                ("not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
                ("math_pipe_throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
            ]
            stalls: list[tuple[str, float]] = []
            for reason_name, metric_name in stall_metrics:
                val = _get(metric_name)
                if val is not None and val > 0:
                    stalls.append((reason_name, val))
            stalls.sort(key=lambda x: x[1], reverse=True)
            m.top_stall_reasons = stalls[:3]

            result.append(m)

        # Filter out PyTorch / library infrastructure kernels so the report
        # focuses on the user-generated kernel(s) (Triton, hand-written CUDA, etc.).
        # If the filter would drop everything, keep the original list.
        filtered = [m for m in result if not _is_infrastructure_kernel(m.kernel_name)]
        if filtered:
            result = filtered

        # Sort by runtime contribution when available; throughput alone does
        # not identify the kernel that dominates end-to-end latency.
        if any(m.duration_us is not None for m in result):
            result.sort(key=lambda m: m.duration_us or 0.0, reverse=True)
        else:
            result.sort(key=lambda m: m.compute_throughput_pct or 0.0, reverse=True)

        if result:
            result[0].raw_csv = csv_text[:3000]

        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering for prompts
# ---------------------------------------------------------------------------

def _render_single_kernel(metrics: NcuMetrics) -> list[str]:
    """Render a single kernel's metrics."""
    lines: list[str] = []

    mem_pct = metrics.memory_throughput_pct
    comp_pct = metrics.compute_throughput_pct

    if metrics.duration_us is not None:
        if metrics.duration_us >= 1000.0:
            lines.append(f"- profiler_duration: {metrics.duration_us / 1000.0:.3f} ms")
        else:
            lines.append(f"- profiler_duration: {metrics.duration_us:.1f} us")

    if mem_pct is not None:
        lines.append(f"- profiler_memory_throughput: {mem_pct:.1f}% of peak")
    if comp_pct is not None:
        lines.append(f"- profiler_compute_throughput: {comp_pct:.1f}% of peak")

    tc = metrics.tensor_core_utilization_pct
    if tc is not None:
        tc_note = " — NOT using tensor cores" if tc < 1.0 else ""
        lines.append(f"- profiler_tensor_core_util: {tc:.1f}%{tc_note}")

    occ = metrics.sm_occupancy_pct
    if occ is not None:
        lines.append(f"- profiler_occupancy: {occ:.1f}%")

    if metrics.achieved_bandwidth_gb_s is not None:
        lines.append(f"- profiler_achieved_bandwidth: {metrics.achieved_bandwidth_gb_s:.1f} GB/s")

    if metrics.l1_hit_rate_pct is not None:
        lines.append(f"- profiler_l1_hit_rate: {metrics.l1_hit_rate_pct:.1f}%")
    if metrics.l2_hit_rate_pct is not None:
        lines.append(f"- profiler_l2_hit_rate: {metrics.l2_hit_rate_pct:.1f}%")

    if metrics.registers_per_thread is not None:
        lines.append(f"- profiler_registers_per_thread: {metrics.registers_per_thread}")
    if metrics.shared_mem_per_block_bytes is not None:
        lines.append(f"- profiler_shared_mem_per_block: {metrics.shared_mem_per_block_bytes} bytes")

    if metrics.top_stall_reasons:
        stalls_str = ", ".join(f"{name} ({val:.2f} ratio)" for name, val in metrics.top_stall_reasons)
        lines.append(f"- profiler_top_stalls: {stalls_str}")

    diagnosis = _diagnose_kernel(metrics)
    if diagnosis:
        lines.append("- profiler_diagnosis: " + "; ".join(diagnosis))

    return lines


def _diagnose_kernel(metrics: NcuMetrics) -> list[str]:
    """Return cautious, prompt-friendly profiler interpretations."""
    notes: list[str] = []

    occ = metrics.sm_occupancy_pct
    regs = metrics.registers_per_thread
    smem = metrics.shared_mem_per_block_bytes
    mem_pct = metrics.memory_throughput_pct
    comp_pct = metrics.compute_throughput_pct
    tc = metrics.tensor_core_utilization_pct

    if occ is not None and occ < 35.0:
        if regs is not None and regs >= 96:
            notes.append("low occupancy likely tied to high register pressure")
        elif smem is not None and smem >= 48 * 1024:
            notes.append("low occupancy likely tied to shared-memory footprint")
        else:
            notes.append("low occupancy may limit latency hiding")

    if tc is not None and tc < 1.0:
        notes.append("tensor cores inactive; relevant if this kernel has fp16/bf16 matmul-like work")

    stall_map = {name: val for name, val in metrics.top_stall_reasons}
    if stall_map.get("long_scoreboard", 0.0) >= 1.0:
        notes.append("long scoreboard stalls suggest memory dependency latency")
    elif stall_map.get("short_scoreboard", 0.0) >= 1.0:
        notes.append("short scoreboard stalls suggest dependency latency")
    if stall_map.get("math_pipe_throttle", 0.0) >= 1.0:
        notes.append("math pipeline throttle suggests compute issue pressure")
    if stall_map.get("wait", 0.0) >= 1.0:
        notes.append("wait stalls suggest synchronization or scheduling latency")

    if mem_pct is not None and comp_pct is not None:
        if mem_pct >= 60.0 and mem_pct > comp_pct * 1.3:
            notes.append("DRAM throughput is high, so memory bandwidth is likely important")
        elif comp_pct >= 60.0 and comp_pct > mem_pct * 1.3:
            notes.append("SM throughput is high, so compute pipeline efficiency is important")
        elif mem_pct < 20.0 and comp_pct < 30.0:
            notes.append("low DRAM and SM throughput points away from simple bandwidth saturation")

    deduped: list[str] = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
        if len(deduped) >= 3:
            break
    return deduped


def render_profiler_summary(metrics: NcuMetrics | list[NcuMetrics]) -> str:
    """Render NcuMetrics into a concise, prompt-friendly text block."""
    if isinstance(metrics, NcuMetrics):
        kernel_list = [metrics]
    else:
        kernel_list = metrics

    all_lines: list[str] = []
    multi = len(kernel_list) > 1

    total_duration_us = sum(m.duration_us or 0.0 for m in kernel_list)

    for idx, km in enumerate(kernel_list):
        if multi:
            label = km.kernel_name or f"kernel_{idx}"
            if len(label) > 80:
                label = label[:77] + "..."
            if total_duration_us > 0 and km.duration_us is not None:
                share = 100.0 * km.duration_us / total_duration_us
                all_lines.append(f"[Kernel {idx}: {label} ({share:.1f}% profiled time)]")
            else:
                all_lines.append(f"[Kernel {idx}: {label}]")

        lines = _render_single_kernel(km)
        all_lines.extend(lines)

        if multi and idx < len(kernel_list) - 1:
            all_lines.append("")

    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# Dict <-> NcuMetrics conversion
# ---------------------------------------------------------------------------

def _single_metrics_to_dict(metrics: NcuMetrics) -> dict[str, Any]:
    """Convert a single NcuMetrics to a flat dict."""
    d: dict[str, Any] = {}
    if metrics.sm_occupancy_pct is not None:
        d["sm_occupancy_pct"] = round(metrics.sm_occupancy_pct, 2)
    if metrics.compute_throughput_pct is not None:
        d["compute_throughput_pct"] = round(metrics.compute_throughput_pct, 2)
    if metrics.memory_throughput_pct is not None:
        d["memory_throughput_pct"] = round(metrics.memory_throughput_pct, 2)
    if metrics.l1_hit_rate_pct is not None:
        d["l1_hit_rate_pct"] = round(metrics.l1_hit_rate_pct, 2)
    if metrics.l2_hit_rate_pct is not None:
        d["l2_hit_rate_pct"] = round(metrics.l2_hit_rate_pct, 2)
    if metrics.achieved_bandwidth_gb_s is not None:
        d["achieved_bandwidth_gb_s"] = round(metrics.achieved_bandwidth_gb_s, 2)
    if metrics.tensor_core_utilization_pct is not None:
        d["tensor_core_utilization_pct"] = round(metrics.tensor_core_utilization_pct, 2)
    if metrics.registers_per_thread is not None:
        d["registers_per_thread"] = metrics.registers_per_thread
    if metrics.shared_mem_per_block_bytes is not None:
        d["shared_mem_per_block_bytes"] = metrics.shared_mem_per_block_bytes
    if metrics.duration_us is not None:
        d["duration_us"] = round(metrics.duration_us, 3)
    if metrics.top_stall_reasons:
        d["top_stall_reasons"] = [(name, round(val, 4)) for name, val in metrics.top_stall_reasons]
    if metrics.kernel_name:
        d["kernel_name"] = metrics.kernel_name
    return d


def metrics_to_dict(metrics: NcuMetrics | list[NcuMetrics]) -> dict[str, Any]:
    """Convert NcuMetrics to `EvalResult.profiler_metrics` shape."""
    if isinstance(metrics, NcuMetrics):
        kernel_list = [metrics]
    else:
        kernel_list = metrics
    return {"kernels": [_single_metrics_to_dict(m) for m in kernel_list]}


def _dict_to_metrics(metrics_dict: dict[str, Any]) -> list[NcuMetrics]:
    """Reconstruct NcuMetrics list from the dict shape stored on EvalResult."""
    kernel_dicts = metrics_dict.get("kernels", [])
    if not kernel_dicts:
        # Legacy single-kernel format (just in case)
        kernel_dicts = [metrics_dict]

    out: list[NcuMetrics] = []
    for kd in kernel_dicts:
        out.append(
            NcuMetrics(
                sm_occupancy_pct=kd.get("sm_occupancy_pct"),
                compute_throughput_pct=kd.get("compute_throughput_pct"),
                memory_throughput_pct=kd.get("memory_throughput_pct"),
                l1_hit_rate_pct=kd.get("l1_hit_rate_pct"),
                l2_hit_rate_pct=kd.get("l2_hit_rate_pct"),
                achieved_bandwidth_gb_s=kd.get("achieved_bandwidth_gb_s"),
                tensor_core_utilization_pct=kd.get("tensor_core_utilization_pct"),
                registers_per_thread=kd.get("registers_per_thread"),
                shared_mem_per_block_bytes=kd.get("shared_mem_per_block_bytes"),
                duration_us=kd.get("duration_us"),
                top_stall_reasons=kd.get("top_stall_reasons", []),
                kernel_name=kd.get("kernel_name", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Profiler interface implementation
# ---------------------------------------------------------------------------

class NcuProfiler(Profiler):
    """NVIDIA Nsight Compute profiler backend."""

    name = "ncu"

    def available(self) -> bool:
        return NCU_AVAILABLE

    def run(
        self,
        cmd: list[str],
        *,
        timeout: int = 120,
        verbose: bool = False,
    ) -> dict[str, Any] | None:
        ncu_metrics = _run_ncu_subprocess(cmd, timeout=timeout, verbose=verbose)
        if ncu_metrics is None:
            return None
        return metrics_to_dict(ncu_metrics)

    def summary_lines(self, metrics: dict[str, Any]) -> list[str]:
        if not metrics:
            return []
        try:
            kernel_list = _dict_to_metrics(metrics)
            rendered = render_profiler_summary(kernel_list)
            return rendered.splitlines() if rendered else []
        except Exception:
            return []
