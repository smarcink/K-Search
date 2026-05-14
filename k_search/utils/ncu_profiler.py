"""NVIDIA Nsight Compute (ncu) profiler integration for K-Search.

Provides structured profiling data from compiled CUDA kernels.
Gracefully returns None if ncu is unavailable or profiling fails.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _find_ncu() -> Optional[str]:
    """Find the best available ncu binary, preferring newer versions."""
    # Check common install paths for newer ncu versions first
    ncu_search_paths = sorted(
        glob.glob("/opt/nvidia/nsight-compute/*/ncu"),
        reverse=True,  # newest version first
    )
    for path in ncu_search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    # Fall back to PATH
    return shutil.which("ncu")


NCU_PATH: Optional[str] = _find_ncu()
NCU_AVAILABLE: bool = NCU_PATH is not None


# ---------------------------------------------------------------------------
# Metrics to collect (stable ncu metric names)
# ---------------------------------------------------------------------------

# These are selected for maximum diagnostic value with minimum profiling overhead.
# Using selective --metrics instead of --set full reduces profiling from ~60s to ~15s.
NCU_METRICS = [
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Compute throughput (% of peak)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory (DRAM) throughput (% of peak)
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # L1 cache hit rate
    "l1tex__t_sector_hit_rate.pct",
    # L2 cache hit rate
    "lts__t_sector_hit_rate.pct",
    # Achieved DRAM bandwidth (bytes/sec)
    "dram__bytes.sum.per_second",
    # Tensor core utilization
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # Registers per thread
    "launch__registers_per_thread",
    # Shared memory per block (dynamic + static)
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static",
    # Stall reasons (top contributors to warp stalls)
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
]

NCU_METRICS_STR = ",".join(NCU_METRICS)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NcuMetrics:
    """Parsed profiler metrics from a single ncu profiling run."""

    # Occupancy: achieved warps as % of peak
    sm_occupancy_pct: float | None = None
    # Compute SM throughput as % of peak
    compute_throughput_pct: float | None = None
    # DRAM throughput as % of peak
    memory_throughput_pct: float | None = None
    # Cache hit rates
    l1_hit_rate_pct: float | None = None
    l2_hit_rate_pct: float | None = None
    # Achieved DRAM bandwidth in GB/s
    achieved_bandwidth_gb_s: float | None = None
    # Tensor core pipe utilization as % of peak
    tensor_core_utilization_pct: float | None = None
    # Launch parameters
    registers_per_thread: int | None = None
    shared_mem_per_block_bytes: int | None = None
    # Stall reasons: list of (reason_name, ratio) sorted desc
    top_stall_reasons: list[tuple[str, float]] = field(default_factory=list)
    # Raw CSV output (bounded) for debugging
    raw_csv: str = ""
    # Kernel name that was profiled
    kernel_name: str = ""


# ---------------------------------------------------------------------------
# Profiling entry point
# ---------------------------------------------------------------------------

def run_ncu_profile(
    cmd: list[str],
    *,
    timeout: int = 120,
    kernel_name_filter: str | None = None,
    verbose: bool = False,
) -> NcuMetrics | None:
    """Run ncu on the given command and parse metrics.

    Args:
        cmd: Command to profile (e.g. [sys.executable, "eval_script.py", ...])
        timeout: Max seconds for profiling to complete
        kernel_name_filter: If set, only profile kernels matching this substring
        verbose: If True, print diagnostic info on failure

    Returns:
        NcuMetrics on success, None on any failure (ncu not found, timeout, parse error)
    """
    if not NCU_AVAILABLE:
        if verbose:
            print("[ncu] SKIPPED: ncu binary not found on PATH")
        return None

    ncu_cmd = [
        NCU_PATH,
        "--metrics", NCU_METRICS_STR,
        "--csv",
        "--target-processes", "all",
    ]

    if kernel_name_filter:
        ncu_cmd.extend(["--kernel-name", kernel_name_filter])

    ncu_cmd.append("--")
    ncu_cmd.extend(cmd)

    if verbose:
        print(f"[ncu] Running: {' '.join(ncu_cmd)}")

    try:
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[ncu] FAILED: timed out after {timeout}s")
        return None
    except Exception as e:
        if verbose:
            print(f"[ncu] FAILED: exception: {e}")
        return None

    if verbose:
        print(f"[ncu] Return code: {result.returncode}")
        if result.stderr.strip():
            print(f"[ncu] stderr (last 2000 chars):\n{result.stderr[-2000:]}")
        if not result.stdout.strip():
            print(f"[ncu] stdout: (empty)")
        else:
            print(f"[ncu] stdout length: {len(result.stdout)} chars")

    if result.returncode != 0:
        return None

    csv_output = result.stdout
    if not csv_output.strip():
        return None

    return _parse_ncu_csv(csv_output)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_ncu_csv(csv_text: str) -> NcuMetrics | None:
    """Parse ncu --csv output into NcuMetrics.

    ncu CSV format has rows like:
        "ID","Process ID","Process Name","Host Name","Kernel Name","...","Metric Name","Metric Unit","Metric Value"
    Each row is one metric for one kernel invocation.
    We aggregate across kernel invocations (take the first kernel launch or average).
    """
    try:
        # Find the header line (starts with "ID" typically)
        lines = csv_text.strip().splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"') or line.startswith("ID"):
                header_idx = i
                break
        if header_idx is None:
            # Try treating entire text as CSV
            header_idx = 0

        csv_content = "\n".join(lines[header_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        # Collect all metric values by metric name (first kernel invocation)
        metrics_by_name: dict[str, list[float]] = {}
        kernel_name = ""

        for row in reader:
            metric_name = row.get("Metric Name", "").strip()
            metric_value_str = row.get("Metric Value", "").strip()
            if not metric_name or not metric_value_str:
                continue

            # Capture kernel name from first row
            if not kernel_name:
                kernel_name = row.get("Kernel Name", "").strip()

            # Parse numeric value (ncu may use commas as thousand separators)
            try:
                clean_val = metric_value_str.replace(",", "")
                val = float(clean_val)
            except (ValueError, TypeError):
                continue

            metrics_by_name.setdefault(metric_name, []).append(val)

        if not metrics_by_name:
            return None

        def _avg(name: str) -> float | None:
            vals = metrics_by_name.get(name)
            if not vals:
                return None
            return sum(vals) / len(vals)

        def _first_int(name: str) -> int | None:
            vals = metrics_by_name.get(name)
            if not vals:
                return None
            return int(vals[0])

        # Build metrics
        m = NcuMetrics()
        m.kernel_name = kernel_name
        m.raw_csv = csv_text[:3000]  # bounded for debugging

        m.sm_occupancy_pct = _avg("sm__warps_active.avg.pct_of_peak_sustained_active")
        m.compute_throughput_pct = _avg("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        m.memory_throughput_pct = _avg("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        m.l1_hit_rate_pct = _avg("l1tex__t_sector_hit_rate.pct")
        m.l2_hit_rate_pct = _avg("lts__t_sector_hit_rate.pct")
        m.tensor_core_utilization_pct = _avg("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")

        # Bandwidth: bytes/sec → GB/s
        bw = _avg("dram__bytes.sum.per_second")
        if bw is not None:
            m.achieved_bandwidth_gb_s = bw / 1e9

        m.registers_per_thread = _first_int("launch__registers_per_thread")

        # Shared memory: dynamic + static
        dyn = _avg("launch__shared_mem_per_block_dynamic")
        static = _avg("launch__shared_mem_per_block_static")
        if dyn is not None or static is not None:
            m.shared_mem_per_block_bytes = int((dyn or 0) + (static or 0))

        # Stall reasons
        stall_metrics = [
            ("long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
            ("short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
            ("wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
            ("not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
            ("math_pipe_throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
        ]
        stalls: list[tuple[str, float]] = []
        for reason_name, metric_name in stall_metrics:
            val = _avg(metric_name)
            if val is not None and val > 0:
                stalls.append((reason_name, val))
        stalls.sort(key=lambda x: x[1], reverse=True)
        m.top_stall_reasons = stalls[:3]

        return m

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering for prompts
# ---------------------------------------------------------------------------

def render_profiler_summary(metrics: NcuMetrics) -> str:
    """Render NcuMetrics into a concise, prompt-friendly text block."""
    lines: list[str] = []

    # Determine if memory-bound or compute-bound
    mem_pct = metrics.memory_throughput_pct
    comp_pct = metrics.compute_throughput_pct
    bound_hint = ""
    if mem_pct is not None and comp_pct is not None:
        if mem_pct > comp_pct * 1.5:
            bound_hint = " — kernel is MEMORY-BOUND"
        elif comp_pct > mem_pct * 1.5:
            bound_hint = " — kernel is COMPUTE-BOUND"

    if mem_pct is not None:
        lines.append(f"- profiler_memory_throughput: {mem_pct:.1f}% of peak{bound_hint if 'MEMORY' in bound_hint else ''}")
    if comp_pct is not None:
        lines.append(f"- profiler_compute_throughput: {comp_pct:.1f}% of peak{bound_hint if 'COMPUTE' in bound_hint else ''}")

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
        stalls_str = ", ".join(f"{name} ({val:.1%})" for name, val in metrics.top_stall_reasons)
        lines.append(f"- profiler_top_stalls: {stalls_str}")

    return "\n".join(lines)


def metrics_to_dict(metrics: NcuMetrics) -> dict[str, Any]:
    """Convert NcuMetrics to a flat dict for EvalResult.profiler_metrics."""
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
    if metrics.top_stall_reasons:
        d["top_stall_reasons"] = [(name, round(val, 4)) for name, val in metrics.top_stall_reasons]
    if metrics.kernel_name:
        d["kernel_name"] = metrics.kernel_name
    return d
