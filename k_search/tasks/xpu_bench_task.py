"""XPU Bench Task — backward-compat shim.

This module re-exports DeviceBenchTask as XpuBenchTask for existing code
that imports from ``k_search.tasks.xpu_bench_task``.  All logic now lives
in ``k_search.tasks.device_bench_task``.
"""

from k_search.tasks.device_bench_task import DeviceBenchTask as XpuBenchTask  # noqa: F401

__all__ = ["XpuBenchTask"]
