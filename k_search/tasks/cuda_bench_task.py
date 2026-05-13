"""CUDA Bench Task — backward-compat shim.

This module re-exports DeviceBenchTask as CudaBenchTask for existing code
that imports from ``k_search.tasks.cuda_bench_task``.  All logic now lives
in ``k_search.tasks.device_bench_task``.
"""

from k_search.tasks.device_bench_task import DeviceBenchTask as CudaBenchTask  # noqa: F401

__all__ = ["CudaBenchTask"]
