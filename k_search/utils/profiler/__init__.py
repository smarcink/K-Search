"""Hardware profiler abstractions for K-Search.

Backend-pluggable profiling: NCU on NVIDIA CUDA, no-op placeholder for XPU
(VTune integration TBD), and easy to extend with new backends.

Public API:
    Profiler           — abstract base class
    select_profiler()  — factory keyed on device string ("cuda:0", "xpu:0", ...)
    NcuProfiler        — NVIDIA Nsight Compute implementation
    NoopProfiler       — placeholder for backends without profiling support
"""

from k_search.utils.profiler.base import Profiler, select_profiler
from k_search.utils.profiler.ncu import NcuProfiler
from k_search.utils.profiler.noop import NoopProfiler

__all__ = [
    "Profiler",
    "select_profiler",
    "NcuProfiler",
    "NoopProfiler",
]
