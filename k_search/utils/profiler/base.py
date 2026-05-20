"""Profiler abstraction and backend selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Profiler(ABC):
    """Abstract hardware profiler.

    A Profiler runs an external command (typically a profile-only Python
    subprocess) under a vendor-specific profiling tool (e.g. NVIDIA Nsight
    Compute, Intel VTune) and returns a normalized metrics dict that fits
    `EvalResult.profiler_metrics` (`{"kernels": [{...}, ...]}`).

    Concrete backends are responsible for:
      - Reporting whether the profiler binary is available on the host.
      - Wrapping the profiled command and parsing its output into a dict.
      - Rendering a short, prompt-friendly summary from a metrics dict.
    """

    #: Short backend identifier ("ncu", "vtune", "noop", ...).
    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool:
        """Return True iff the profiler is installed and usable on this host."""

    @abstractmethod
    def run(
        self,
        cmd: list[str],
        *,
        timeout: int = 120,
        verbose: bool = False,
    ) -> dict[str, Any] | None:
        """Profile `cmd` and return a metrics dict, or None on any failure.

        The returned dict shape is `{"kernels": [<per-kernel dict>, ...]}`
        so it slots directly into `EvalResult.profiler_metrics`.
        """

    @abstractmethod
    def summary_lines(self, metrics: dict[str, Any]) -> list[str]:
        """Render `metrics` into a short list of prompt-friendly lines."""


def select_profiler(device: str) -> Profiler:
    """Pick the right profiler implementation for the given device string.

    Examples:
        select_profiler("cuda:0")  -> NcuProfiler
        select_profiler("xpu:0")   -> NoopProfiler  (VTune support TBD)
        select_profiler("cpu")     -> NoopProfiler
    """
    backend = (device or "").split(":")[0].strip().lower()

    if backend == "cuda":
        from k_search.utils.profiler.ncu import NcuProfiler
        return NcuProfiler()

    # XPU / CPU / unknown: no profiler wired up yet.
    from k_search.utils.profiler.noop import NoopProfiler
    return NoopProfiler(backend=backend or "unknown")
