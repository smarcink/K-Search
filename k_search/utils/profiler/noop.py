"""No-op profiler placeholder for backends without a wired-up profiler.

Used today for Intel XPU (VTune integration TBD) and any unknown backend.
Always reports unavailable so the task gracefully skips profiling.
"""

from __future__ import annotations

from typing import Any

from k_search.utils.profiler.base import Profiler


class NoopProfiler(Profiler):
    name = "noop"

    def __init__(self, *, backend: str = "unknown") -> None:
        self._backend = str(backend or "unknown")

    def available(self) -> bool:
        return False

    def run(
        self,
        cmd: list[str],
        *,
        timeout: int = 120,
        verbose: bool = False,
    ) -> dict[str, Any] | None:
        if verbose:
            print(f"[profiler] No profiler wired up for backend '{self._backend}'; skipping.")
        return None

    def summary_lines(self, metrics: dict[str, Any]) -> list[str]:
        return []
