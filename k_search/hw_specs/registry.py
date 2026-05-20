"""HW Spec registry: resolves target_gpu strings to HWSpec objects.

Scans a directory of JSON spec files and matches by name/aliases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from k_search.hw_specs.spec import HWSpec

# Default bundled specs directory (sibling to this file).
_BUNDLED_SPECS_DIR = Path(__file__).parent / "specs"


class HWSpecRegistry:
    """Registry that maps GPU name strings to HWSpec objects."""

    def __init__(self, spec_dir: str | Path | None = None) -> None:
        self._spec_dir = Path(spec_dir) if spec_dir else _BUNDLED_SPECS_DIR
        self._specs: list[HWSpec] | None = None  # lazy-loaded

    def _load(self) -> list[HWSpec]:
        if self._specs is not None:
            return self._specs
        self._specs = []
        if not self._spec_dir.is_dir():
            return self._specs
        for p in sorted(self._spec_dir.glob("*.json")):
            try:
                self._specs.append(HWSpec.from_file(p))
            except Exception:
                continue
        return self._specs

    def resolve(self, target_gpu: str) -> HWSpec | None:
        """Fuzzy-match a target_gpu string to a loaded HWSpec.

        Matching is case-insensitive substring against name + aliases.
        Prefers exact matches, then shortest name that contains the query.
        """
        key = (target_gpu or "").strip().lower()
        if not key:
            return None
        matches: list[tuple[int, HWSpec]] = []
        for spec in self._load():
            candidates = [spec.name] + list(spec.aliases)
            for c in candidates:
                cl = c.lower()
                if key == cl:
                    return spec  # exact match — return immediately
                if key in cl or cl in key:
                    matches.append((len(c), spec))
                    break
        if matches:
            matches.sort(key=lambda t: t[0])
            return matches[0][1]
        return None

    def list_specs(self) -> list[HWSpec]:
        """Return all loaded specs."""
        return list(self._load())


# Module-level convenience -------------------------------------------------

_default_registry: HWSpecRegistry | None = None


def get_hw_spec(
    target_gpu: str,
    *,
    spec_dir: str | Path | None = None,
    spec_path: str | Path | None = None,
) -> Optional[HWSpec]:
    """Convenience: resolve a target_gpu string to an HWSpec.

    Args:
        target_gpu: GPU name string (e.g. "H100", "Intel Arc B580").
        spec_dir:   Optional directory to scan (defaults to bundled specs).
        spec_path:  If provided, load this specific file instead of registry lookup.
    """
    # Explicit path takes priority.
    if spec_path:
        return HWSpec.from_file(spec_path)

    global _default_registry
    if spec_dir:
        reg = HWSpecRegistry(spec_dir)
    else:
        if _default_registry is None:
            _default_registry = HWSpecRegistry()
        reg = _default_registry
    return reg.resolve(target_gpu)
