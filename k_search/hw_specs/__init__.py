"""Hardware specification system for K-Search.

Provides structured GPU specs (compute, memory, cache, matrix accelerators)
that are rendered into LLM prompts using vendor-native terminology.

Public API:
    HWSpec          — dataclass representing a GPU hardware spec
    HWSpecRegistry  — resolves target_gpu strings to HWSpec objects
    get_hw_spec()   — convenience function for registry lookup
"""

from k_search.hw_specs.spec import HWSpec, MatrixAccelOp
from k_search.hw_specs.registry import HWSpecRegistry, get_hw_spec

__all__ = [
    "HWSpec",
    "MatrixAccelOp",
    "HWSpecRegistry",
    "get_hw_spec",
]
