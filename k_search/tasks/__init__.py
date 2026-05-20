"""Task adapters (evaluation backends) for k-search.

Keep this module **lightweight**:
- Importing `k_search.tasks.<something>` triggers `k_search.tasks.__init__` first.
- Avoid importing heavy optional deps at import time.
"""

from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages, Task, code_from_solution

__all__ = [
    "BuildSpec",
    "EvalResult",
    "Solution",
    "SourceFile",
    "SupportedLanguages",
    "Task",
    "code_from_solution",
]



