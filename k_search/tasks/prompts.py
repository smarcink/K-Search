"""Shared prompt blocks used by multiple task backends (cuda_kernel, device_bench, etc.)."""

from __future__ import annotations


def _cuda_xml_and_guidelines_block(*, target_gpu: str) -> str:
    tg = str(target_gpu or "H100")
    return f"""IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
- MUST include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- If the model has learnable parameters, expose a "set_params" function that accepts
  ALL model parameters as torch::Tensor positional args (in model.parameters() order)
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {tg}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Use PyTorch tensor API (torch::Tensor) for all tensor arguments in the "run" function
- Convert PyTorch tensors to CUDA pointers using .data_ptr<float>() or similar methods
- Ensure proper CUDA stream synchronization and error handling"""


def code_format_text(*, language: str, target_gpu: str) -> str:
    """
    Format/output guidelines used by world-model prompts (`{code_format}`).
    Keep this small and stable across phases.
    """
    lang = str(language or "").strip().lower()
    tg = str(target_gpu or "H100")
    if lang in ("triton", "python"):
        return _triton_wrapper_and_output_guidelines_block().strip()
    if lang == "cuda":
        return _cuda_xml_and_guidelines_block(target_gpu=tg).strip()
    return ""


def _triton_wrapper_and_output_guidelines_block() -> str:
    return """The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- Use math.log(2), math.pi, math.e instead of hex literals
- All code must be valid Python that passes ast.parse()

Triton output format guidelines:
- You MUST expose a "run" entry point function that can be called to execute the kernel.
- Return only the code (no explanations, no markdown formatting)."""
