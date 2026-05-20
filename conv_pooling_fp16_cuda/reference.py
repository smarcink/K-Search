"""Conv2d + ReLU + AvgPool2d reference for CudaKernelTask.

This reference compiles the initial CUDA kernel (initial_kernel/) and uses it
as the ground-truth implementation.  K-Search will optimize a new CUDA kernel
to beat this one in performance while matching its outputs.

Pipeline:  x -> Conv2d(3x3, pad=1) -> ReLU -> AvgPool2d(2x2) -> out
"""

import os
import torch
import torch.nn as nn
import torch.utils.cpp_extension
from pathlib import Path


# Compile the initial CUDA kernel as the reference implementation
# Handle both direct execution (__file__ available) and exec() loading
try:
    _THIS_DIR = Path(os.path.abspath(__file__)).parent
except NameError:
    # When loaded via exec(compile(code, path, ...)), __file__ isn't set
    # but the compile path is available in the code object
    import inspect
    _frame = inspect.currentframe()
    _THIS_DIR = Path(os.path.abspath(_frame.f_code.co_filename)).parent if _frame else Path.cwd()
_KERNEL_DIR = _THIS_DIR / "initial_kernel"


def _compile_reference_kernel():
    """Compile the initial CUDA kernel for use as reference."""
    kernel_h = (_KERNEL_DIR / "kernel.h").read_text()
    kernel_cu = (_KERNEL_DIR / "kernel.cu").read_text()
    main_cpp = (_KERNEL_DIR / "main.cpp").read_text()

    build_dir = str(_KERNEL_DIR / "build")
    os.makedirs(build_dir, exist_ok=True)
    Path(build_dir, "kernel.h").write_text(kernel_h)

    module = torch.utils.cpp_extension.load_inline(
        name="ref_conv_pool_cuda",
        cpp_sources=[main_cpp],
        cuda_sources=[kernel_cu],
        extra_include_paths=[str(_KERNEL_DIR), build_dir],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
        build_directory=build_dir,
    )
    return module


_ref_module = None


def _get_ref_module():
    global _ref_module
    if _ref_module is None:
        _ref_module = _compile_reference_kernel()
    return _ref_module


class Model(nn.Module):
    """Reference model backed by compiled CUDA kernel."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Keep a real Conv2d for its weight/bias parameters
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
        )
        self._params_set = False

    def _ensure_params(self):
        if not self._params_set:
            mod = _get_ref_module()
            w = self.conv.weight.data
            b = self.conv.bias.data if self.conv.bias is not None else torch.Tensor()
            mod.set_params(w, b)
            self._params_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_params()
        mod = _get_ref_module()
        out = mod.run(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def get_inputs():
    return [torch.randn(1, 16, 720, 1280, dtype=torch.float16)]


def get_init_inputs():
    return [16, 32]  # in_channels, out_channels


if __name__ == "__main__":
    model = Model(*get_init_inputs()).half().cuda().eval()
    inputs = [x.cuda() for x in get_inputs()]
    with torch.no_grad():
        out = model(*inputs)
    print(f"Input:  {inputs[0].shape}  dtype={inputs[0].dtype}")
    print(f"Output: {out.shape}  dtype={out.dtype}")
