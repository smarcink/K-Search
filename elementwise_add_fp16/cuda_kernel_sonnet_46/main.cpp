#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "kernel.h"

std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(y.dtype() == torch::kFloat16, "y must be fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same number of elements");

    torch::Tensor z = torch::empty_like(x);

    int n = static_cast<int>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_elementwise_add_fp16(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(z.data_ptr<at::Half>()),
        n,
        stream
    );

    return {z};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Vectorized fp16 elementwise add (128-bit loads, half2 arithmetic, 2x ILP)");
}