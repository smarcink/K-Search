#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernel.h"

static bool g_params_set = false;

void set_params(torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    auto w = weight.contiguous().to(torch::kHalf);
    auto b = bias.contiguous().to(torch::kHalf);
    set_conv_params(reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(b.data_ptr<at::Half>()));
    g_params_set = true;
}

std::vector<torch::Tensor> run(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(g_params_set, "Call set_params first");
    auto x = input.contiguous().to(torch::kHalf);

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
    auto output = torch::empty({1, COUT, H_OUT, W_OUT}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_fused_conv_relu_pool(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        stream
    );

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused Conv+ReLU+AvgPool");
    m.def("set_params", &set_params, "Set conv weight and bias");
}