#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernel.h"

static torch::Tensor g_weight_half;
static torch::Tensor g_bias_half;
static bool g_params_set = false;

void set_params(torch::Tensor weight, torch::Tensor bias) {
    auto w = weight.to(torch::kHalf).contiguous().cpu();
    auto b = bias.to(torch::kHalf).contiguous().cpu();
    g_weight_half = w;
    g_bias_half = b;
    set_weights(w.data_ptr<at::Half>(), b.data_ptr<at::Half>());
    g_params_set = true;
}

std::vector<torch::Tensor> run(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    auto in = input.contiguous();
    if (in.scalar_type() != torch::kHalf) {
        in = in.to(torch::kHalf);
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(in.device());
    auto output = torch::empty({1, OUT_C, OUT_H, OUT_W}, options);

    const __half* in_ptr = reinterpret_cast<const __half*>(in.data_ptr<at::Half>());
    __half* out_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_fused(in_ptr, out_ptr, stream);

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused conv+relu+avgpool");
    m.def("set_params", &set_params, "Set conv weights and bias");
}