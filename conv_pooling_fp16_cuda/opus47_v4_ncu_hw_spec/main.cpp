#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernel.h"

void set_params(torch::Tensor weight, torch::Tensor bias) {
    auto w = weight.to(torch::kHalf).contiguous().cuda();
    auto b = bias.to(torch::kHalf).contiguous().cuda();
    set_conv_params(w.data_ptr<at::Half>(), b.data_ptr<at::Half>());
}

std::vector<torch::Tensor> run(torch::Tensor input) {
    auto x = input.contiguous();
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "input must be fp16");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = 32;
    int PH = H / 2;
    int PW = W / 2;

    auto out = torch::empty({N, K, PH, PW}, x.options());

    launch_fused_conv_relu_pool(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N, C, H, W, K,
        c10::cuda::getCurrentCUDAStream()
    );

    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused Conv+ReLU+AvgPool");
    m.def("set_params", &set_params, "Set conv weight and bias");
}