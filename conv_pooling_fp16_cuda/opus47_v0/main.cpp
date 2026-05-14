#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernel.h"

void launch_fused_conv_relu_pool_f32(
    const float* input,
    float* output,
    int N, int C_in, int H, int W,
    int C_out,
    cudaStream_t stream);

static int g_in_channels = 0;
static int g_out_channels = 0;
static bool g_params_set = false;
static at::ScalarType g_param_dtype = at::kFloat;

void set_params(torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    g_out_channels = weight.size(0);
    g_in_channels = weight.size(1);
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "weight must be 3x3");
    g_param_dtype = weight.scalar_type();

    auto w_contig = weight.contiguous();
    auto b_contig = bias.contiguous();

    auto w_half = w_contig.to(at::kHalf);
    auto b_half = b_contig.to(at::kHalf);

    set_conv_params(w_half.data_ptr<at::Half>(),
                    b_half.data_ptr<at::Half>(),
                    g_out_channels, g_in_channels);
    g_params_set = true;
}

std::vector<torch::Tensor> run(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(g_params_set, "params not set");

    auto x = input.contiguous();
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int OH = H / 2;
    int OW = W / 2;

    auto out = torch::empty({N, g_out_channels, OH, OW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::kHalf) {
        launch_fused_conv_relu_pool(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            N, C, H, W, g_out_channels, stream);
    } else if (x.scalar_type() == at::kFloat) {
        launch_fused_conv_relu_pool_f32(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W, g_out_channels, stream);
    } else {
        TORCH_CHECK(false, "input dtype must be fp16 or fp32");
    }

    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_params", &set_params, "Set conv weights and bias");
    m.def("run", &run, "Run fused conv+relu+pool");
}