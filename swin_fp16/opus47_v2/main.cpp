#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernel.h"

static torch::Tensor g_rpe_bias;
static torch::Tensor g_norm1_w, g_norm1_b;
static torch::Tensor g_qkv_w, g_qkv_b;
static torch::Tensor g_proj_w, g_proj_b;
static torch::Tensor g_norm2_w, g_norm2_b;
static torch::Tensor g_fc1_w, g_fc1_b;
static torch::Tensor g_fc2_w, g_fc2_b;

void set_params(
    torch::Tensor rpe_bias,
    torch::Tensor norm1_w, torch::Tensor norm1_b,
    torch::Tensor qkv_w, torch::Tensor qkv_b,
    torch::Tensor proj_w, torch::Tensor proj_b,
    torch::Tensor norm2_w, torch::Tensor norm2_b,
    torch::Tensor fc1_w, torch::Tensor fc1_b,
    torch::Tensor fc2_w, torch::Tensor fc2_b)
{
    g_rpe_bias = rpe_bias.contiguous();
    g_norm1_w = norm1_w.contiguous();
    g_norm1_b = norm1_b.contiguous();
    g_qkv_w = qkv_w.contiguous();
    g_qkv_b = qkv_b.contiguous();
    g_proj_w = proj_w.contiguous();
    g_proj_b = proj_b.contiguous();
    g_norm2_w = norm2_w.contiguous();
    g_norm2_b = norm2_b.contiguous();
    g_fc1_w = fc1_w.contiguous();
    g_fc1_b = fc1_b.contiguous();
    g_fc2_w = fc2_w.contiguous();
    g_fc2_b = fc2_b.contiguous();
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    auto xc = x.contiguous();
    int B = xc.size(0);
    int H = xc.size(1);
    int W = xc.size(2);
    int C = xc.size(3);
    auto out = torch::empty_like(xc);

    launch_swin_block(
        reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_norm1_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_norm1_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_qkv_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_qkv_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_proj_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_proj_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_norm2_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_norm2_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_fc1_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_fc1_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_fc2_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_fc2_b.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g_rpe_bias.data_ptr<at::Half>()),
        B, H, W, C,
        c10::cuda::getCurrentCUDAStream());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_params", &set_params, "Set model parameters");
    m.def("run", &run, "Run swin block");
}