#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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

static torch::Tensor to_half_cuda(torch::Tensor t) {
    if (!t.is_cuda()) t = t.cuda();
    if (t.scalar_type() != torch::kHalf) t = t.to(torch::kHalf);
    return t.contiguous();
}

void set_params(
    torch::Tensor rpe_bias,
    torch::Tensor norm1_w, torch::Tensor norm1_b,
    torch::Tensor qkv_w, torch::Tensor qkv_b,
    torch::Tensor proj_w, torch::Tensor proj_b,
    torch::Tensor norm2_w, torch::Tensor norm2_b,
    torch::Tensor fc1_w, torch::Tensor fc1_b,
    torch::Tensor fc2_w, torch::Tensor fc2_b)
{
    g_rpe_bias = to_half_cuda(rpe_bias);
    g_norm1_w = to_half_cuda(norm1_w);
    g_norm1_b = to_half_cuda(norm1_b);
    g_qkv_w = to_half_cuda(qkv_w);
    g_qkv_b = to_half_cuda(qkv_b);
    g_proj_w = to_half_cuda(proj_w);
    g_proj_b = to_half_cuda(proj_b);
    g_norm2_w = to_half_cuda(norm2_w);
    g_norm2_b = to_half_cuda(norm2_b);
    g_fc1_w = to_half_cuda(fc1_w);
    g_fc1_b = to_half_cuda(fc1_b);
    g_fc2_w = to_half_cuda(fc2_w);
    g_fc2_b = to_half_cuda(fc2_b);
}

std::vector<torch::Tensor> run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    if (x.scalar_type() != torch::kHalf) x = x.to(torch::kHalf);
    x = x.contiguous();

    int B = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    int C = x.size(3);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_swin_block(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
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
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        B, H, W, C, stream);

    return {out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Run fused Swin block");
    m.def("set_params", &set_params, "Set model parameters");
}