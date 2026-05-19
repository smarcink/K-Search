#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void set_conv_params(const void* weight, const void* bias);
void launch_fused_conv_relu_pool(
    const __half* input, __half* output,
    int N, int C, int H, int W, int K,
    cudaStream_t stream);