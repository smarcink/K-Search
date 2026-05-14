#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void set_conv_params(const void* weight_ptr, const void* bias_ptr,
                     int out_channels, int in_channels);

void launch_fused_conv_relu_pool(
    const __half* input,
    __half* output,
    int N, int C_in, int H, int W,
    int C_out,
    cudaStream_t stream);

#endif