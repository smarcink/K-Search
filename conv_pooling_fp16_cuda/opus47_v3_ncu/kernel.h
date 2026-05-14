#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_fused_conv_relu_pool(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int N, int Cin, int Cout, int H, int W,
    cudaStream_t stream);

#endif