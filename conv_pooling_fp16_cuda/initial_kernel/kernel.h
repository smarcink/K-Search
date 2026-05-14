#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused Conv2d 3x3 + ReLU, stride=1, pad=1 on NCHW fp16.
// Input:   (N, Cin, H, W)
// Weight:  (Cout, Cin, 3, 3)
// Bias:    (Cout,) or nullptr
// Output:  (N, Cout, H, W)  with ReLU applied
void launch_conv2d_3x3_relu(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int N, int Cin, int Cout, int H, int W,
    cudaStream_t stream);

// AvgPool2x2 on NCHW fp16 tensor.
// Input:  (N, C, H, W)  — H and W must be even.
// Output: (N, C, H/2, W/2)
void launch_avgpool2x2(
    const __half* in, __half* out,
    int N, int C, int H, int W,
    cudaStream_t stream);

#endif
