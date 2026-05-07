#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_avgpool2x2_nhwc(const __half* in, __half* out,
                             int N, int H, int W, int C,
                             cudaStream_t stream);

void launch_relu_avgpool2x2_nhwc(const __half* in, __half* out,
                                  int N, int H, int W, int C,
                                  cudaStream_t stream);

void launch_relu_avgpool2x2_nhwc_to_nchw(const __half* in, __half* out,
                                          int N, int H, int W, int C,
                                          cudaStream_t stream);

void launch_relu_avgpool2x2(const __half* in, __half* out,
                             int N, int C, int H, int W,
                             cudaStream_t stream);

#endif