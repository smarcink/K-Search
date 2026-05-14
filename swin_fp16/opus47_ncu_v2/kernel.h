#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_swin_block(
    const __half* x,
    const __half* norm1_w, const __half* norm1_b,
    const __half* qkv_w, const __half* qkv_b,
    const __half* proj_w, const __half* proj_b,
    const __half* norm2_w, const __half* norm2_b,
    const __half* fc1_w, const __half* fc1_b,
    const __half* fc2_w, const __half* fc2_b,
    const __half* rpe_bias,
    __half* out,
    int B, int H, int W, int C,
    cudaStream_t stream);

#endif