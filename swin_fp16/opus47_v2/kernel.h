#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

void launch_swin_block(
    const __half* x,
    __half* out,
    const __half* ln1_w, const __half* ln1_b,
    const __half* qkv_w, const __half* qkv_b,
    const __half* proj_w, const __half* proj_b,
    const __half* ln2_w, const __half* ln2_b,
    const __half* fc1_w, const __half* fc1_b,
    const __half* fc2_w, const __half* fc2_b,
    const __half* rpe_bias,
    int B, int H, int W, int C,
    cudaStream_t stream);

#endif