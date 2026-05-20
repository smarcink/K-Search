#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define IN_C 16
#define OUT_C 32
#define IN_H 720
#define IN_W 1280
#define OUT_H 360
#define OUT_W 640

// Pooled tile dims
#define TILE_PH 8
#define TILE_PW 16
// Conv-output tile dims (2x pooled)
#define TILE_CH 16
#define TILE_CW 32
// Shared input tile (padded 1 on each side)
#define SMEM_H (TILE_CH + 2)
#define SMEM_W (TILE_CW + 2)

void set_weights(const void* w_half, const void* b_half);
void launch_fused(const __half* input, __half* output, cudaStream_t stream);

#endif