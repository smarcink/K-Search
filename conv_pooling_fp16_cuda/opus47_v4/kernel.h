#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CIN 16
#define COUT 32
#define H_IN 720
#define W_IN 1280
#define H_OUT 360
#define W_OUT 640

#define TILE_OUT_H 8
#define TILE_OUT_W 16
#define TILE_CONV_H (TILE_OUT_H * 2)
#define TILE_CONV_W (TILE_OUT_W * 2)
#define TILE_IN_H (TILE_CONV_H + 2)
#define TILE_IN_W (TILE_CONV_W + 2)

#define COUT_GROUP 4
#define COUT_PER_THREAD (COUT / COUT_GROUP)
#define COUT_PER_THREAD_H2 (COUT_PER_THREAD / 2)

void set_conv_params(const __half* weight, const __half* bias);
void launch_fused_conv_relu_pool(const __half* input, __half* output, cudaStream_t stream);

#endif