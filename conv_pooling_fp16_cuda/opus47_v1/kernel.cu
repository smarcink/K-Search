#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__constant__ float c_weight[OUT_C * IN_C * 3 * 3];
__constant__ float c_bias[OUT_C];

void set_weights(const void* w_half, const void* b_half) {
    const __half* wh = reinterpret_cast<const __half*>(w_half);
    const __half* bh = reinterpret_cast<const __half*>(b_half);
    float wf[OUT_C * IN_C * 3 * 3];
    float bf[OUT_C];
    for (int i = 0; i < OUT_C * IN_C * 3 * 3; ++i) wf[i] = __half2float(wh[i]);
    for (int i = 0; i < OUT_C; ++i) bf[i] = __half2float(bh[i]);
    cudaMemcpyToSymbol(c_weight, wf, sizeof(wf));
    cudaMemcpyToSymbol(c_bias, bf, sizeof(bf));
}

#define THREADS_PER_BLOCK (TILE_PH * TILE_PW)  // 128

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4)
fused_conv_relu_pool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output)
{
    const int pty = blockIdx.y * TILE_PH;
    const int ptx = blockIdx.x * TILE_PW;
    const int cty = pty * 2;
    const int ctx = ptx * 2;
    const int ity = cty - 1;
    const int itx = ctx - 1;

    const int tid = threadIdx.y * TILE_PW + threadIdx.x;
    const int py = threadIdx.y;
    const int px = threadIdx.x;

    __shared__ __half smem[2][SMEM_H * SMEM_W];

    float conv_acc[OUT_C][4];
    #pragma unroll
    for (int oc = 0; oc < OUT_C; ++oc) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) conv_acc[oc][k] = 0.0f;
    }

    const int smem_total = SMEM_H * SMEM_W;

    const bool interior = (ity >= 0) && (itx >= 0) &&
                          (ity + SMEM_H <= IN_H) && (itx + SMEM_W <= IN_W);

    auto load_channel = [&](int ic, int buf) {
        if (interior) {
            const __half* base = input + ic * IN_H * IN_W + ity * IN_W + itx;
            #pragma unroll
            for (int idx = tid; idx < smem_total; idx += THREADS_PER_BLOCK) {
                int sy = idx / SMEM_W;
                int sx = idx - sy * SMEM_W;
                smem[buf][idx] = __ldg(&base[sy * IN_W + sx]);
            }
        } else {
            #pragma unroll
            for (int idx = tid; idx < smem_total; idx += THREADS_PER_BLOCK) {
                int sy = idx / SMEM_W;
                int sx = idx - sy * SMEM_W;
                int gy = ity + sy;
                int gx = itx + sx;
                __half v;
                if (gy >= 0 && gy < IN_H && gx >= 0 && gx < IN_W) {
                    v = __ldg(&input[(ic * IN_H + gy) * IN_W + gx]);
                } else {
                    v = __float2half(0.0f);
                }
                smem[buf][idx] = v;
            }
        }
    };

    load_channel(0, 0);
    __syncthreads();

    const int cy0 = py * 2;
    const int cx0 = px * 2;

    for (int ic = 0; ic < IN_C; ++ic) {
        int cur = ic & 1;
        int nxt = 1 - cur;

        if (ic + 1 < IN_C) {
            load_channel(ic + 1, nxt);
        }

        float patch[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                patch[i][j] = __half2float(smem[cur][(cy0 + i) * SMEM_W + (cx0 + j)]);
            }
        }

        #pragma unroll
        for (int oc = 0; oc < OUT_C; ++oc) {
            const float* wp = &c_weight[((oc * IN_C) + ic) * 9];
            float w00 = wp[0], w01 = wp[1], w02 = wp[2];
            float w10 = wp[3], w11 = wp[4], w12 = wp[5];
            float w20 = wp[6], w21 = wp[7], w22 = wp[8];

            float s00 = patch[0][0]*w00 + patch[0][1]*w01 + patch[0][2]*w02
                      + patch[1][0]*w10 + patch[1][1]*w11 + patch[1][2]*w12
                      + patch[2][0]*w20 + patch[2][1]*w21 + patch[2][2]*w22;
            float s01 = patch[0][1]*w00 + patch[0][2]*w01 + patch[0][3]*w02
                      + patch[1][1]*w10 + patch[1][2]*w11 + patch[1][3]*w12
                      + patch[2][1]*w20 + patch[2][2]*w21 + patch[2][3]*w22;
            float s10 = patch[1][0]*w00 + patch[1][1]*w01 + patch[1][2]*w02
                      + patch[2][0]*w10 + patch[2][1]*w11 + patch[2][2]*w12
                      + patch[3][0]*w20 + patch[3][1]*w21 + patch[3][2]*w22;
            float s11 = patch[1][1]*w00 + patch[1][2]*w01 + patch[1][3]*w02
                      + patch[2][1]*w10 + patch[2][2]*w11 + patch[2][3]*w12
                      + patch[3][1]*w20 + patch[3][2]*w21 + patch[3][3]*w22;

            conv_acc[oc][0] += s00;
            conv_acc[oc][1] += s01;
            conv_acc[oc][2] += s10;
            conv_acc[oc][3] += s11;
        }
        __syncthreads();
    }

    const int poy = pty + py;
    const int pox = ptx + px;
    if (poy < OUT_H && pox < OUT_W) {
        #pragma unroll
        for (int oc = 0; oc < OUT_C; ++oc) {
            float b = c_bias[oc];
            float v0 = conv_acc[oc][0] + b; if (v0 < 0.0f) v0 = 0.0f;
            float v1 = conv_acc[oc][1] + b; if (v1 < 0.0f) v1 = 0.0f;
            float v2 = conv_acc[oc][2] + b; if (v2 < 0.0f) v2 = 0.0f;
            float v3 = conv_acc[oc][3] + b; if (v3 < 0.0f) v3 = 0.0f;
            float sum = (v0 + v1 + v2 + v3) * 0.25f;
            output[(oc * OUT_H + poy) * OUT_W + pox] = __float2half(sum);
        }
    }
}

void launch_fused(const __half* input, __half* output, cudaStream_t stream) {
    dim3 block(TILE_PW, TILE_PH);
    dim3 grid((OUT_W + TILE_PW - 1) / TILE_PW, (OUT_H + TILE_PH - 1) / TILE_PH);
    fused_conv_relu_pool_kernel<<<grid, block, 0, stream>>>(input, output);
}