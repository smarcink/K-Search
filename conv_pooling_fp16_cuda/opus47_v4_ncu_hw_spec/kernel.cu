#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__constant__ __half c_weight[32 * 16 * 3 * 3];
__constant__ __half c_bias[32];

void set_conv_params(const void* weight, const void* bias) {
    cudaMemcpyToSymbol(c_weight, weight, sizeof(__half) * 32 * 16 * 3 * 3);
    cudaMemcpyToSymbol(c_bias, bias, sizeof(__half) * 32);
}

#define TILE_PH 8
#define TILE_PW 32
#define CONV_H (2 * TILE_PH)   // 16
#define CONV_W (2 * TILE_PW)   // 64
#define HALO_H (CONV_H + 2)    // 18
#define HALO_W (CONV_W + 2)    // 66
#define HALO_W_PAD 68
#define C_IN 16
#define K_OUT 32

#define THREADS 256

extern "C" __global__ __launch_bounds__(THREADS, 4)
void fused_conv_relu_pool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int H, int W, int PH, int PW)
{
    int tile_pw_idx = blockIdx.x;
    int tile_ph_idx = blockIdx.y;

    int pool_row_base = tile_ph_idx * TILE_PH;
    int pool_col_base = tile_pw_idx * TILE_PW;

    int conv_row_base = pool_row_base * 2;
    int conv_col_base = pool_col_base * 2;

    int in_row_base = conv_row_base - 1;
    int in_col_base = conv_col_base - 1;

    __shared__ __half smem_in[C_IN][HALO_H][HALO_W_PAD];

    int tid = threadIdx.x;

    bool interior = (in_row_base >= 0) && (in_col_base >= 0) &&
                    (in_row_base + HALO_H <= H) && (in_col_base + HALO_W <= W);

    const int total_elems = C_IN * HALO_H * HALO_W;

    if (interior && ((in_col_base & 1) == 0)) {
        const int H8_PER_ROW = 8;
        const int total_h8 = C_IN * HALO_H * H8_PER_ROW;
        #pragma unroll 1
        for (int idx = tid; idx < total_h8; idx += THREADS) {
            int c = idx / (HALO_H * H8_PER_ROW);
            int rem = idx - c * (HALO_H * H8_PER_ROW);
            int rr = rem / H8_PER_ROW;
            int cc8 = rem - rr * H8_PER_ROW;
            int in_r = in_row_base + rr;
            int in_c = in_col_base + cc8 * 8;
            float4 v = *reinterpret_cast<const float4*>(&input[(c * H + in_r) * W + in_c]);
            *reinterpret_cast<float4*>(&smem_in[c][rr][cc8 * 8]) = v;
        }
        const int total_tail = C_IN * HALO_H;
        #pragma unroll 1
        for (int idx = tid; idx < total_tail; idx += THREADS) {
            int c = idx / HALO_H;
            int rr = idx - c * HALO_H;
            int in_r = in_row_base + rr;
            int in_c = in_col_base + 64;
            __half2 v = *reinterpret_cast<const __half2*>(&input[(c * H + in_r) * W + in_c]);
            *reinterpret_cast<__half2*>(&smem_in[c][rr][64]) = v;
        }
    } else if (interior) {
        #pragma unroll 1
        for (int idx = tid; idx < total_elems; idx += THREADS) {
            int c = idx / (HALO_H * HALO_W);
            int rem = idx - c * (HALO_H * HALO_W);
            int rr = rem / HALO_W;
            int cc = rem - rr * HALO_W;
            int in_r = in_row_base + rr;
            int in_c = in_col_base + cc;
            smem_in[c][rr][cc] = input[(c * H + in_r) * W + in_c];
        }
    } else {
        #pragma unroll 1
        for (int idx = tid; idx < total_elems; idx += THREADS) {
            int c = idx / (HALO_H * HALO_W);
            int rem = idx - c * (HALO_H * HALO_W);
            int rr = rem / HALO_W;
            int cc = rem - rr * HALO_W;
            int in_r = in_row_base + rr;
            int in_c = in_col_base + cc;
            __half v;
            if (((unsigned)in_r < (unsigned)H) && ((unsigned)in_c < (unsigned)W)) {
                v = input[(c * H + in_r) * W + in_c];
            } else {
                v = __float2half(0.0f);
            }
            smem_in[c][rr][cc] = v;
        }
    }

    __syncthreads();

    int local_pr = tid / TILE_PW;
    int local_pc = tid - local_pr * TILE_PW;

    int pool_r = pool_row_base + local_pr;
    int pool_c = pool_col_base + local_pc;

    bool in_bounds = (pool_r < PH) && (pool_c < PW);

    int lr0 = 2 * local_pr;
    int lc0 = 2 * local_pc;

    #pragma unroll 1
    for (int k = 0; k < K_OUT; k += 2) {
        __half2 b2 = __halves2half2(c_bias[k], c_bias[k + 1]);
        __half2 a00 = b2, a01 = b2, a10 = b2, a11 = b2;

        #pragma unroll
        for (int c = 0; c < C_IN; ++c) {
            const __half* base = &smem_in[c][lr0][lc0];

            __half2 r0a = *reinterpret_cast<const __half2*>(base + 0 * HALO_W_PAD + 0);
            __half2 r0b = *reinterpret_cast<const __half2*>(base + 0 * HALO_W_PAD + 2);
            __half2 r1a = *reinterpret_cast<const __half2*>(base + 1 * HALO_W_PAD + 0);
            __half2 r1b = *reinterpret_cast<const __half2*>(base + 1 * HALO_W_PAD + 2);
            __half2 r2a = *reinterpret_cast<const __half2*>(base + 2 * HALO_W_PAD + 0);
            __half2 r2b = *reinterpret_cast<const __half2*>(base + 2 * HALO_W_PAD + 2);
            __half2 r3a = *reinterpret_cast<const __half2*>(base + 3 * HALO_W_PAD + 0);
            __half2 r3b = *reinterpret_cast<const __half2*>(base + 3 * HALO_W_PAD + 2);

            __half p00h = __low2half(r0a),  p01h = __high2half(r0a);
            __half p02h = __low2half(r0b),  p03h = __high2half(r0b);
            __half p10h = __low2half(r1a),  p11h = __high2half(r1a);
            __half p12h = __low2half(r1b),  p13h = __high2half(r1b);
            __half p20h = __low2half(r2a),  p21h = __high2half(r2a);
            __half p22h = __low2half(r2b),  p23h = __high2half(r2b);
            __half p30h = __low2half(r3a),  p31h = __high2half(r3a);
            __half p32h = __low2half(r3b),  p33h = __high2half(r3b);

            __half2 p00 = __half2half2(p00h), p01 = __half2half2(p01h), p02 = __half2half2(p02h), p03 = __half2half2(p03h);
            __half2 p10 = __half2half2(p10h), p11 = __half2half2(p11h), p12 = __half2half2(p12h), p13 = __half2half2(p13h);
            __half2 p20 = __half2half2(p20h), p21 = __half2half2(p21h), p22 = __half2half2(p22h), p23 = __half2half2(p23h);
            __half2 p30 = __half2half2(p30h), p31 = __half2half2(p31h), p32 = __half2half2(p32h), p33 = __half2half2(p33h);

            const __half* wA = &c_weight[((k + 0) * C_IN + c) * 9];
            const __half* wB = &c_weight[((k + 1) * C_IN + c) * 9];
            __half2 w00 = __halves2half2(wA[0], wB[0]);
            __half2 w01 = __halves2half2(wA[1], wB[1]);
            __half2 w02 = __halves2half2(wA[2], wB[2]);
            __half2 w10 = __halves2half2(wA[3], wB[3]);
            __half2 w11 = __halves2half2(wA[4], wB[4]);
            __half2 w12 = __halves2half2(wA[5], wB[5]);
            __half2 w20 = __halves2half2(wA[6], wB[6]);
            __half2 w21 = __halves2half2(wA[7], wB[7]);
            __half2 w22 = __halves2half2(wA[8], wB[8]);

            a00 = __hfma2(p00, w00, a00);
            a00 = __hfma2(p01, w01, a00);
            a00 = __hfma2(p02, w02, a00);
            a00 = __hfma2(p10, w10, a00);
            a00 = __hfma2(p11, w11, a00);
            a00 = __hfma2(p12, w12, a00);
            a00 = __hfma2(p20, w20, a00);
            a00 = __hfma2(p21, w21, a00);
            a00 = __hfma2(p22, w22, a00);

            a01 = __hfma2(p01, w00, a01);
            a01 = __hfma2(p02, w01, a01);
            a01 = __hfma2(p03, w02, a01);
            a01 = __hfma2(p11, w10, a01);
            a01 = __hfma2(p12, w11, a01);
            a01 = __hfma2(p13, w12, a01);
            a01 = __hfma2(p21, w20, a01);
            a01 = __hfma2(p22, w21, a01);
            a01 = __hfma2(p23, w22, a01);

            a10 = __hfma2(p10, w00, a10);
            a10 = __hfma2(p11, w01, a10);
            a10 = __hfma2(p12, w02, a10);
            a10 = __hfma2(p20, w10, a10);
            a10 = __hfma2(p21, w11, a10);
            a10 = __hfma2(p22, w12, a10);
            a10 = __hfma2(p30, w20, a10);
            a10 = __hfma2(p31, w21, a10);
            a10 = __hfma2(p32, w22, a10);

            a11 = __hfma2(p11, w00, a11);
            a11 = __hfma2(p12, w01, a11);
            a11 = __hfma2(p13, w02, a11);
            a11 = __hfma2(p21, w10, a11);
            a11 = __hfma2(p22, w11, a11);
            a11 = __hfma2(p23, w12, a11);
            a11 = __hfma2(p31, w20, a11);
            a11 = __hfma2(p32, w21, a11);
            a11 = __hfma2(p33, w22, a11);
        }

        const __half2 zero2 = __float2half2_rn(0.0f);
        __half2 v00 = __hmax2(a00, zero2);
        __half2 v01 = __hmax2(a01, zero2);
        __half2 v10 = __hmax2(a10, zero2);
        __half2 v11 = __hmax2(a11, zero2);
        __half2 sum = __hadd2(__hadd2(v00, v01), __hadd2(v10, v11));
        __half2 quarter = __float2half2_rn(0.25f);
        __half2 pooled = __hmul2(sum, quarter);

        if (in_bounds) {
            output[((k + 0) * PH + pool_r) * PW + pool_c] = __low2half(pooled);
            output[((k + 1) * PH + pool_r) * PW + pool_c] = __high2half(pooled);
        }
    }
}

void launch_fused_conv_relu_pool(
    const __half* input, __half* output,
    int N, int C, int H, int W, int K,
    cudaStream_t stream)
{
    int PH = H / 2;
    int PW = W / 2;

    dim3 block(THREADS);
    dim3 grid((PW + TILE_PW - 1) / TILE_PW, (PH + TILE_PH - 1) / TILE_PH, 1);

    fused_conv_relu_pool_kernel<<<grid, block, 0, stream>>>(input, output, H, W, PH, PW);
}