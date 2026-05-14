#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Tile dimensions in OUTPUT (post-pool) space.
#define TILE_OH 8
#define TILE_OW 16
#define CONV_TILE_H (TILE_OH * 2)   // 16
#define CONV_TILE_W (TILE_OW * 2)   // 32
#define IN_TILE_H (CONV_TILE_H + 2) // 18
#define IN_TILE_W (CONV_TILE_W + 2) // 34

#define CO_PER_BLOCK 16

__constant__ __half c_weight[32 * 16 * 3 * 3];
__constant__ __half c_bias[32];

__global__ __launch_bounds__(128, 6)
void fused_conv_relu_pool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int Cin, int Cout, int H, int W,
    int has_bias)
{
    int oH = H / 2;
    int oW = W / 2;

    int tile_ox = blockIdx.x * TILE_OW;
    int tile_oy = blockIdx.y * TILE_OH;
    int nco_block = blockIdx.z;
    int co_groups = Cout / CO_PER_BLOCK;
    int n  = nco_block / co_groups;
    int co_base = (nco_block % co_groups) * CO_PER_BLOCK;

    int conv_x0 = tile_ox * 2;
    int conv_y0 = tile_oy * 2;
    int in_x0 = conv_x0 - 1;
    int in_y0 = conv_y0 - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    // Shared memory holds two channels interleaved as __half2 per spatial location.
    __shared__ __align__(16) __half2 s_in2[IN_TILE_H * IN_TILE_W];

    float acc[CO_PER_BLOCK][4];

    #pragma unroll
    for (int k = 0; k < CO_PER_BLOCK; ++k) {
        float b = has_bias ? __half2float(c_bias[co_base + k]) : 0.0f;
        acc[k][0] = b; acc[k][1] = b; acc[k][2] = b; acc[k][3] = b;
    }

    int in_tile_size = IN_TILE_H * IN_TILE_W;

    int my_cy = ty * 2;
    int my_cx = tx * 2;

    bool tile_fully_inside = (in_y0 >= 0) && (in_y0 + IN_TILE_H <= H) &&
                             (in_x0 >= 0) && (in_x0 + IN_TILE_W <= W);

    // Iterate over Cin in pairs of 2 channels at a time.
    for (int ci = 0; ci < Cin; ci += 2) {
        const __half* in_chan0 = input + ((n * Cin + ci    ) * H) * W;
        const __half* in_chan1 = input + ((n * Cin + ci + 1) * H) * W;

        if (tile_fully_inside) {
            #pragma unroll
            for (int i = tid; i < in_tile_size; i += nthreads) {
                int ly = i / IN_TILE_W;
                int lx = i - ly * IN_TILE_W;
                int gidx = (in_y0 + ly) * W + (in_x0 + lx);
                __half a = in_chan0[gidx];
                __half b = in_chan1[gidx];
                s_in2[i] = __halves2half2(a, b);
            }
        } else {
            for (int i = tid; i < in_tile_size; i += nthreads) {
                int ly = i / IN_TILE_W;
                int lx = i - ly * IN_TILE_W;
                int gy = in_y0 + ly;
                int gx = in_x0 + lx;
                __half a = __float2half(0.0f);
                __half b = __float2half(0.0f);
                if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
                    int gidx = gy * W + gx;
                    a = in_chan0[gidx];
                    b = in_chan1[gidx];
                }
                s_in2[i] = __halves2half2(a, b);
            }
        }
        __syncthreads();

        // Load 4x4 patch as half2 pairs (ch_even, ch_odd interleaved).
        int base = my_cy * IN_TILE_W + my_cx;
        __half2 ph[4][4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                ph[r][c] = s_in2[base + r * IN_TILE_W + c];
            }
        }

        #pragma unroll
        for (int k = 0; k < CO_PER_BLOCK; ++k) {
            int co = co_base + k;
            const __half* w0_ptr = &c_weight[(co * Cin + ci    ) * 9];
            const __half* w1_ptr = &c_weight[(co * Cin + ci + 1) * 9];

            __half2 w00 = __halves2half2(w0_ptr[0], w1_ptr[0]);
            __half2 w01 = __halves2half2(w0_ptr[1], w1_ptr[1]);
            __half2 w02 = __halves2half2(w0_ptr[2], w1_ptr[2]);
            __half2 w10 = __halves2half2(w0_ptr[3], w1_ptr[3]);
            __half2 w11 = __halves2half2(w0_ptr[4], w1_ptr[4]);
            __half2 w12 = __halves2half2(w0_ptr[5], w1_ptr[5]);
            __half2 w20 = __halves2half2(w0_ptr[6], w1_ptr[6]);
            __half2 w21 = __halves2half2(w0_ptr[7], w1_ptr[7]);
            __half2 w22 = __halves2half2(w0_ptr[8], w1_ptr[8]);

            __half2 s0 = __float2half2_rn(0.0f);
            __half2 s1 = __float2half2_rn(0.0f);
            __half2 s2 = __float2half2_rn(0.0f);
            __half2 s3 = __float2half2_rn(0.0f);

            s0 = __hfma2(ph[0][0], w00, s0);
            s0 = __hfma2(ph[0][1], w01, s0);
            s0 = __hfma2(ph[0][2], w02, s0);
            s0 = __hfma2(ph[1][0], w10, s0);
            s0 = __hfma2(ph[1][1], w11, s0);
            s0 = __hfma2(ph[1][2], w12, s0);
            s0 = __hfma2(ph[2][0], w20, s0);
            s0 = __hfma2(ph[2][1], w21, s0);
            s0 = __hfma2(ph[2][2], w22, s0);

            s1 = __hfma2(ph[0][1], w00, s1);
            s1 = __hfma2(ph[0][2], w01, s1);
            s1 = __hfma2(ph[0][3], w02, s1);
            s1 = __hfma2(ph[1][1], w10, s1);
            s1 = __hfma2(ph[1][2], w11, s1);
            s1 = __hfma2(ph[1][3], w12, s1);
            s1 = __hfma2(ph[2][1], w20, s1);
            s1 = __hfma2(ph[2][2], w21, s1);
            s1 = __hfma2(ph[2][3], w22, s1);

            s2 = __hfma2(ph[1][0], w00, s2);
            s2 = __hfma2(ph[1][1], w01, s2);
            s2 = __hfma2(ph[1][2], w02, s2);
            s2 = __hfma2(ph[2][0], w10, s2);
            s2 = __hfma2(ph[2][1], w11, s2);
            s2 = __hfma2(ph[2][2], w12, s2);
            s2 = __hfma2(ph[3][0], w20, s2);
            s2 = __hfma2(ph[3][1], w21, s2);
            s2 = __hfma2(ph[3][2], w22, s2);

            s3 = __hfma2(ph[1][1], w00, s3);
            s3 = __hfma2(ph[1][2], w01, s3);
            s3 = __hfma2(ph[1][3], w02, s3);
            s3 = __hfma2(ph[2][1], w10, s3);
            s3 = __hfma2(ph[2][2], w11, s3);
            s3 = __hfma2(ph[2][3], w12, s3);
            s3 = __hfma2(ph[3][1], w20, s3);
            s3 = __hfma2(ph[3][2], w21, s3);
            s3 = __hfma2(ph[3][3], w22, s3);

            float2 f0 = __half22float2(s0);
            float2 f1 = __half22float2(s1);
            float2 f2 = __half22float2(s2);
            float2 f3 = __half22float2(s3);
            acc[k][0] += f0.x + f0.y;
            acc[k][1] += f1.x + f1.y;
            acc[k][2] += f2.x + f2.y;
            acc[k][3] += f3.x + f3.y;
        }
        __syncthreads();
    }

    int oy = tile_oy + ty;
    int ox = tile_ox + tx;
    if (oy >= oH || ox >= oW) return;

    #pragma unroll
    for (int k = 0; k < CO_PER_BLOCK; ++k) {
        int co = co_base + k;
        float a0 = fmaxf(acc[k][0], 0.0f);
        float a1 = fmaxf(acc[k][1], 0.0f);
        float a2 = fmaxf(acc[k][2], 0.0f);
        float a3 = fmaxf(acc[k][3], 0.0f);
        float avg = (a0 + a1 + a2 + a3) * 0.25f;
        int out_idx = ((n * Cout + co) * oH + oy) * oW + ox;
        output[out_idx] = __float2half(avg);
    }
}

void launch_fused_conv_relu_pool(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int N, int Cin, int Cout, int H, int W,
    cudaStream_t stream)
{
    cudaMemcpyToSymbolAsync(c_weight, weight,
        Cout * Cin * 9 * sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    int has_bias = (bias != nullptr) ? 1 : 0;
    if (has_bias) {
        cudaMemcpyToSymbolAsync(c_bias, bias,
            Cout * sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    }

    int oH = H / 2;
    int oW = W / 2;
    int co_groups = Cout / CO_PER_BLOCK;
    dim3 grid((oW + TILE_OW - 1) / TILE_OW,
              (oH + TILE_OH - 1) / TILE_OH,
              N * co_groups);
    dim3 block(TILE_OW, TILE_OH);

    fused_conv_relu_pool_kernel<<<grid, block, 0, stream>>>(
        input, output, N, Cin, Cout, H, W, has_bias);
}