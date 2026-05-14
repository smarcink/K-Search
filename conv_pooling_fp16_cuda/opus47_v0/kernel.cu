#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__constant__ float c_weight[32 * 16 * 3 * 3];
__constant__ float c_bias[32];

void set_conv_params(const void* weight_ptr, const void* bias_ptr,
                     int out_channels, int in_channels) {
    int w_count = out_channels * in_channels * 3 * 3;
    int b_count = out_channels;

    __half* h_w = new __half[w_count];
    __half* h_b = new __half[b_count];
    cudaMemcpy(h_w, weight_ptr, w_count * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, bias_ptr, b_count * sizeof(__half), cudaMemcpyDeviceToHost);

    float* f_w = new float[w_count];
    float* f_b = new float[b_count];
    for (int i = 0; i < w_count; ++i) f_w[i] = __half2float(h_w[i]);
    for (int i = 0; i < b_count; ++i) f_b[i] = __half2float(h_b[i]);

    cudaMemcpyToSymbol(c_weight, f_w, w_count * sizeof(float));
    cudaMemcpyToSymbol(c_bias, f_b, b_count * sizeof(float));

    delete[] h_w; delete[] h_b; delete[] f_w; delete[] f_b;
}

#define TILE_OH 8
#define TILE_OW 16
#define CONV_H (2*TILE_OH)
#define CONV_W (2*TILE_OW)
#define SMEM_H (CONV_H + 2)
#define SMEM_W (CONV_W + 2)

#define C_IN 16
#define C_OUT 32

// Load all 16 input channels into shared memory at once, then iterate OC chunks.
// Shared memory: 16 * 18 * 34 * 4B = ~39 KB - fits per block on RTX5090.
__global__ void __launch_bounds__(128, 8) fused_conv_relu_pool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int H, int W, int OH, int OW)
{
    int tile_oh_start = blockIdx.y * TILE_OH;
    int tile_ow_start = blockIdx.x * TILE_OW;

    int in_row_start = tile_oh_start * 2 - 1;
    int in_col_start = tile_ow_start * 2 - 1;

    __shared__ float s_in[C_IN][SMEM_H][SMEM_W];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    int local_oh = threadIdx.y;
    int local_ow = threadIdx.x;
    int oh = tile_oh_start + local_oh;
    int ow = tile_ow_start + local_ow;
    bool valid = (oh < OH) && (ow < OW);

    // Load all input channels into shared memory
    int total = C_IN * SMEM_H * SMEM_W;
    for (int idx = tid; idx < total; idx += nthreads) {
        int ic = idx / (SMEM_H * SMEM_W);
        int rem = idx - ic * (SMEM_H * SMEM_W);
        int r = rem / SMEM_W;
        int c = rem - r * SMEM_W;
        int gr = in_row_start + r;
        int gc = in_col_start + c;
        float v = 0.f;
        if (gr >= 0 && gr < H && gc >= 0 && gc < W) {
            v = __half2float(input[((ic) * H + gr) * W + gc]);
        }
        s_in[ic][r][c] = v;
    }
    __syncthreads();

    int sr0 = local_oh * 2;
    int sc0 = local_ow * 2;

    constexpr int OC_CHUNK = 8;
    float acc[OC_CHUNK][4];

    for (int oc_base = 0; oc_base < C_OUT; oc_base += OC_CHUNK) {
        #pragma unroll
        for (int k = 0; k < OC_CHUNK; ++k) {
            float b = c_bias[oc_base + k];
            acc[k][0] = b; acc[k][1] = b; acc[k][2] = b; acc[k][3] = b;
        }

        #pragma unroll
        for (int ic = 0; ic < C_IN; ++ic) {
            float p[4][4];
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    p[rr][cc] = s_in[ic][sr0 + rr][sc0 + cc];
                }
            }

            #pragma unroll
            for (int k = 0; k < OC_CHUNK; ++k) {
                int oc = oc_base + k;
                const float* w = &c_weight[((oc * C_IN) + ic) * 9];
                float w00=w[0], w01=w[1], w02=w[2];
                float w10=w[3], w11=w[4], w12=w[5];
                float w20=w[6], w21=w[7], w22=w[8];

                acc[k][0] += p[0][0]*w00 + p[0][1]*w01 + p[0][2]*w02
                           + p[1][0]*w10 + p[1][1]*w11 + p[1][2]*w12
                           + p[2][0]*w20 + p[2][1]*w21 + p[2][2]*w22;
                acc[k][1] += p[0][1]*w00 + p[0][2]*w01 + p[0][3]*w02
                           + p[1][1]*w10 + p[1][2]*w11 + p[1][3]*w12
                           + p[2][1]*w20 + p[2][2]*w21 + p[2][3]*w22;
                acc[k][2] += p[1][0]*w00 + p[1][1]*w01 + p[1][2]*w02
                           + p[2][0]*w10 + p[2][1]*w11 + p[2][2]*w12
                           + p[3][0]*w20 + p[3][1]*w21 + p[3][2]*w22;
                acc[k][3] += p[1][1]*w00 + p[1][2]*w01 + p[1][3]*w02
                           + p[2][1]*w10 + p[2][2]*w11 + p[2][3]*w12
                           + p[3][1]*w20 + p[3][2]*w21 + p[3][3]*w22;
            }
        }

        if (valid) {
            #pragma unroll
            for (int k = 0; k < OC_CHUNK; ++k) {
                float v0 = fmaxf(acc[k][0], 0.f);
                float v1 = fmaxf(acc[k][1], 0.f);
                float v2 = fmaxf(acc[k][2], 0.f);
                float v3 = fmaxf(acc[k][3], 0.f);
                float avg = 0.25f * (v0 + v1 + v2 + v3);
                int oc = oc_base + k;
                int out_idx = (oc * OH + oh) * OW + ow;
                output[out_idx] = __float2half(avg);
            }
        }
    }
}

// Float32 variant
__global__ void fused_conv_relu_pool_kernel_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W, int OH, int OW)
{
    int tile_oh_start = blockIdx.y * TILE_OH;
    int tile_ow_start = blockIdx.x * TILE_OW;

    int in_row_start = tile_oh_start * 2 - 1;
    int in_col_start = tile_ow_start * 2 - 1;

    __shared__ float s_in[SMEM_H * SMEM_W];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    int local_oh = threadIdx.y;
    int local_ow = threadIdx.x;
    int oh = tile_oh_start + local_oh;
    int ow = tile_ow_start + local_ow;
    bool valid = (oh < OH) && (ow < OW);

    constexpr int OC_CHUNK = 8;
    float acc[OC_CHUNK][4];

    int N_n = 0;

    for (int oc_base = 0; oc_base < C_OUT; oc_base += OC_CHUNK) {
        #pragma unroll
        for (int k = 0; k < OC_CHUNK; ++k) {
            float b = c_bias[oc_base + k];
            acc[k][0] = b; acc[k][1] = b; acc[k][2] = b; acc[k][3] = b;
        }

        for (int ic = 0; ic < C_IN; ++ic) {
            const float* in_ch = input + ((N_n * C_IN + ic) * H) * W;

            for (int idx = tid; idx < SMEM_H * SMEM_W; idx += nthreads) {
                int r = idx / SMEM_W;
                int c = idx % SMEM_W;
                int gr = in_row_start + r;
                int gc = in_col_start + c;
                float v = 0.f;
                if (gr >= 0 && gr < H && gc >= 0 && gc < W) {
                    v = in_ch[gr * W + gc];
                }
                s_in[idx] = v;
            }
            __syncthreads();

            int sr0 = local_oh * 2;
            int sc0 = local_ow * 2;

            float p[4][4];
            #pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    p[rr][cc] = s_in[(sr0 + rr) * SMEM_W + (sc0 + cc)];
                }
            }

            #pragma unroll
            for (int k = 0; k < OC_CHUNK; ++k) {
                int oc = oc_base + k;
                const float* w = &c_weight[((oc * C_IN) + ic) * 9];
                float w00=w[0], w01=w[1], w02=w[2];
                float w10=w[3], w11=w[4], w12=w[5];
                float w20=w[6], w21=w[7], w22=w[8];

                acc[k][0] += p[0][0]*w00 + p[0][1]*w01 + p[0][2]*w02
                           + p[1][0]*w10 + p[1][1]*w11 + p[1][2]*w12
                           + p[2][0]*w20 + p[2][1]*w21 + p[2][2]*w22;
                acc[k][1] += p[0][1]*w00 + p[0][2]*w01 + p[0][3]*w02
                           + p[1][1]*w10 + p[1][2]*w11 + p[1][3]*w12
                           + p[2][1]*w20 + p[2][2]*w21 + p[2][3]*w22;
                acc[k][2] += p[1][0]*w00 + p[1][1]*w01 + p[1][2]*w02
                           + p[2][0]*w10 + p[2][1]*w11 + p[2][2]*w12
                           + p[3][0]*w20 + p[3][1]*w21 + p[3][2]*w22;
                acc[k][3] += p[1][1]*w00 + p[1][2]*w01 + p[1][3]*w02
                           + p[2][1]*w10 + p[2][2]*w11 + p[2][3]*w12
                           + p[3][1]*w20 + p[3][2]*w21 + p[3][3]*w22;
            }

            __syncthreads();
        }

        if (valid) {
            #pragma unroll
            for (int k = 0; k < OC_CHUNK; ++k) {
                float v0 = fmaxf(acc[k][0], 0.f);
                float v1 = fmaxf(acc[k][1], 0.f);
                float v2 = fmaxf(acc[k][2], 0.f);
                float v3 = fmaxf(acc[k][3], 0.f);
                float avg = 0.25f * (v0 + v1 + v2 + v3);
                int oc = oc_base + k;
                int out_idx = ((N_n * C_OUT + oc) * OH + oh) * OW + ow;
                output[out_idx] = avg;
            }
        }
    }
}

void launch_fused_conv_relu_pool(
    const __half* input,
    __half* output,
    int N, int C_in, int H, int W,
    int C_out,
    cudaStream_t stream)
{
    int OH = H / 2;
    int OW = W / 2;

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid((OW + TILE_OW - 1) / TILE_OW,
              (OH + TILE_OH - 1) / TILE_OH,
              1);

    fused_conv_relu_pool_kernel<<<grid, block, 0, stream>>>(
        input, output, H, W, OH, OW);
}

void launch_fused_conv_relu_pool_f32(
    const float* input,
    float* output,
    int N, int C_in, int H, int W,
    int C_out,
    cudaStream_t stream)
{
    int OH = H / 2;
    int OW = W / 2;

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid((OW + TILE_OW - 1) / TILE_OW,
              (OH + TILE_OH - 1) / TILE_OH,
              1);

    fused_conv_relu_pool_kernel_f32<<<grid, block, 0, stream>>>(
        input, output, H, W, OH, OW);
}