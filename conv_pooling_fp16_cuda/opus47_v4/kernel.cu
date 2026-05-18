#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Weight layout: [CIN][9][COUT]
__constant__ __half c_weight[CIN * 9 * COUT];
__constant__ __half c_bias[COUT];

void set_conv_params(const __half* weight, const __half* bias) {
    __half h_w[CIN * 9 * COUT];
    __half tmp[COUT * CIN * 9];
    cudaMemcpy(tmp, weight, sizeof(__half) * COUT * CIN * 9, cudaMemcpyDeviceToHost);
    for (int oc = 0; oc < COUT; ++oc) {
        for (int c = 0; c < CIN; ++c) {
            for (int k = 0; k < 9; ++k) {
                h_w[(c * 9 + k) * COUT + oc] = tmp[(oc * CIN + c) * 9 + k];
            }
        }
    }
    cudaMemcpyToSymbol(c_weight, h_w, sizeof(__half) * CIN * 9 * COUT);
    cudaMemcpyToSymbol(c_bias, bias, sizeof(__half) * COUT);
}

__global__ __launch_bounds__(512, 2)
void fused_conv_relu_pool_kernel(const __half* __restrict__ input,
                                  __half* __restrict__ output) {
    __shared__ __half smem_in[CIN][TILE_IN_H][TILE_IN_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = (tz * TILE_OUT_H + ty) * TILE_OUT_W + tx;
    const int nthreads = TILE_OUT_W * TILE_OUT_H * COUT_GROUP;  // 512

    const int out_x0 = blockIdx.x * TILE_OUT_W;
    const int out_y0 = blockIdx.y * TILE_OUT_H;
    const int in_x0 = out_x0 * 2 - 1;
    const int in_y0 = out_y0 * 2 - 1;

    const int tile_pixels = TILE_IN_H * TILE_IN_W;  // 34*18 = 612

    bool tile_in_bounds = (in_x0 >= 0) && (in_y0 >= 0) &&
                          (in_x0 + TILE_IN_W <= W_IN) && (in_y0 + TILE_IN_H <= H_IN);

    if (tile_in_bounds) {
        #pragma unroll 1
        for (int idx = tid; idx < tile_pixels; idx += nthreads) {
            int ly = idx / TILE_IN_W;
            int lx = idx - ly * TILE_IN_W;
            int gy = in_y0 + ly;
            int gx = in_x0 + lx;
            #pragma unroll
            for (int c = 0; c < CIN; ++c) {
                smem_in[c][ly][lx] = input[((c) * H_IN + gy) * W_IN + gx];
            }
        }
    } else {
        #pragma unroll 1
        for (int idx = tid; idx < tile_pixels; idx += nthreads) {
            int ly = idx / TILE_IN_W;
            int lx = idx - ly * TILE_IN_W;
            int gy = in_y0 + ly;
            int gx = in_x0 + lx;
            bool ib = (gy >= 0 && gy < H_IN && gx >= 0 && gx < W_IN);
            #pragma unroll
            for (int c = 0; c < CIN; ++c) {
                __half v = __float2half(0.0f);
                if (ib) v = input[((c) * H_IN + gy) * W_IN + gx];
                smem_in[c][ly][lx] = v;
            }
        }
    }
    __syncthreads();

    const int cy0 = 1 + 2 * ty;
    const int cx0 = 1 + 2 * tx;

    const int gx_out = out_x0 + tx;
    const int gy_out = out_y0 + ty;
    bool valid = (gx_out < W_OUT && gy_out < H_OUT);

    __half2 acc00[COUT_PER_THREAD_H2];
    __half2 acc01[COUT_PER_THREAD_H2];
    __half2 acc10[COUT_PER_THREAD_H2];
    __half2 acc11[COUT_PER_THREAD_H2];

    const __half2 zero2 = __float2half2_rn(0.0f);
    #pragma unroll
    for (int o = 0; o < COUT_PER_THREAD_H2; ++o) {
        acc00[o] = zero2; acc01[o] = zero2; acc10[o] = zero2; acc11[o] = zero2;
    }

    const int oc_h2_base = tz * COUT_PER_THREAD_H2;

    #pragma unroll
    for (int c = 0; c < CIN; ++c) {
        __half patch[4][4];
        #pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 4; ++dx) {
                patch[dy][dx] = smem_in[c][cy0 - 1 + dy][cx0 - 1 + dx];
            }
        }

        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                int k = ky * 3 + kx;
                const __half2* w_ptr = reinterpret_cast<const __half2*>(
                    &c_weight[(c * 9 + k) * COUT]) + oc_h2_base;

                __half2 p00 = __half2half2(patch[ky + 0][kx + 0]);
                __half2 p01 = __half2half2(patch[ky + 0][kx + 1]);
                __half2 p10 = __half2half2(patch[ky + 1][kx + 0]);
                __half2 p11 = __half2half2(patch[ky + 1][kx + 1]);

                #pragma unroll
                for (int o = 0; o < COUT_PER_THREAD_H2; ++o) {
                    __half2 w = w_ptr[o];
                    acc00[o] = __hfma2(w, p00, acc00[o]);
                    acc01[o] = __hfma2(w, p01, acc01[o]);
                    acc10[o] = __hfma2(w, p10, acc10[o]);
                    acc11[o] = __hfma2(w, p11, acc11[o]);
                }
            }
        }
    }

    if (!valid) return;

    const __half2 quarter = __float2half2_rn(0.25f);
    const __half2 zero_h2 = __float2half2_rn(0.0f);

    #pragma unroll
    for (int o = 0; o < COUT_PER_THREAD_H2; ++o) {
        int oc_h2 = oc_h2_base + o;
        __half2 b = *reinterpret_cast<const __half2*>(&c_bias[oc_h2 * 2]);
        __half2 v00 = __hmax2(__hadd2(acc00[o], b), zero_h2);
        __half2 v01 = __hmax2(__hadd2(acc01[o], b), zero_h2);
        __half2 v10 = __hmax2(__hadd2(acc10[o], b), zero_h2);
        __half2 v11 = __hmax2(__hadd2(acc11[o], b), zero_h2);
        __half2 sum = __hadd2(__hadd2(v00, v01), __hadd2(v10, v11));
        __half2 pooled = __hmul2(sum, quarter);

        int oc0 = oc_h2 * 2;
        output[((oc0)     * H_OUT + gy_out) * W_OUT + gx_out] = __low2half(pooled);
        output[((oc0 + 1) * H_OUT + gy_out) * W_OUT + gx_out] = __high2half(pooled);
    }
}

void launch_fused_conv_relu_pool(const __half* input, __half* output, cudaStream_t stream) {
    dim3 block(TILE_OUT_W, TILE_OUT_H, COUT_GROUP);
    dim3 grid((W_OUT + TILE_OUT_W - 1) / TILE_OUT_W,
              (H_OUT + TILE_OUT_H - 1) / TILE_OUT_H);
    fused_conv_relu_pool_kernel<<<grid, block, 0, stream>>>(input, output);
}