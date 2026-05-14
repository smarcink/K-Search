#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Specialized for Cin=16, Cout=32, fp16 NCHW.
// Optimized: half2-vectorized inner loop with __hfma2 to exploit FP16 SIMD.
// Two output channels are processed together via half2.

#define POOL_TILE_H 8
#define POOL_TILE_W 16
#define CONV_TILE_H (POOL_TILE_H * 2)   // 16
#define CONV_TILE_W (POOL_TILE_W * 2)   // 32
#define IN_TILE_H  (CONV_TILE_H + 2)    // 18
#define IN_TILE_W  (CONV_TILE_W + 2)    // 34

// Weight layout: rearranged so for each (co_pair, ci, k) we store
// half2(weight[co_pair*2+0][ci][k], weight[co_pair*2+1][ci][k]).
// Size: (Cout/2) * Cin * 9 half2 = 16 * 16 * 9 = 2304 half2 = 4608 halfs
__constant__ __half2 c_weight_h2[16 * 16 * 9];
__constant__ __half2 c_bias_h2[16];  // pairs of bias values

extern __shared__ __half smem[];

__global__ void fused_conv_relu_avgpool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int H, int W, int has_bias)
{
    constexpr int Cin = 16;
    constexpr int Cout = 32;
    constexpr int CoutPairs = Cout / 2;  // 16
    const int oH = H >> 1;
    const int oW = W >> 1;

    const int tile_pw0 = blockIdx.x * POOL_TILE_W;
    const int tile_ph0 = blockIdx.y * POOL_TILE_H;
    const int n = blockIdx.z;

    const int conv_h0 = tile_ph0 * 2;
    const int conv_w0 = tile_pw0 * 2;
    const int in_h0 = conv_h0 - 1;
    const int in_w0 = conv_w0 - 1;

    const int tid = threadIdx.x;
    const int BLOCK = blockDim.x;  // 256

    const int IN_TILE_SIZE = IN_TILE_H * IN_TILE_W;  // 612
    __half* s_input = smem;

    const int in_total = Cin * IN_TILE_SIZE;
    #pragma unroll 4
    for (int i = tid; i < in_total; i += BLOCK) {
        int ci = i / IN_TILE_SIZE;
        int rem = i - ci * IN_TILE_SIZE;
        int ty = rem / IN_TILE_W;
        int tx = rem - ty * IN_TILE_W;
        int ih = in_h0 + ty;
        int iw = in_w0 + tx;
        __half v;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            v = input[((n * Cin + ci) * H + ih) * W + iw];
        } else {
            v = __float2half(0.0f);
        }
        s_input[i] = v;
    }

    __syncthreads();

    // 256 threads, work is CoutPairs(16) * POOL_TILE_H(8) * POOL_TILE_W(16) = 2048
    // Each thread does 8 outputs (each output covers 2 channels).
    const int total_pair_pool = CoutPairs * POOL_TILE_H * POOL_TILE_W;  // 2048
    const int per_thread = total_pair_pool / 256;                        // 8

    #pragma unroll
    for (int k = 0; k < per_thread; ++k) {
        int idx = tid + k * BLOCK;
        // idx layout: co_pair (outer) * POOL_TILE_H * POOL_TILE_W + ph * POOL_TILE_W + pw
        int pw = idx & (POOL_TILE_W - 1);     // 0..15
        int tmp = idx >> 4;                    // /16
        int ph = tmp & (POOL_TILE_H - 1);      // 0..7
        int co_pair = tmp >> 3;                // 0..15

        int cy0 = ph * 2;
        int cx0 = pw * 2;

        __half2 a00 = __float2half2_rn(0.f);
        __half2 a01 = __float2half2_rn(0.f);
        __half2 a10 = __float2half2_rn(0.f);
        __half2 a11 = __float2half2_rn(0.f);

        const __half2* wptr = c_weight_h2 + co_pair * Cin * 9;

        #pragma unroll
        for (int ci = 0; ci < 16; ++ci) {
            const __half* in_ch = s_input + ci * IN_TILE_SIZE;
            const __half2* wch = wptr + ci * 9;

            // Load 4x4 input patch
            __half p[4][4];
            #pragma unroll
            for (int yy = 0; yy < 4; ++yy) {
                const __half* row = in_ch + (cy0 + yy) * IN_TILE_W + cx0;
                p[yy][0] = row[0];
                p[yy][1] = row[1];
                p[yy][2] = row[2];
                p[yy][3] = row[3];
            }

            __half2 w00 = wch[0], w01 = wch[1], w02 = wch[2];
            __half2 w10 = wch[3], w11 = wch[4], w12 = wch[5];
            __half2 w20 = wch[6], w21 = wch[7], w22 = wch[8];

            // a00: conv at (0,0)
            a00 = __hfma2(w00, __half2half2(p[0][0]), a00);
            a00 = __hfma2(w01, __half2half2(p[0][1]), a00);
            a00 = __hfma2(w02, __half2half2(p[0][2]), a00);
            a00 = __hfma2(w10, __half2half2(p[1][0]), a00);
            a00 = __hfma2(w11, __half2half2(p[1][1]), a00);
            a00 = __hfma2(w12, __half2half2(p[1][2]), a00);
            a00 = __hfma2(w20, __half2half2(p[2][0]), a00);
            a00 = __hfma2(w21, __half2half2(p[2][1]), a00);
            a00 = __hfma2(w22, __half2half2(p[2][2]), a00);

            // a01: conv at (0,1)
            a01 = __hfma2(w00, __half2half2(p[0][1]), a01);
            a01 = __hfma2(w01, __half2half2(p[0][2]), a01);
            a01 = __hfma2(w02, __half2half2(p[0][3]), a01);
            a01 = __hfma2(w10, __half2half2(p[1][1]), a01);
            a01 = __hfma2(w11, __half2half2(p[1][2]), a01);
            a01 = __hfma2(w12, __half2half2(p[1][3]), a01);
            a01 = __hfma2(w20, __half2half2(p[2][1]), a01);
            a01 = __hfma2(w21, __half2half2(p[2][2]), a01);
            a01 = __hfma2(w22, __half2half2(p[2][3]), a01);

            // a10: conv at (1,0)
            a10 = __hfma2(w00, __half2half2(p[1][0]), a10);
            a10 = __hfma2(w01, __half2half2(p[1][1]), a10);
            a10 = __hfma2(w02, __half2half2(p[1][2]), a10);
            a10 = __hfma2(w10, __half2half2(p[2][0]), a10);
            a10 = __hfma2(w11, __half2half2(p[2][1]), a10);
            a10 = __hfma2(w12, __half2half2(p[2][2]), a10);
            a10 = __hfma2(w20, __half2half2(p[3][0]), a10);
            a10 = __hfma2(w21, __half2half2(p[3][1]), a10);
            a10 = __hfma2(w22, __half2half2(p[3][2]), a10);

            // a11: conv at (1,1)
            a11 = __hfma2(w00, __half2half2(p[1][1]), a11);
            a11 = __hfma2(w01, __half2half2(p[1][2]), a11);
            a11 = __hfma2(w02, __half2half2(p[1][3]), a11);
            a11 = __hfma2(w10, __half2half2(p[2][1]), a11);
            a11 = __hfma2(w11, __half2half2(p[2][2]), a11);
            a11 = __hfma2(w12, __half2half2(p[2][3]), a11);
            a11 = __hfma2(w20, __half2half2(p[3][1]), a11);
            a11 = __hfma2(w21, __half2half2(p[3][2]), a11);
            a11 = __hfma2(w22, __half2half2(p[3][3]), a11);
        }

        // Bias + ReLU
        __half2 bv = has_bias ? c_bias_h2[co_pair] : __float2half2_rn(0.f);
        const __half2 zero = __float2half2_rn(0.f);
        a00 = __hmax2(__hadd2(a00, bv), zero);
        a01 = __hmax2(__hadd2(a01, bv), zero);
        a10 = __hmax2(__hadd2(a10, bv), zero);
        a11 = __hmax2(__hadd2(a11, bv), zero);

        // Average (use fp32 path for stability via float conversion not needed; __hadd2 then mul)
        __half2 sum = __hadd2(__hadd2(a00, a01), __hadd2(a10, a11));
        __half2 pooled = __hmul2(sum, __float2half2_rn(0.25f));

        int out_h = tile_ph0 + ph;
        int out_w = tile_pw0 + pw;
        if (out_h < oH && out_w < oW) {
            // Write two channels: co_pair*2 and co_pair*2+1
            int co0 = co_pair * 2;
            int co1 = co0 + 1;
            __half lo = __low2half(pooled);
            __half hi = __high2half(pooled);
            output[((n * Cout + co0) * oH + out_h) * oW + out_w] = lo;
            output[((n * Cout + co1) * oH + out_h) * oW + out_w] = hi;
        }
    }
}

// Helper to rearrange weights on device into half2 layout.
// Weight original layout: (Cout, Cin, 3, 3) row-major, fp16.
// New layout: c_weight_h2[co_pair * Cin * 9 + ci * 9 + k] =
//             half2(w[co_pair*2+0, ci, kh, kw], w[co_pair*2+1, ci, kh, kw])
__global__ void rearrange_weights_kernel(const __half* __restrict__ w_in,
                                          __half2* __restrict__ w_out,
                                          int Cin)
{
    int co_pair = blockIdx.x;  // 0..15
    int ci = blockIdx.y;       // 0..15
    int k  = threadIdx.x;      // 0..8
    if (k >= 9) return;

    int co0 = co_pair * 2;
    int co1 = co0 + 1;
    __half a = w_in[(co0 * Cin + ci) * 9 + k];
    __half b = w_in[(co1 * Cin + ci) * 9 + k];
    w_out[(co_pair * Cin + ci) * 9 + k] = __halves2half2(a, b);
}

__global__ void rearrange_bias_kernel(const __half* __restrict__ b_in,
                                       __half2* __restrict__ b_out)
{
    int co_pair = threadIdx.x;  // 0..15
    if (co_pair >= 16) return;
    __half a = b_in[co_pair * 2];
    __half b = b_in[co_pair * 2 + 1];
    b_out[co_pair] = __halves2half2(a, b);
}

void launch_fused_conv_relu_avgpool(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int N, int Cin, int Cout, int H, int W,
    cudaStream_t stream)
{
    // Rearrange weights into half2 layout in a temporary device buffer,
    // then copy to constant memory.
    static __half2* d_w_tmp = nullptr;
    static __half2* d_b_tmp = nullptr;
    if (!d_w_tmp) {
        cudaMalloc(&d_w_tmp, 16 * 16 * 9 * sizeof(__half2));
        cudaMalloc(&d_b_tmp, 16 * sizeof(__half2));
    }

    dim3 wgrid(16, 16);
    rearrange_weights_kernel<<<wgrid, 16, 0, stream>>>(weight, d_w_tmp, Cin);
    cudaMemcpyToSymbolAsync(c_weight_h2, d_w_tmp, 16 * 16 * 9 * sizeof(__half2),
                            0, cudaMemcpyDeviceToDevice, stream);

    int has_bias = 0;
    if (bias != nullptr) {
        rearrange_bias_kernel<<<1, 16, 0, stream>>>(bias, d_b_tmp);
        cudaMemcpyToSymbolAsync(c_bias_h2, d_b_tmp, 16 * sizeof(__half2),
                                0, cudaMemcpyDeviceToDevice, stream);
        has_bias = 1;
    }

    const int oH = H / 2;
    const int oW = W / 2;

    dim3 grid((oW + POOL_TILE_W - 1) / POOL_TILE_W,
              (oH + POOL_TILE_H - 1) / POOL_TILE_H,
              N);
    dim3 block(256);

    const int IN_TILE_SIZE = IN_TILE_H * IN_TILE_W;
    size_t smem_bytes = Cin * IN_TILE_SIZE * sizeof(__half);

    fused_conv_relu_avgpool_kernel<<<grid, block, smem_bytes, stream>>>(
        input, output, N, H, W, has_bias);
}