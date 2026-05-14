#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* ====================================================================
 * Conv2d 3×3, stride=1, pad=1  (NCHW, fp16)
 *
 * Each thread computes one output element out[n, co, h, w] by iterating
 * over Cin × 3 × 3.  Deliberately simple — no shared-memory tiling,
 * no im2col — so K-Search has room to optimise.
 * ==================================================================== */
__global__ void conv2d_3x3_kernel(
    const __half* __restrict__ input,   // (N, Cin, H, W)
    const __half* __restrict__ weight,  // (Cout, Cin, 3, 3)
    const __half* __restrict__ bias,    // (Cout,) or nullptr
    __half* __restrict__ output,        // (N, Cout, H, W)
    int N, int Cin, int Cout, int H, int W)
{
    int total = N * Cout * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W;
    int h_out = (idx / W) % H;
    int co    = (idx / (W * H)) % Cout;
    int n     = idx / (W * H * Cout);

    float sum = 0.0f;

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h_out + kh - 1;  // pad=1
                int iw = w_out + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float iv = __half2float(input[((n * Cin + ci) * H + ih) * W + iw]);
                    float wv = __half2float(weight[((co * Cin + ci) * 3 + kh) * 3 + kw]);
                    sum += iv * wv;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __half2float(bias[co]);
    }

    output[((n * Cout + co) * H + h_out) * W + w_out] = __float2half(sum);
}

void launch_conv2d_3x3(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int N, int Cin, int Cout, int H, int W,
    cudaStream_t stream)
{
    int total = N * Cout * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_3x3_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, N, Cin, Cout, H, W);
}

/* ====================================================================
 * Fused ReLU + AvgPool 2×2  (NCHW, fp16)
 * ==================================================================== */
__global__ void relu_avgpool2x2_kernel(
    const __half* __restrict__ in,
    __half* __restrict__ out,
    int N, int C, int H, int W)
{
    int oH = H / 2;
    int oW = W / 2;
    int total = N * C * oH * oW;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose linear index -> (n, c, oh, ow)
    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int c  = (idx / (oW * oH)) % C;
    int n  = idx / (oW * oH * C);

    int ih = oh * 2;
    int iw = ow * 2;

    // Gather the 2x2 window, apply ReLU, average
    int base = ((n * C + c) * H + ih) * W + iw;
    float v00 = fmaxf(__half2float(in[base]),         0.0f);
    float v01 = fmaxf(__half2float(in[base + 1]),     0.0f);
    float v10 = fmaxf(__half2float(in[base + W]),     0.0f);
    float v11 = fmaxf(__half2float(in[base + W + 1]), 0.0f);

    float avg = (v00 + v01 + v10 + v11) * 0.25f;

    int out_idx = ((n * C + c) * oH + oh) * oW + ow;
    out[out_idx] = __float2half(avg);
}

void launch_relu_avgpool2x2(
    const __half* in, __half* out,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int oH = H / 2;
    int oW = W / 2;
    int total = N * C * oH * oW;

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    relu_avgpool2x2_kernel<<<blocks, threads, 0, stream>>>(in, out, N, C, H, W);
}
