#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Each thread processes 2 float4 (16 half elements) per iteration for better ILP
__global__ void __launch_bounds__(256, 4) elementwise_add_fp16_vec8(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ z,
    int n_vec8,
    int n_total
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    const float4* __restrict__ y4 = reinterpret_cast<const float4*>(y);
    float4* __restrict__ z4 = reinterpret_cast<float4*>(z);

    // Process 2 float4 per iteration for better memory-level parallelism
    int n_vec8_2 = n_vec8 / 2;
    for (int i = tid; i < n_vec8_2; i += stride) {
        int i0 = 2 * i;
        int i1 = 2 * i + 1;

        float4 xv0 = __ldg(&x4[i0]);
        float4 yv0 = __ldg(&y4[i0]);
        float4 xv1 = __ldg(&x4[i1]);
        float4 yv1 = __ldg(&y4[i1]);

        float4 zv0, zv1;
        __half2* zh0 = reinterpret_cast<__half2*>(&zv0);
        __half2* zh1 = reinterpret_cast<__half2*>(&zv1);
        const __half2* xh0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* yh0 = reinterpret_cast<const __half2*>(&yv0);
        const __half2* xh1 = reinterpret_cast<const __half2*>(&xv1);
        const __half2* yh1 = reinterpret_cast<const __half2*>(&yv1);

        zh0[0] = __hadd2(xh0[0], yh0[0]);
        zh0[1] = __hadd2(xh0[1], yh0[1]);
        zh0[2] = __hadd2(xh0[2], yh0[2]);
        zh0[3] = __hadd2(xh0[3], yh0[3]);

        zh1[0] = __hadd2(xh1[0], yh1[0]);
        zh1[1] = __hadd2(xh1[1], yh1[1]);
        zh1[2] = __hadd2(xh1[2], yh1[2]);
        zh1[3] = __hadd2(xh1[3], yh1[3]);

        z4[i0] = zv0;
        z4[i1] = zv1;
    }

    // Handle remaining single float4 chunks
    int tail_vec8_start = n_vec8_2 * 2;
    for (int i = tail_vec8_start + tid; i < n_vec8; i += stride) {
        float4 xv = __ldg(&x4[i]);
        float4 yv = __ldg(&y4[i]);

        float4 zv;
        __half2* zh = reinterpret_cast<__half2*>(&zv);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* yh = reinterpret_cast<const __half2*>(&yv);

        zh[0] = __hadd2(xh[0], yh[0]);
        zh[1] = __hadd2(xh[1], yh[1]);
        zh[2] = __hadd2(xh[2], yh[2]);
        zh[3] = __hadd2(xh[3], yh[3]);

        z4[i] = zv;
    }

    // Scalar tail for non-multiple-of-8 sizes
    int tail_start = n_vec8 * 8;
    for (int i = tail_start + tid; i < n_total; i += stride) {
        z[i] = __hadd(x[i], y[i]);
    }
}

void launch_elementwise_add_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ z,
    int n,
    cudaStream_t stream
)
{
    int n_vec8 = n / 8;
    int n_vec8_2 = n_vec8 / 2;

    const int THREADS = 256;
    // RTX5090 has 170 SMs
    // With 2 float4 per thread, we need n_vec8_2 = 65536 thread-iterations
    // Use exactly needed blocks to avoid wasted work, capped at 4 blocks/SM
    int num_sms = 170;
    int blocks_per_sm = 4;
    int max_blocks = num_sms * blocks_per_sm; // 680
    int needed_blocks = (n_vec8_2 + THREADS - 1) / THREADS; // ceil(65536/256) = 256
    int grid = min(max_blocks, needed_blocks);
    if (grid < 1) grid = 1;

    elementwise_add_fp16_vec8<<<grid, THREADS, 0, stream>>>(x, y, z, n_vec8, n);

    cudaGetLastError();
}