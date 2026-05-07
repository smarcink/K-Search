#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// NHWC AvgPool 2x2 - optimized
__global__ void avgpool2x2_nhwc_kernel(const __half* __restrict__ in,
                                        __half* __restrict__ out,
                                        int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;

    int n  = blockIdx.z;
    int oh = blockIdx.y;
    int ow_idx = blockIdx.x;

    int ih = oh << 1;
    int iw = ow_idx << 1;

    int C_v = C >> 3;
    int c_v = threadIdx.x;
    if (c_v >= C_v) return;

    const __half* p00 = in + (((n * H + ih    ) * W + (iw    )) * C);
    const __half* p01 = in + (((n * H + ih    ) * W + (iw + 1)) * C);
    const __half* p10 = in + (((n * H + ih + 1) * W + (iw    )) * C);
    const __half* p11 = in + (((n * H + ih + 1) * W + (iw + 1)) * C);

    const float4* v00 = reinterpret_cast<const float4*>(p00) + c_v;
    const float4* v01 = reinterpret_cast<const float4*>(p01) + c_v;
    const float4* v10 = reinterpret_cast<const float4*>(p10) + c_v;
    const float4* v11 = reinterpret_cast<const float4*>(p11) + c_v;

    float4 r00 = __ldg(v00);
    float4 r01 = __ldg(v01);
    float4 r10 = __ldg(v10);
    float4 r11 = __ldg(v11);

    __half2 a00[4], a01[4], a10[4], a11[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a00[i] = *(reinterpret_cast<__half2*>(&r00) + i);
        a01[i] = *(reinterpret_cast<__half2*>(&r01) + i);
        a10[i] = *(reinterpret_cast<__half2*>(&r10) + i);
        a11[i] = *(reinterpret_cast<__half2*>(&r11) + i);
    }

    const __half2 quarter = __float2half2_rn(0.25f);
    __half2 o[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __half2 s = __hadd2(__hadd2(a00[i], a01[i]), __hadd2(a10[i], a11[i]));
        o[i] = __hmul2(s, quarter);
    }

    float4 packed;
    packed.x = *reinterpret_cast<float*>(&o[0]);
    packed.y = *reinterpret_cast<float*>(&o[1]);
    packed.z = *reinterpret_cast<float*>(&o[2]);
    packed.w = *reinterpret_cast<float*>(&o[3]);

    __half* op = out + (((n * oH + oh) * oW + ow_idx) * C);
    float4* outv = reinterpret_cast<float4*>(op) + c_v;
    *outv = packed;
}

// Tiled multi-output kernel
template<int OW_PER_BLOCK>
__global__ void avgpool2x2_nhwc_tiled_kernel(const __half* __restrict__ in,
                                              __half* __restrict__ out,
                                              int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;

    int C_v = C >> 3;
    int n  = blockIdx.z;
    int oh = blockIdx.y;
    int ow_base = blockIdx.x * OW_PER_BLOCK;

    int tid = threadIdx.x;
    int c_v = tid % C_v;
    int ow_off = tid / C_v;

    if (ow_off >= OW_PER_BLOCK) return;
    int ow_idx = ow_base + ow_off;
    if (ow_idx >= oW) return;

    int ih = oh << 1;
    int iw = ow_idx << 1;

    const __half* p00 = in + (((n * H + ih    ) * W + (iw    )) * C);
    const __half* p01 = in + (((n * H + ih    ) * W + (iw + 1)) * C);
    const __half* p10 = in + (((n * H + ih + 1) * W + (iw    )) * C);
    const __half* p11 = in + (((n * H + ih + 1) * W + (iw + 1)) * C);

    const float4* v00 = reinterpret_cast<const float4*>(p00) + c_v;
    const float4* v01 = reinterpret_cast<const float4*>(p01) + c_v;
    const float4* v10 = reinterpret_cast<const float4*>(p10) + c_v;
    const float4* v11 = reinterpret_cast<const float4*>(p11) + c_v;

    float4 r00 = __ldg(v00);
    float4 r01 = __ldg(v01);
    float4 r10 = __ldg(v10);
    float4 r11 = __ldg(v11);

    __half2 a00[4], a01[4], a10[4], a11[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a00[i] = *(reinterpret_cast<__half2*>(&r00) + i);
        a01[i] = *(reinterpret_cast<__half2*>(&r01) + i);
        a10[i] = *(reinterpret_cast<__half2*>(&r10) + i);
        a11[i] = *(reinterpret_cast<__half2*>(&r11) + i);
    }

    const __half2 quarter = __float2half2_rn(0.25f);
    __half2 o[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        __half2 s = __hadd2(__hadd2(a00[i], a01[i]), __hadd2(a10[i], a11[i]));
        o[i] = __hmul2(s, quarter);
    }

    float4 packed;
    packed.x = *reinterpret_cast<float*>(&o[0]);
    packed.y = *reinterpret_cast<float*>(&o[1]);
    packed.z = *reinterpret_cast<float*>(&o[2]);
    packed.w = *reinterpret_cast<float*>(&o[3]);

    __half* op = out + (((n * oH + oh) * oW + ow_idx) * C);
    float4* outv = reinterpret_cast<float4*>(op) + c_v;
    *outv = packed;
}

__global__ void avgpool2x2_nhwc_generic_kernel(const __half* __restrict__ in,
                                                __half* __restrict__ out,
                                                int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;
    int total = N * oH * oW * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx % C;
    int ow = (idx / C) % oW;
    int oh = (idx / (C * oW)) % oH;
    int n = idx / (C * oW * oH);

    int ih = oh * 2;
    int iw = ow * 2;

    int b00 = (((n * H + ih    ) * W + (iw    )) * C) + c;
    int b01 = (((n * H + ih    ) * W + (iw + 1)) * C) + c;
    int b10 = (((n * H + ih + 1) * W + (iw    )) * C) + c;
    int b11 = (((n * H + ih + 1) * W + (iw + 1)) * C) + c;

    float v = __half2float(in[b00]) + __half2float(in[b01])
            + __half2float(in[b10]) + __half2float(in[b11]);
    out[idx] = __float2half(v * 0.25f);
}

void launch_avgpool2x2_nhwc(const __half* in, __half* out,
                             int N, int H, int W, int C,
                             cudaStream_t stream) {
    if ((C % 8) == 0 && (H % 2) == 0 && (W % 2) == 0) {
        int oH = H / 2;
        int oW = W / 2;
        int C_v = C / 8;

        if (C_v <= 32) {
            constexpr int OW_PER_BLOCK = 8;
            int threads = C_v * OW_PER_BLOCK;
            dim3 block(threads, 1, 1);
            dim3 grid((oW + OW_PER_BLOCK - 1) / OW_PER_BLOCK, oH, N);
            avgpool2x2_nhwc_tiled_kernel<OW_PER_BLOCK><<<grid, block, 0, stream>>>(in, out, N, H, W, C);
        } else {
            if (C_v > 1024) C_v = 1024;
            dim3 block(C_v, 1, 1);
            dim3 grid(oW, oH, N);
            avgpool2x2_nhwc_kernel<<<grid, block, 0, stream>>>(in, out, N, H, W, C);
        }
    } else {
        int oH = H / 2;
        int oW = W / 2;
        int total = N * oH * oW * C;
        int block = 256;
        int grid = (total + block - 1) / block;
        avgpool2x2_nhwc_generic_kernel<<<grid, block, 0, stream>>>(in, out, N, H, W, C);
    }
}

// Fused NHWC ReLU + AvgPool 2x2 -> NCHW: optimized to process more ow per thread (4 ow per thread)
// using 16-byte vectorized loads for adjacent 2 ow positions
template<int OW_PER_THREAD, int OH_PER_BLOCK>
__global__ void relu_avgpool2x2_nhwc_to_nchw_v2_kernel(const __half* __restrict__ in,
                                                        __half* __restrict__ out,
                                                        int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;

    int C_v = C >> 3;  // num of 8-half chunks per row
    int n  = blockIdx.z;
    int oh_base = blockIdx.y * OH_PER_BLOCK;
    int ow_base = blockIdx.x * OW_PER_THREAD;

    int c_v = threadIdx.x;
    if (c_v >= C_v) return;

    const __half2 zero = __float2half2_rn(0.f);
    const __half2 quarter = __float2half2_rn(0.25f);

    int c_base = c_v << 3;
    int nc_stride = oH * oW;
    __half* op_n = out + n * C * nc_stride;

    #pragma unroll
    for (int oh_off = 0; oh_off < OH_PER_BLOCK; oh_off++) {
        int oh = oh_base + oh_off;
        if (oh >= oH) break;
        int ih = oh << 1;

        #pragma unroll
        for (int ow_off = 0; ow_off < OW_PER_THREAD; ow_off++) {
            int ow_idx = ow_base + ow_off;
            if (ow_idx >= oW) break;
            int iw = ow_idx << 1;

            const __half* p00 = in + (((n * H + ih    ) * W + (iw    )) * C);
            const __half* p01 = in + (((n * H + ih    ) * W + (iw + 1)) * C);
            const __half* p10 = in + (((n * H + ih + 1) * W + (iw    )) * C);
            const __half* p11 = in + (((n * H + ih + 1) * W + (iw + 1)) * C);

            float4 r00 = __ldg(reinterpret_cast<const float4*>(p00) + c_v);
            float4 r01 = __ldg(reinterpret_cast<const float4*>(p01) + c_v);
            float4 r10 = __ldg(reinterpret_cast<const float4*>(p10) + c_v);
            float4 r11 = __ldg(reinterpret_cast<const float4*>(p11) + c_v);

            __half2 o[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __half2 a00 = *(reinterpret_cast<__half2*>(&r00) + i);
                __half2 a01 = *(reinterpret_cast<__half2*>(&r01) + i);
                __half2 a10 = *(reinterpret_cast<__half2*>(&r10) + i);
                __half2 a11 = *(reinterpret_cast<__half2*>(&r11) + i);
                a00 = __hmax2(a00, zero);
                a01 = __hmax2(a01, zero);
                a10 = __hmax2(a10, zero);
                a11 = __hmax2(a11, zero);
                __half2 s = __hadd2(__hadd2(a00, a01), __hadd2(a10, a11));
                o[i] = __hmul2(s, quarter);
            }

            int spatial = oh * oW + ow_idx;
            __half* op_base = op_n + spatial;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __half lo = __low2half(o[i]);
                __half hi = __high2half(o[i]);
                op_base[(c_base + i*2 + 0) * nc_stride] = lo;
                op_base[(c_base + i*2 + 1) * nc_stride] = hi;
            }
        }
    }
}

template<int OW_PER_BLOCK, int OH_PER_BLOCK>
__global__ void relu_avgpool2x2_nhwc_to_nchw_kernel(const __half* __restrict__ in,
                                                     __half* __restrict__ out,
                                                     int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;

    int C_v = C >> 3;
    int n  = blockIdx.z;
    int oh_base = blockIdx.y * OH_PER_BLOCK;
    int ow_base = blockIdx.x * OW_PER_BLOCK;

    int tid = threadIdx.x;
    int c_v = tid % C_v;
    int ow_off = tid / C_v;

    if (ow_off >= OW_PER_BLOCK) return;
    int ow_idx = ow_base + ow_off;
    if (ow_idx >= oW) return;

    int iw = ow_idx << 1;

    const __half2 zero = __float2half2_rn(0.f);
    const __half2 quarter = __float2half2_rn(0.25f);

    int c_base = c_v << 3;

    #pragma unroll
    for (int oh_off = 0; oh_off < OH_PER_BLOCK; oh_off++) {
        int oh = oh_base + oh_off;
        if (oh >= oH) break;
        int ih = oh << 1;

        const __half* p00 = in + (((n * H + ih    ) * W + (iw    )) * C);
        const __half* p01 = in + (((n * H + ih    ) * W + (iw + 1)) * C);
        const __half* p10 = in + (((n * H + ih + 1) * W + (iw    )) * C);
        const __half* p11 = in + (((n * H + ih + 1) * W + (iw + 1)) * C);

        const float4* v00 = reinterpret_cast<const float4*>(p00) + c_v;
        const float4* v01 = reinterpret_cast<const float4*>(p01) + c_v;
        const float4* v10 = reinterpret_cast<const float4*>(p10) + c_v;
        const float4* v11 = reinterpret_cast<const float4*>(p11) + c_v;

        float4 r00 = __ldg(v00);
        float4 r01 = __ldg(v01);
        float4 r10 = __ldg(v10);
        float4 r11 = __ldg(v11);

        __half2 o[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __half2 a00 = *(reinterpret_cast<__half2*>(&r00) + i);
            __half2 a01 = *(reinterpret_cast<__half2*>(&r01) + i);
            __half2 a10 = *(reinterpret_cast<__half2*>(&r10) + i);
            __half2 a11 = *(reinterpret_cast<__half2*>(&r11) + i);
            a00 = __hmax2(a00, zero);
            a01 = __hmax2(a01, zero);
            a10 = __hmax2(a10, zero);
            a11 = __hmax2(a11, zero);
            __half2 s = __hadd2(__hadd2(a00, a01), __hadd2(a10, a11));
            o[i] = __hmul2(s, quarter);
        }

        int spatial = oh * oW + ow_idx;
        int nc_stride = oH * oW;
        __half* op_base = out + n * C * nc_stride + spatial;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __half lo = __low2half(o[i]);
            __half hi = __high2half(o[i]);
            op_base[(c_base + i*2 + 0) * nc_stride] = lo;
            op_base[(c_base + i*2 + 1) * nc_stride] = hi;
        }
    }
}

template<int OW_PER_BLOCK, int OH_PER_BLOCK>
__global__ void relu_avgpool2x2_nhwc_tiled2_kernel(const __half* __restrict__ in,
                                                    __half* __restrict__ out,
                                                    int N, int H, int W, int C) {
    int oH = H >> 1;
    int oW = W >> 1;

    int C_v = C >> 3;
    int n  = blockIdx.z;
    int oh_base = blockIdx.y * OH_PER_BLOCK;
    int ow_base = blockIdx.x * OW_PER_BLOCK;

    int tid = threadIdx.x;
    int c_v = tid % C_v;
    int ow_off = tid / C_v;

    if (ow_off >= OW_PER_BLOCK) return;
    int ow_idx = ow_base + ow_off;
    if (ow_idx >= oW) return;

    int iw = ow_idx << 1;

    const __half2 zero = __float2half2_rn(0.f);
    const __half2 quarter = __float2half2_rn(0.25f);

    #pragma unroll
    for (int oh_off = 0; oh_off < OH_PER_BLOCK; oh_off++) {
        int oh = oh_base + oh_off;
        if (oh >= oH) return;
        int ih = oh << 1;

        const __half* p00 = in + (((n * H + ih    ) * W + (iw    )) * C);
        const __half* p01 = in + (((n * H + ih    ) * W + (iw + 1)) * C);
        const __half* p10 = in + (((n * H + ih + 1) * W + (iw    )) * C);
        const __half* p11 = in + (((n * H + ih + 1) * W + (iw + 1)) * C);

        const float4* v00 = reinterpret_cast<const float4*>(p00) + c_v;
        const float4* v01 = reinterpret_cast<const float4*>(p01) + c_v;
        const float4* v10 = reinterpret_cast<const float4*>(p10) + c_v;
        const float4* v11 = reinterpret_cast<const float4*>(p11) + c_v;

        float4 r00 = __ldg(v00);
        float4 r01 = __ldg(v01);
        float4 r10 = __ldg(v10);
        float4 r11 = __ldg(v11);

        __half2 o[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __half2 a00 = *(reinterpret_cast<__half2*>(&r00) + i);
            __half2 a01 = *(reinterpret_cast<__half2*>(&r01) + i);
            __half2 a10 = *(reinterpret_cast<__half2*>(&r10) + i);
            __half2 a11 = *(reinterpret_cast<__half2*>(&r11) + i);
            a00 = __hmax2(a00, zero);
            a01 = __hmax2(a01, zero);
            a10 = __hmax2(a10, zero);
            a11 = __hmax2(a11, zero);
            __half2 s = __hadd2(__hadd2(a00, a01), __hadd2(a10, a11));
            o[i] = __hmul2(s, quarter);
        }

        float4 packed;
        packed.x = *reinterpret_cast<float*>(&o[0]);
        packed.y = *reinterpret_cast<float*>(&o[1]);
        packed.z = *reinterpret_cast<float*>(&o[2]);
        packed.w = *reinterpret_cast<float*>(&o[3]);

        __half* op = out + (((n * oH + oh) * oW + ow_idx) * C);
        float4* outv = reinterpret_cast<float4*>(op) + c_v;
        *outv = packed;
    }
}

void launch_relu_avgpool2x2_nhwc(const __half* in, __half* out,
                                  int N, int H, int W, int C,
                                  cudaStream_t stream) {
    if ((C % 8) == 0 && (H % 2) == 0 && (W % 2) == 0) {
        int oH = H / 2;
        int oW = W / 2;
        int C_v = C / 8;

        if (C_v <= 32) {
            constexpr int OW_PER_BLOCK = 8;
            constexpr int OH_PER_BLOCK = 2;
            int threads = C_v * OW_PER_BLOCK;
            dim3 block(threads, 1, 1);
            dim3 grid((oW + OW_PER_BLOCK - 1) / OW_PER_BLOCK,
                      (oH + OH_PER_BLOCK - 1) / OH_PER_BLOCK,
                      N);
            relu_avgpool2x2_nhwc_tiled2_kernel<OW_PER_BLOCK, OH_PER_BLOCK><<<grid, block, 0, stream>>>(in, out, N, H, W, C);
        } else {
            launch_avgpool2x2_nhwc(in, out, N, H, W, C, stream);
        }
    } else {
        launch_avgpool2x2_nhwc(in, out, N, H, W, C, stream);
    }
}

void launch_relu_avgpool2x2_nhwc_to_nchw(const __half* in, __half* out,
                                          int N, int H, int W, int C,
                                          cudaStream_t stream) {
    if ((C % 8) == 0 && (H % 2) == 0 && (W % 2) == 0) {
        int oH = H / 2;
        int oW = W / 2;
        int C_v = C / 8;

        if (C_v <= 32) {
            constexpr int OW_PER_BLOCK = 8;
            constexpr int OH_PER_BLOCK = 2;
            int threads = C_v * OW_PER_BLOCK;
            dim3 block(threads, 1, 1);
            dim3 grid((oW + OW_PER_BLOCK - 1) / OW_PER_BLOCK,
                      (oH + OH_PER_BLOCK - 1) / OH_PER_BLOCK,
                      N);
            relu_avgpool2x2_nhwc_to_nchw_kernel<OW_PER_BLOCK, OH_PER_BLOCK><<<grid, block, 0, stream>>>(in, out, N, H, W, C);
        }
    }
}

// ============= NCHW kernels (fallback) =============

static __device__ __forceinline__ __half2 relu_h2(__half2 v) {
    const __half2 zero = __float2half2_rn(0.f);
    return __hmax2(v, zero);
}

static __device__ __forceinline__ __half pool4(__half2 a, __half2 b) {
    __half2 s = __hadd2(relu_h2(a), relu_h2(b));
    float sum = __half2float(s.x) + __half2float(s.y);
    return __float2half(sum * 0.25f);
}

__global__ void relu_avgpool2x2_vec8x2_kernel(const __half* __restrict__ in,
                                              __half* __restrict__ out,
                                              int N, int C, int H, int W) {
    int oH = H >> 1;
    int oW = W >> 1;

    int ow_oct = blockIdx.x * blockDim.x + threadIdx.x;
    int oh_pair = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    int oW_o = oW >> 3;
    int oH_p = oH >> 1;
    if (ow_oct >= oW_o || oh_pair >= oH_p) return;

    int ow0 = ow_oct << 3;
    int iw0 = ow0 << 1;
    int oh0 = oh_pair << 1;
    int ih0 = oh0 << 1;

    #pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        int ih = ih0 + (rr << 1);
        int oh = oh0 + rr;

        int row0_base = (nc * H + ih) * W + iw0;
        int row1_base = row0_base + W;

        const float4* p0 = reinterpret_cast<const float4*>(in + row0_base);
        const float4* p1 = reinterpret_cast<const float4*>(in + row1_base);
        float4 r0a = __ldg(p0);
        float4 r0b = __ldg(p0 + 1);
        float4 r1a = __ldg(p1);
        float4 r1b = __ldg(p1 + 1);

        __half2 a0 = *reinterpret_cast<__half2*>(&r0a.x);
        __half2 a1 = *reinterpret_cast<__half2*>(&r0a.y);
        __half2 a2 = *reinterpret_cast<__half2*>(&r0a.z);
        __half2 a3 = *reinterpret_cast<__half2*>(&r0a.w);
        __half2 a4 = *reinterpret_cast<__half2*>(&r0b.x);
        __half2 a5 = *reinterpret_cast<__half2*>(&r0b.y);
        __half2 a6 = *reinterpret_cast<__half2*>(&r0b.z);
        __half2 a7 = *reinterpret_cast<__half2*>(&r0b.w);

        __half2 b0 = *reinterpret_cast<__half2*>(&r1a.x);
        __half2 b1 = *reinterpret_cast<__half2*>(&r1a.y);
        __half2 b2 = *reinterpret_cast<__half2*>(&r1a.z);
        __half2 b3 = *reinterpret_cast<__half2*>(&r1a.w);
        __half2 b4 = *reinterpret_cast<__half2*>(&r1b.x);
        __half2 b5 = *reinterpret_cast<__half2*>(&r1b.y);
        __half2 b6 = *reinterpret_cast<__half2*>(&r1b.z);
        __half2 b7 = *reinterpret_cast<__half2*>(&r1b.w);

        __half o0 = pool4(a0, b0);
        __half o1 = pool4(a1, b1);
        __half o2 = pool4(a2, b2);
        __half o3 = pool4(a3, b3);
        __half o4 = pool4(a4, b4);
        __half o5 = pool4(a5, b5);
        __half o6 = pool4(a6, b6);
        __half o7 = pool4(a7, b7);

        __half2 out01 = __halves2half2(o0, o1);
        __half2 out23 = __halves2half2(o2, o3);
        __half2 out45 = __halves2half2(o4, o5);
        __half2 out67 = __halves2half2(o6, o7);

        int out_base = (nc * oH + oh) * oW + ow0;
        float4 packed;
        packed.x = *reinterpret_cast<float*>(&out01);
        packed.y = *reinterpret_cast<float*>(&out23);
        packed.z = *reinterpret_cast<float*>(&out45);
        packed.w = *reinterpret_cast<float*>(&out67);
        *reinterpret_cast<float4*>(out + out_base) = packed;
    }
}

__global__ void relu_avgpool2x2_kernel(const __half* __restrict__ in,
                                        __half* __restrict__ out,
                                        int N, int C, int H, int W) {
    int oH = H / 2;
    int oW = W / 2;
    int total = N * C * oH * oW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int c  = (idx / (oW * oH)) % C;
    int n  = idx / (oW * oH * C);

    int ih = oh * 2;
    int iw = ow * 2;
    int base = ((n * C + c) * H + ih) * W + iw;

    float v0 = __half2float(in[base]);
    float v1 = __half2float(in[base + 1]);
    float v2 = __half2float(in[base + W]);
    float v3 = __half2float(in[base + W + 1]);

    v0 = v0 > 0.f ? v0 : 0.f;
    v1 = v1 > 0.f ? v1 : 0.f;
    v2 = v2 > 0.f ? v2 : 0.f;
    v3 = v3 > 0.f ? v3 : 0.f;

    float avg = (v0 + v1 + v2 + v3) * 0.25f;
    out[idx] = __float2half(avg);
}

void launch_relu_avgpool2x2(const __half* in, __half* out,
                             int N, int C, int H, int W,
                             cudaStream_t stream) {
    int oH = H / 2;
    int oW = W / 2;
    if ((W % 16) == 0 && (H % 4) == 0) {
        int oW_o = oW / 8;
        int oH_p = oH / 2;
        dim3 block(32, 4, 1);
        dim3 grid((oW_o + block.x - 1) / block.x,
                  (oH_p + block.y - 1) / block.y,
                  N * C);
        relu_avgpool2x2_vec8x2_kernel<<<grid, block, 0, stream>>>(in, out, N, C, H, W);
    } else {
        int total = N * C * oH * oW;
        int block = 256;
        int grid = (total + block - 1) / block;
        relu_avgpool2x2_kernel<<<grid, block, 0, stream>>>(in, out, N, C, H, W);
    }
}