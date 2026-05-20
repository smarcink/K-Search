#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

using namespace nvcuda;

#define DIM 32
#define HIDDEN 128
#define N_TOK 16
#define WH 4
#define WW 4
#define WINS_PER_BLOCK 4

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__constant__ __half c_norm1_w[DIM];
__constant__ __half c_norm1_b[DIM];
__constant__ __half c_qkv_b[3*DIM];
__constant__ __half c_proj_b[DIM];
__constant__ __half c_norm2_w[DIM];
__constant__ __half c_norm2_b[DIM];
__constant__ __half c_fc1_b[HIDDEN];
__constant__ __half c_fc2_b[DIM];
__constant__ __half c_rpe_bias[N_TOK*N_TOK];

struct WinSmem {
    __half resid[N_TOK*DIM];
    __half ln[N_TOK*DIM];
    __half qkv[N_TOK*3*DIM];
    __half attn[N_TOK*N_TOK];
    __half attno[N_TOK*DIM];
    __half hidden[N_TOK*HIDDEN];
};

__global__ __launch_bounds__(128, 4) void swin_block_kernel_v2(
    const __half* __restrict__ x_in,
    __half* __restrict__ x_out,
    const __half* __restrict__ qkv_w,
    const __half* __restrict__ proj_w,
    const __half* __restrict__ fc1_w,
    const __half* __restrict__ fc2_w,
    int B, int H, int W, int C,
    int nH, int nW)
{
    int wlocal = threadIdx.y;
    int win_id = blockIdx.x * WINS_PER_BLOCK + wlocal;
    int total_windows = B * nH * nW;
    if (win_id >= total_windows) return;

    int total_per_b = nH * nW;
    int b = win_id / total_per_b;
    int rem = win_id % total_per_b;
    int wh = rem / nW;
    int ww = rem % nW;

    int lane = threadIdx.x;

    extern __shared__ __half smem_all[];
    WinSmem* sm = reinterpret_cast<WinSmem*>(smem_all) + wlocal;

    __half* sh_resid = sm->resid;
    __half* sh_ln    = sm->ln;
    __half* sh_qkv   = sm->qkv;
    __half* sh_attn  = sm->attn;
    __half* sh_attno = sm->attno;
    __half* sh_hidden= sm->hidden;

    __shared__ float scratch_fp32[WINS_PER_BLOCK][16*16];
    float* sscratch = scratch_fp32[wlocal];

    if (lane < N_TOK) {
        int tok = lane;
        int h_idx = wh * WH + (tok >> 2);
        int w_idx = ww * WW + (tok & 3);
        const __half* x_ptr = x_in + ((b * H + h_idx) * W + w_idx) * C;

        float4 q0 = *reinterpret_cast<const float4*>(x_ptr + 0);
        float4 q1 = *reinterpret_cast<const float4*>(x_ptr + 8);
        float4 q2 = *reinterpret_cast<const float4*>(x_ptr + 16);
        float4 q3 = *reinterpret_cast<const float4*>(x_ptr + 24);

        *reinterpret_cast<float4*>(&sh_resid[tok * DIM + 0])  = q0;
        *reinterpret_cast<float4*>(&sh_resid[tok * DIM + 8])  = q1;
        *reinterpret_cast<float4*>(&sh_resid[tok * DIM + 16]) = q2;
        *reinterpret_cast<float4*>(&sh_resid[tok * DIM + 24]) = q3;

        const __half* h0 = reinterpret_cast<const __half*>(&q0);
        const __half* h1 = reinterpret_cast<const __half*>(&q1);
        const __half* h2 = reinterpret_cast<const __half*>(&q2);
        const __half* h3 = reinterpret_cast<const __half*>(&q3);

        float vs[DIM];
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; k++) { vs[k]      = __half2float(h0[k]); sum += vs[k]; }
        #pragma unroll
        for (int k = 0; k < 8; k++) { vs[k+8]    = __half2float(h1[k]); sum += vs[k+8]; }
        #pragma unroll
        for (int k = 0; k < 8; k++) { vs[k+16]   = __half2float(h2[k]); sum += vs[k+16]; }
        #pragma unroll
        for (int k = 0; k < 8; k++) { vs[k+24]   = __half2float(h3[k]); sum += vs[k+24]; }

        float mean = sum * (1.0f / (float)DIM);
        float sq = 0.f;
        #pragma unroll
        for (int k = 0; k < DIM; k++) { float d = vs[k]-mean; sq += d*d; }
        float invstd = rsqrtf(sq * (1.0f / (float)DIM) + 1e-5f);
        #pragma unroll
        for (int k = 0; k < DIM; k += 2) {
            float w0 = __half2float(c_norm1_w[k]);
            float b0 = __half2float(c_norm1_b[k]);
            float w1 = __half2float(c_norm1_w[k+1]);
            float b1 = __half2float(c_norm1_b[k+1]);
            __half2 hh = __floats2half2_rn((vs[k]-mean)*invstd*w0 + b0,
                                           (vs[k+1]-mean)*invstd*w1 + b1);
            *reinterpret_cast<__half2*>(&sh_ln[tok*DIM + k]) = hh;
        }
    }
    __syncwarp();

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b0, b1;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        wmma::load_matrix_sync(a0, sh_ln + 0,  DIM);
        wmma::load_matrix_sync(a1, sh_ln + 16, DIM);

        #pragma unroll
        for (int nt = 0; nt < 6; nt++) {
            wmma::fill_fragment(c, 0.0f);
            const __half* bp0 = qkv_w + (nt*16)*32 + 0;
            const __half* bp1 = qkv_w + (nt*16)*32 + 16;
            wmma::load_matrix_sync(b0, bp0, 32);
            wmma::load_matrix_sync(b1, bp1, 32);
            wmma::mma_sync(c, a0, b0, c);
            wmma::mma_sync(c, a1, b1, c);
            wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = (lane * 4 + e) * 2;
                int r = idx >> 4;
                int col = idx & 15;
                int oc = nt*16 + col;
                float v0 = sscratch[r*16 + col]     + __half2float(c_qkv_b[oc]);
                float v1 = sscratch[r*16 + col + 1] + __half2float(c_qkv_b[oc + 1]);
                __half2 h2 = __floats2half2_rn(v0, v1);
                *reinterpret_cast<__half2*>(&sh_qkv[r * (3*DIM) + oc]) = h2;
            }
            __syncwarp();
        }
    }

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> qa0, qa1;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> kb0c, kb1c;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        wmma::load_matrix_sync(qa0, sh_qkv + 0,  3*DIM);
        wmma::load_matrix_sync(qa1, sh_qkv + 16, 3*DIM);
        wmma::load_matrix_sync(kb0c, sh_qkv + 32 + 0,  3*DIM);
        wmma::load_matrix_sync(kb1c, sh_qkv + 32 + 16, 3*DIM);

        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, qa0, kb0c, c);
        wmma::mma_sync(c, qa1, kb1c, c);
        wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
        __syncwarp();

        float scale = 1.0f / sqrtf((float)DIM);
        if (lane < N_TOK) {
            int i = lane;
            float vals[N_TOK];
            float mx = -CUDART_INF_F;
            #pragma unroll
            for (int j = 0; j < N_TOK; j++) {
                vals[j] = sscratch[i*16 + j] * scale + __half2float(c_rpe_bias[i*N_TOK + j]);
                mx = fmaxf(mx, vals[j]);
            }
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < N_TOK; j++) { vals[j] = __expf(vals[j]-mx); s += vals[j]; }
            float inv_s = 1.0f / s;
            #pragma unroll
            for (int j = 0; j < N_TOK; j += 2) {
                __half2 h2 = __floats2half2_rn(vals[j]*inv_s, vals[j+1]*inv_s);
                *reinterpret_cast<__half2*>(&sh_attn[i*N_TOK + j]) = h2;
            }
        }
        __syncwarp();
    }

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b0;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        wmma::load_matrix_sync(a, sh_attn, N_TOK);

        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fill_fragment(c, 0.0f);
            wmma::load_matrix_sync(b0, sh_qkv + 64 + nt*16, 3*DIM);
            wmma::mma_sync(c, a, b0, c);
            wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = (lane * 4 + e) * 2;
                int r = idx >> 4;
                int col = idx & 15;
                int oc = nt*16 + col;
                __half2 h2 = __floats2half2_rn(sscratch[r*16 + col], sscratch[r*16 + col + 1]);
                *reinterpret_cast<__half2*>(&sh_attno[r*DIM + oc]) = h2;
            }
            __syncwarp();
        }
    }

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b0, b1;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        wmma::load_matrix_sync(a0, sh_attno + 0,  DIM);
        wmma::load_matrix_sync(a1, sh_attno + 16, DIM);

        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fill_fragment(c, 0.0f);
            wmma::load_matrix_sync(b0, proj_w + (nt*16)*32 + 0,  32);
            wmma::load_matrix_sync(b1, proj_w + (nt*16)*32 + 16, 32);
            wmma::mma_sync(c, a0, b0, c);
            wmma::mma_sync(c, a1, b1, c);
            wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = (lane * 4 + e) * 2;
                int r = idx >> 4;
                int col = idx & 15;
                int oc = nt*16 + col;
                float v0 = sscratch[r*16 + col]     + __half2float(c_proj_b[oc]);
                float v1 = sscratch[r*16 + col + 1] + __half2float(c_proj_b[oc + 1]);
                __half2 res = *reinterpret_cast<__half2*>(&sh_resid[r*DIM + oc]);
                float2 rf = __half22float2(res);
                __half2 h2 = __floats2half2_rn(v0 + rf.x, v1 + rf.y);
                *reinterpret_cast<__half2*>(&sh_resid[r*DIM + oc]) = h2;
            }
            __syncwarp();
        }
    }

    if (lane < N_TOK) {
        int tok = lane;
        float vs[DIM];
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < DIM; k++) {
            vs[k] = __half2float(sh_resid[tok * DIM + k]);
            sum += vs[k];
        }
        float mean = sum / (float)DIM;
        float sq = 0.f;
        #pragma unroll
        for (int k = 0; k < DIM; k++) { float d = vs[k]-mean; sq += d*d; }
        float invstd = rsqrtf(sq/(float)DIM + 1e-5f);
        #pragma unroll
        for (int k = 0; k < DIM; k++) {
            float w = __half2float(c_norm2_w[k]);
            float bb = __half2float(c_norm2_b[k]);
            sh_ln[tok*DIM + k] = __float2half((vs[k]-mean)*invstd*w + bb);
        }
    }
    __syncwarp();

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b0, b1;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        wmma::load_matrix_sync(a0, sh_ln + 0,  DIM);
        wmma::load_matrix_sync(a1, sh_ln + 16, DIM);

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wmma::fill_fragment(c, 0.0f);
            wmma::load_matrix_sync(b0, fc1_w + (nt*16)*32 + 0,  32);
            wmma::load_matrix_sync(b1, fc1_w + (nt*16)*32 + 16, 32);
            wmma::mma_sync(c, a0, b0, c);
            wmma::mma_sync(c, a1, b1, c);
            wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = (lane * 4 + e) * 2;
                int r = idx >> 4;
                int col = idx & 15;
                int oc = nt*16 + col;
                float v0 = sscratch[r*16 + col]     + __half2float(c_fc1_b[oc]);
                float v1 = sscratch[r*16 + col + 1] + __half2float(c_fc1_b[oc + 1]);
                __half2 h2 = __floats2half2_rn(gelu(v0), gelu(v1));
                *reinterpret_cast<__half2*>(&sh_hidden[r*HIDDEN + oc]) = h2;
            }
            __syncwarp();
        }
    }

    {
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a[8];
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16,16,16, float> c;

        #pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            wmma::load_matrix_sync(a[kt], sh_hidden + kt*16, HIDDEN);
        }

        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                wmma::load_matrix_sync(b, fc2_w + (nt*16)*128 + kt*16, 128);
                wmma::mma_sync(c, a[kt], b, c);
            }
            wmma::store_matrix_sync(sscratch, c, 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = (lane * 4 + e) * 2;
                int r = idx >> 4;
                int col = idx & 15;
                int oc = nt*16 + col;
                float v0 = sscratch[r*16 + col]     + __half2float(c_fc2_b[oc]);
                float v1 = sscratch[r*16 + col + 1] + __half2float(c_fc2_b[oc + 1]);
                __half2 res = *reinterpret_cast<__half2*>(&sh_resid[r*DIM + oc]);
                float2 rf = __half22float2(res);
                __half2 h2 = __floats2half2_rn(v0 + rf.x, v1 + rf.y);
                *reinterpret_cast<__half2*>(&sh_resid[r*DIM + oc]) = h2;
            }
            __syncwarp();
        }
    }

    if (lane < N_TOK) {
        int tok = lane;
        int h_idx = wh * WH + (tok >> 2);
        int w_idx = ww * WW + (tok & 3);
        __half* o_ptr = x_out + ((b * H + h_idx) * W + w_idx) * C;
        float4 q0 = *reinterpret_cast<const float4*>(&sh_resid[tok * DIM + 0]);
        float4 q1 = *reinterpret_cast<const float4*>(&sh_resid[tok * DIM + 8]);
        float4 q2 = *reinterpret_cast<const float4*>(&sh_resid[tok * DIM + 16]);
        float4 q3 = *reinterpret_cast<const float4*>(&sh_resid[tok * DIM + 24]);
        *reinterpret_cast<float4*>(o_ptr + 0)  = q0;
        *reinterpret_cast<float4*>(o_ptr + 8)  = q1;
        *reinterpret_cast<float4*>(o_ptr + 16) = q2;
        *reinterpret_cast<float4*>(o_ptr + 24) = q3;
    }
}

void launch_swin_block(
    const __half* x,
    const __half* norm1_w, const __half* norm1_b,
    const __half* qkv_w, const __half* qkv_b,
    const __half* proj_w, const __half* proj_b,
    const __half* norm2_w, const __half* norm2_b,
    const __half* fc1_w, const __half* fc1_b,
    const __half* fc2_w, const __half* fc2_b,
    const __half* rpe_bias,
    __half* out,
    int B, int H, int W, int C,
    cudaStream_t stream)
{
    cudaMemcpyToSymbolAsync(c_norm1_w, norm1_w, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_norm1_b, norm1_b, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_qkv_b, qkv_b, 3*DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_proj_b, proj_b, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_norm2_w, norm2_w, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_norm2_b, norm2_b, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_fc1_b, fc1_b, HIDDEN*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_fc2_b, fc2_b, DIM*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyToSymbolAsync(c_rpe_bias, rpe_bias, N_TOK*N_TOK*sizeof(__half), 0, cudaMemcpyDeviceToDevice, stream);

    int nH = H / WH;
    int nW = W / WW;
    int num_windows = B * nH * nW;
    int num_blocks = (num_windows + WINS_PER_BLOCK - 1) / WINS_PER_BLOCK;

    dim3 grid(num_blocks);
    dim3 block(32, WINS_PER_BLOCK);

    size_t smem_bytes = WINS_PER_BLOCK * sizeof(WinSmem);

    swin_block_kernel_v2<<<grid, block, smem_bytes, stream>>>(
        x, out, qkv_w, proj_w, fc1_w, fc2_w, B, H, W, C, nH, nW);
}