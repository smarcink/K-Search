#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

#define WH 4
#define WW 4
#define N 16
#define C_DIM 32
#define HIDDEN 128
#define WINS_PER_CTA 2

__global__ __launch_bounds__(32 * WINS_PER_CTA, 4) void swin_block_kernel(
    const __half* __restrict__ x_in,
    __half* __restrict__ x_out,
    const __half* __restrict__ ln1_w, const __half* __restrict__ ln1_b,
    const __half* __restrict__ qkv_w, const __half* __restrict__ qkv_b,
    const __half* __restrict__ proj_w, const __half* __restrict__ proj_b,
    const __half* __restrict__ ln2_w, const __half* __restrict__ ln2_b,
    const __half* __restrict__ fc1_w, const __half* __restrict__ fc1_b,
    const __half* __restrict__ fc2_w, const __half* __restrict__ fc2_b,
    const __half* __restrict__ rpe_bias,
    int H, int W)
{
    const int warp_id = threadIdx.y;
    const int tid = threadIdx.x;
    const int nW = W / WW;
    const int global_win_id = blockIdx.x * WINS_PER_CTA + warp_id;
    const int total_wins = (H / WH) * nW;
    if (global_win_id >= total_wins) return;
    const int b = blockIdx.y;
    const int nh = global_win_id / nW;
    const int nw = global_win_id % nW;

    __shared__ __half x_h_s[WINS_PER_CTA][N][C_DIM];
    __shared__ __half qkv_h_s[WINS_PER_CTA][N][3 * C_DIM];
    __shared__ __half kT_h_s[WINS_PER_CTA][C_DIM][N];
    __shared__ __half attn_h_s[WINS_PER_CTA][N][N];
    __shared__ __half out32_h_s[WINS_PER_CTA][N][C_DIM];
    __shared__ __half mlp_h_s[WINS_PER_CTA][N][HIDDEN];
    __shared__ float  resid_s[WINS_PER_CTA][N][C_DIM];
    __shared__ float  attn_f_s[WINS_PER_CTA][N][N];

    auto x_h = x_h_s[warp_id];
    auto qkv_h = qkv_h_s[warp_id];
    auto kT_h = kT_h_s[warp_id];
    auto attn_h = attn_h_s[warp_id];
    auto out32_h = out32_h_s[warp_id];
    auto mlp_h = mlp_h_s[warp_id];
    auto resid = resid_s[warp_id];
    auto attn_f = attn_f_s[warp_id];

    int hh_base = nh * WH;
    int ww_base = nw * WW;

    const int gid = tid >> 2;
    const int tig = tid & 3;
    int frag_row[8], frag_col[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int rr = gid + ((i & 2) ? 8 : 0);
        int cc = tig * 2 + (i & 1) + ((i & 4) ? 8 : 0);
        frag_row[i] = rr;
        frag_col[i] = cc;
    }

    // ---------- Load input window ----------
    #pragma unroll
    for (int i = 0; i < WH; i++) {
        int hh = hh_base + i;
        const __half2* src = reinterpret_cast<const __half2*>(
            x_in + ((b * H + hh) * W + ww_base) * C_DIM);
        int idx0 = tid;
        int idx1 = tid + 32;
        __half2 v0 = src[idx0];
        __half2 v1 = src[idx1];
        int t0 = idx0 / 16, c0 = (idx0 % 16) * 2;
        int t1 = idx1 / 16, c1 = (idx1 % 16) * 2;
        int n0 = i * WW + t0;
        int n1 = i * WW + t1;
        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        resid[n0][c0]   = f0.x; resid[n0][c0+1] = f0.y;
        resid[n1][c1]   = f1.x; resid[n1][c1+1] = f1.y;
        x_h[n0][c0]   = v0.x; x_h[n0][c0+1] = v0.y;
        x_h[n1][c1]   = v1.x; x_h[n1][c1+1] = v1.y;
    }

    // ---------- LN1 ----------
    {
        float w = __half2float(ln1_w[tid]);
        float b_ = __half2float(ln1_b[tid]);
        #pragma unroll
        for (int n = 0; n < N; n++) {
            float v = __half2float(x_h[n][tid]);
            float s = v;
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
            float mean = s * (1.0f / (float)C_DIM);
            float d = v - mean;
            float sq = d * d;
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
            float var = sq * (1.0f / (float)C_DIM);
            float inv = rsqrtf(var + 1e-5f);
            x_h[n][tid] = __float2half(d * inv * w + b_);
        }
    }
    __syncwarp();

    // ---------- QKV ----------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        #pragma unroll
        for (int nt = 0; nt < 6; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                wmma::load_matrix_sync(a_frag, &x_h[0][kt * 16], C_DIM);
                wmma::load_matrix_sync(b_frag, qkv_w + (nt * 16) * C_DIM + kt * 16, C_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = frag_row[i];
                int c = frag_col[i];
                int gcol = nt * 16 + c;
                float v = c_frag.x[i] + __half2float(qkv_b[gcol]);
                qkv_h[r][gcol] = __float2half(v);
            }
        }
    }
    __syncwarp();

    // Build K^T
    #pragma unroll
    for (int idx = tid; idx < C_DIM * N; idx += 32) {
        int k = idx / N;
        int n = idx % N;
        kT_h[k][n] = qkv_h[n][C_DIM + k];
    }
    __syncwarp();

    // ---------- Attn scores ----------
    const float scale = 1.0f / sqrtf((float)C_DIM);
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < 2; kt++) {
            wmma::load_matrix_sync(a_frag, &qkv_h[0][kt * 16], 3 * C_DIM);
            wmma::load_matrix_sync(b_frag, &kT_h[kt * 16][0], N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(&attn_f[0][0], c_frag, N, wmma::mem_row_major);
    }
    __syncwarp();

    if (tid < N) {
        int i = tid;
        float mx = -1e30f;
        float vals[N];
        #pragma unroll
        for (int j = 0; j < N; j++) {
            float v = attn_f[i][j] * scale + __half2float(rpe_bias[i * N + j]);
            vals[j] = v;
            mx = fmaxf(mx, v);
        }
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < N; j++) {
            float e = __expf(vals[j] - mx);
            vals[j] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        #pragma unroll
        for (int j = 0; j < N; j++) attn_h[i][j] = __float2half(vals[j] * inv);
    }
    __syncwarp();

    // ---------- attn @ V ----------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &attn_h[0][0], N);
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::load_matrix_sync(b_frag, &qkv_h[0][2 * C_DIM + nt * 16], 3 * C_DIM);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = frag_row[i];
                int c = frag_col[i];
                x_h[r][nt * 16 + c] = __float2half(c_frag.x[i]);
            }
        }
    }
    __syncwarp();

    // ---------- proj ----------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                wmma::load_matrix_sync(a_frag, &x_h[0][kt * 16], C_DIM);
                wmma::load_matrix_sync(b_frag, proj_w + (nt * 16) * C_DIM + kt * 16, C_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = frag_row[i];
                int c = frag_col[i];
                int gcol = nt * 16 + c;
                float v = c_frag.x[i] + __half2float(proj_b[gcol]);
                float y = resid[r][gcol] + v;
                resid[r][gcol] = y;
                out32_h[r][gcol] = __float2half(y);
            }
        }
    }
    __syncwarp();

    // ---------- LN2 ----------
    {
        float w = __half2float(ln2_w[tid]);
        float b_ = __half2float(ln2_b[tid]);
        #pragma unroll
        for (int n = 0; n < N; n++) {
            float v = __half2float(out32_h[n][tid]);
            float s = v;
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
            float mean = s * (1.0f / (float)C_DIM);
            float d = v - mean;
            float sq = d * d;
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
            float var = sq * (1.0f / (float)C_DIM);
            float inv = rsqrtf(var + 1e-5f);
            out32_h[n][tid] = __float2half(d * inv * w + b_);
        }
    }
    __syncwarp();

    // ---------- FC1 + GELU ----------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag_0;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag_1;
        wmma::load_matrix_sync(a_frag_0, &out32_h[0][0], C_DIM);
        wmma::load_matrix_sync(a_frag_1, &out32_h[0][16], C_DIM);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::load_matrix_sync(b_frag, fc1_w + (nt * 16) * C_DIM, C_DIM);
            wmma::mma_sync(c_frag, a_frag_0, b_frag, c_frag);
            wmma::load_matrix_sync(b_frag, fc1_w + (nt * 16) * C_DIM + 16, C_DIM);
            wmma::mma_sync(c_frag, a_frag_1, b_frag, c_frag);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = frag_row[i];
                int c = frag_col[i];
                int gcol = nt * 16 + c;
                float a = c_frag.x[i] + __half2float(fc1_b[gcol]);
                const float k0 = 0.7978845608f;
                const float k1 = 0.044715f;
                float t = k0 * (a + k1 * a * a * a);
                float g = 0.5f * a * (1.0f + tanhf(t));
                mlp_h[r][gcol] = __float2half(g);
            }
        }
    }
    __syncwarp();

    // ---------- FC2 ----------
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                wmma::load_matrix_sync(a_frag, &mlp_h[0][kt * 16], HIDDEN);
                wmma::load_matrix_sync(b_frag, fc2_w + (nt * 16) * HIDDEN + kt * 16, HIDDEN);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = frag_row[i];
                int c = frag_col[i];
                int gcol = nt * 16 + c;
                float v = c_frag.x[i] + __half2float(fc2_b[gcol]);
                resid[r][gcol] = resid[r][gcol] + v;
            }
        }
    }
    __syncwarp();

    // ---------- Store back ----------
    #pragma unroll
    for (int i = 0; i < WH; i++) {
        int hh = hh_base + i;
        __half2* dst = reinterpret_cast<__half2*>(
            x_out + ((b * H + hh) * W + ww_base) * C_DIM);
        int idx0 = tid;
        int idx1 = tid + 32;
        int t0 = idx0 / 16, c0 = (idx0 % 16) * 2;
        int t1 = idx1 / 16, c1 = (idx1 % 16) * 2;
        int n0 = i * WW + t0;
        int n1 = i * WW + t1;
        __half2 h0 = __floats2half2_rn(resid[n0][c0], resid[n0][c0+1]);
        __half2 h1 = __floats2half2_rn(resid[n1][c1], resid[n1][c1+1]);
        dst[idx0] = h0;
        dst[idx1] = h1;
    }
}

void launch_swin_block(
    const __half* x,
    __half* out,
    const __half* ln1_w, const __half* ln1_b,
    const __half* qkv_w, const __half* qkv_b,
    const __half* proj_w, const __half* proj_b,
    const __half* ln2_w, const __half* ln2_b,
    const __half* fc1_w, const __half* fc1_b,
    const __half* fc2_w, const __half* fc2_b,
    const __half* rpe_bias,
    int B, int H, int W, int C,
    cudaStream_t stream)
{
    int nH = H / WH;
    int nW = W / WW;
    int num_windows = nH * nW;
    int num_ctas = (num_windows + WINS_PER_CTA - 1) / WINS_PER_CTA;
    dim3 grid(num_ctas, B);
    dim3 block(32, WINS_PER_CTA);
    swin_block_kernel<<<grid, block, 0, stream>>>(
        x, out,
        ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
        ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b,
        rpe_bias, H, W);
}