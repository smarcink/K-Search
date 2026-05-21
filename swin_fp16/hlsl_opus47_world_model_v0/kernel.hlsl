// Fused Swin block, single window per group. v3: process 2 windows per group with 128 threads
// to amortize barrier costs and increase ILP. Keep fp16 smem, fp32 accumulation for LN/softmax.

#define H_DIM 720
#define W_DIM 1280
#define C_DIM 32
#define WIN 4
#define N_TOK 16
#define HIDDEN 128
#define NW (W_DIM / WIN)
#define WPG 2          // windows per group
#define TPG 128        // threads per group
#define TPW 64         // threads per window

Buffer<float16_t> in_x        : register(t0);
Buffer<float16_t> rpe_bias    : register(t1);
Buffer<float16_t> norm1_w     : register(t2);
Buffer<float16_t> norm1_b     : register(t3);
Buffer<float16_t> qkv_w       : register(t4);
Buffer<float16_t> qkv_b       : register(t5);
Buffer<float16_t> proj_w      : register(t6);
Buffer<float16_t> proj_b      : register(t7);
Buffer<float16_t> norm2_w     : register(t8);
Buffer<float16_t> norm2_b     : register(t9);
Buffer<float16_t> fc1_w       : register(t10);
Buffer<float16_t> fc1_b       : register(t11);
Buffer<float16_t> fc2_w       : register(t12);
Buffer<float16_t> fc2_b       : register(t13);

RWBuffer<float16_t> out_y     : register(u0);

// Per-window groupshared, indexed by window slot [WPG]
groupshared float16_t sx    [WPG][N_TOK][C_DIM];
groupshared float16_t sn    [WPG][N_TOK][C_DIM];
groupshared float16_t sq    [WPG][N_TOK][C_DIM];
groupshared float16_t sk    [WPG][N_TOK][C_DIM];
groupshared float16_t sv    [WPG][N_TOK][C_DIM];
groupshared float    sattn [WPG][N_TOK][N_TOK];
groupshared float16_t sres1 [WPG][N_TOK][C_DIM];
groupshared float16_t shid  [WPG][N_TOK][HIDDEN];
// total = WPG * (5*512 + 256 + 512 + 2048) * 2B (fp16) + WPG*256*4B (sattn)
// = 2 * (5*512 + 512 + 2048)*2 + 2*256*4 + 2*256*2
// fp16 bytes: 2*(2560 + 512 + 2048 + 256)*2 wait recompute:
// sx,sn,sq,sk,sv,sres1: 6 arrays * 512 fp16 = 3072 fp16 per window = 6144 B
// shid: 2048 fp16 = 4096 B
// sattn: 256 fp32 = 1024 B
// per window: 6144 + 4096 + 1024 = 11264 B
// 2 windows: 22528 B < 32 KiB. OK.

[numthreads(TPG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    uint w_slot = tid / TPW;        // 0 or 1
    uint w_tid  = tid % TPW;        // 0..63
    uint win_idx = gid.x * WPG + w_slot;

    uint nh = win_idx / NW;
    uint nw = win_idx % NW;
    uint row0 = nh * WIN;
    uint col0 = nw * WIN;

    // Load input: 512 elems / 64 threads => 8 each
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / C_DIM;
        uint c = idx % C_DIM;
        uint th = t / WIN;
        uint tw = t % WIN;
        uint gh = row0 + th;
        uint gw = col0 + tw;
        uint gidx = ((gh * W_DIM) + gw) * C_DIM + c;
        sx[w_slot][t][c] = in_x[gidx];
    }
    GroupMemoryBarrierWithGroupSync();

    // LayerNorm1: 16 tokens per window, fp32 accum
    if (w_tid < N_TOK) {
        uint t = w_tid;
        float sum = 0.0;
        [unroll] for (uint c = 0; c < C_DIM; ++c) sum += (float)sx[w_slot][t][c];
        float mean = sum / (float)C_DIM;
        float vs = 0.0;
        [unroll] for (uint c2 = 0; c2 < C_DIM; ++c2) {
            float d = (float)sx[w_slot][t][c2] - mean;
            vs += d * d;
        }
        float var = vs / (float)C_DIM;
        float inv = rsqrt(var + 1e-5);
        [unroll] for (uint c3 = 0; c3 < C_DIM; ++c3) {
            float v = ((float)sx[w_slot][t][c3] - mean) * inv;
            v = v * (float)norm1_w[c3] + (float)norm1_b[c3];
            sn[w_slot][t][c3] = (float16_t)v;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // QKV: 512 outputs / 64 = 8
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / C_DIM;
        uint c = idx % C_DIM;
        float q = (float)qkv_b[c];
        float k = (float)qkv_b[c + 32];
        float v = (float)qkv_b[c + 64];
        [unroll] for (uint ic = 0; ic < C_DIM; ++ic) {
            float xn = (float)sn[w_slot][t][ic];
            q += xn * (float)qkv_w[c * C_DIM + ic];
            k += xn * (float)qkv_w[(c + 32) * C_DIM + ic];
            v += xn * (float)qkv_w[(c + 64) * C_DIM + ic];
        }
        sq[w_slot][t][c] = (float16_t)q;
        sk[w_slot][t][c] = (float16_t)k;
        sv[w_slot][t][c] = (float16_t)v;
    }
    GroupMemoryBarrierWithGroupSync();

    // Attention scores: 256 / 64 = 4
    float scale = rsqrt((float)C_DIM);
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        uint idx = w_tid + i * TPW;
        uint t1 = idx / N_TOK;
        uint t2 = idx % N_TOK;
        float s = 0.0;
        [unroll] for (uint c = 0; c < C_DIM; ++c) {
            s += (float)sq[w_slot][t1][c] * (float)sk[w_slot][t2][c];
        }
        s = s * scale + (float)rpe_bias[t1 * N_TOK + t2];
        sattn[w_slot][t1][t2] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    // Softmax row-wise, fp32
    if (w_tid < N_TOK) {
        uint t1 = w_tid;
        float mx = sattn[w_slot][t1][0];
        [unroll] for (uint t2 = 1; t2 < N_TOK; ++t2) mx = max(mx, sattn[w_slot][t1][t2]);
        float sm = 0.0;
        [unroll] for (uint t2b = 0; t2b < N_TOK; ++t2b) {
            float e = exp(sattn[w_slot][t1][t2b] - mx);
            sattn[w_slot][t1][t2b] = e;
            sm += e;
        }
        float inv = 1.0 / sm;
        [unroll] for (uint t2c = 0; t2c < N_TOK; ++t2c) sattn[w_slot][t1][t2c] *= inv;
    }
    GroupMemoryBarrierWithGroupSync();

    // attn @ v -> reuse sq
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / C_DIM;
        uint c = idx % C_DIM;
        float s = 0.0;
        [unroll] for (uint t2 = 0; t2 < N_TOK; ++t2) {
            s += sattn[w_slot][t][t2] * (float)sv[w_slot][t2][c];
        }
        sq[w_slot][t][c] = (float16_t)s;
    }
    GroupMemoryBarrierWithGroupSync();

    // proj + residual
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / C_DIM;
        uint c = idx % C_DIM;
        float s = (float)proj_b[c];
        [unroll] for (uint ic = 0; ic < C_DIM; ++ic) {
            s += (float)sq[w_slot][t][ic] * (float)proj_w[c * C_DIM + ic];
        }
        float r = (float)sx[w_slot][t][c] + s;
        sres1[w_slot][t][c] = (float16_t)r;
    }
    GroupMemoryBarrierWithGroupSync();

    // LayerNorm2
    if (w_tid < N_TOK) {
        uint t = w_tid;
        float sum = 0.0;
        [unroll] for (uint c = 0; c < C_DIM; ++c) sum += (float)sres1[w_slot][t][c];
        float mean = sum / (float)C_DIM;
        float vs = 0.0;
        [unroll] for (uint c2 = 0; c2 < C_DIM; ++c2) {
            float d = (float)sres1[w_slot][t][c2] - mean;
            vs += d * d;
        }
        float var = vs / (float)C_DIM;
        float inv = rsqrt(var + 1e-5);
        [unroll] for (uint c3 = 0; c3 < C_DIM; ++c3) {
            float v = ((float)sres1[w_slot][t][c3] - mean) * inv;
            v = v * (float)norm2_w[c3] + (float)norm2_b[c3];
            sn[w_slot][t][c3] = (float16_t)v;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // fc1 + GELU: 2048 / 64 = 32
    [unroll]
    for (uint i = 0; i < 32; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / HIDDEN;
        uint h = idx % HIDDEN;
        float s = (float)fc1_b[h];
        [unroll] for (uint ic = 0; ic < C_DIM; ++ic) {
            s += (float)sn[w_slot][t][ic] * (float)fc1_w[h * C_DIM + ic];
        }
        float x3 = s * s * s;
        float inner = 0.7978845608 * (s + 0.044715 * x3);
        float th = tanh(inner);
        float gelu = 0.5 * s * (1.0 + th);
        shid[w_slot][t][h] = (float16_t)gelu;
    }
    GroupMemoryBarrierWithGroupSync();

    // fc2 + residual + store: 512 / 64 = 8
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = w_tid + i * TPW;
        uint t = idx / C_DIM;
        uint c = idx % C_DIM;
        float s = (float)fc2_b[c];
        [unroll] for (uint h = 0; h < HIDDEN; ++h) {
            s += (float)shid[w_slot][t][h] * (float)fc2_w[c * HIDDEN + h];
        }
        float final_v = (float)sres1[w_slot][t][c] + s;
        uint th = t / WIN;
        uint tw = t % WIN;
        uint gh = row0 + th;
        uint gw = col0 + tw;
        uint gidx = ((gh * W_DIM) + gw) * C_DIM + c;
        out_y[gidx] = (float16_t)final_v;
    }
}