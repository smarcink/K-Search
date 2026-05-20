// Swin block - revert to base WPG=8 which scored 2.34x (current WPG=4 regressed to 1.26x).
// Keep proj_w and fc2_w in registers via SRV loads; only qkv_w and fc1_w in groupshared.

Buffer<float16_t> in_x       : register(t0);
Buffer<float16_t> rpe_bias   : register(t1);
Buffer<float16_t> norm1_w    : register(t2);
Buffer<float16_t> norm1_b    : register(t3);
Buffer<float16_t> qkv_w      : register(t4);
Buffer<float16_t> qkv_b      : register(t5);
Buffer<float16_t> proj_w     : register(t6);
Buffer<float16_t> proj_b     : register(t7);
Buffer<float16_t> norm2_w    : register(t8);
Buffer<float16_t> norm2_b    : register(t9);
Buffer<float16_t> fc1_w      : register(t10);
Buffer<float16_t> fc1_b      : register(t11);
Buffer<float16_t> fc2_w      : register(t12);
Buffer<float16_t> fc2_b      : register(t13);

RWBuffer<float16_t> out_y    : register(u0);

#define H 720
#define W 1280
#define C 32
#define HID 128
#define WH 4
#define WW 4
#define N 16
#define WPG 8
#define TPG (N * WPG)   // 128

groupshared float16_t s_qkv_w[96*32];
groupshared float16_t s_fc1_w[128*32];
groupshared float16_t s_K[WPG][N][C];
groupshared float16_t s_V[WPG][N][C];

[numthreads(TPG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint lt = tid.x;
    uint t  = lt & (N - 1);
    uint w  = lt / N;

    uint wx = gid.x * WPG + w;
    uint wy = gid.y;

    [unroll]
    for (uint i = 0; i < 24; ++i) {
        uint idx = i * TPG + lt;
        s_qkv_w[idx] = qkv_w[idx];
    }
    [unroll]
    for (uint i = 0; i < 32; ++i) {
        uint idx = i * TPG + lt;
        s_fc1_w[idx] = fc1_w[idx];
    }

    uint tr = t / WW;
    uint tc = t & (WW - 1);
    uint gy = wy * WH + tr;
    uint gx = wx * WW + tc;
    uint base_in = ((gy * W) + gx) * C;

    float x[C];
    [unroll] for (uint c = 0; c < C; ++c) x[c] = (float)in_x[base_in + c];

    float x_orig[C];
    [unroll] for (uint c = 0; c < C; ++c) x_orig[c] = x[c];

    float sum = 0.0;
    [unroll] for (uint c = 0; c < C; ++c) sum += x[c];
    float mean = sum * (1.0 / (float)C);
    float vs = 0.0;
    [unroll] for (uint c = 0; c < C; ++c) { float d = x[c] - mean; vs += d * d; }
    float invstd = rsqrt(vs * (1.0 / (float)C) + 1e-5);

    GroupMemoryBarrierWithGroupSync();

    float xn[C];
    [unroll] for (uint c = 0; c < C; ++c) {
        xn[c] = (x[c] - mean) * invstd * (float)norm1_w[c] + (float)norm1_b[c];
    }

    float q[C], kk[C], vv[C];
    [unroll]
    for (uint i = 0; i < C; ++i) {
        float qa = (float)qkv_b[i];
        float ka = (float)qkv_b[C + i];
        float va = (float)qkv_b[2*C + i];
        [unroll]
        for (uint c = 0; c < C; ++c) {
            float xc = xn[c];
            qa += xc * (float)s_qkv_w[i * C + c];
            ka += xc * (float)s_qkv_w[(C + i) * C + c];
            va += xc * (float)s_qkv_w[(2*C + i) * C + c];
        }
        q[i] = qa; kk[i] = ka; vv[i] = va;
    }

    [unroll] for (uint c = 0; c < C; ++c) {
        s_K[w][t][c] = (float16_t)kk[c];
        s_V[w][t][c] = (float16_t)vv[c];
    }

    GroupMemoryBarrierWithGroupSync();

    float scale = rsqrt((float)C);
    float scores[N];
    float maxs = -1e30;
    [unroll]
    for (uint j = 0; j < N; ++j) {
        float s = 0.0;
        [unroll] for (uint c = 0; c < C; ++c) s += q[c] * (float)s_K[w][j][c];
        s = s * scale + (float)rpe_bias[t * N + j];
        scores[j] = s;
        maxs = max(maxs, s);
    }
    float esum = 0.0;
    [unroll] for (uint j = 0; j < N; ++j) { scores[j] = exp(scores[j] - maxs); esum += scores[j]; }
    float inv_esum = 1.0 / esum;
    [unroll] for (uint j = 0; j < N; ++j) scores[j] *= inv_esum;

    float ao[C];
    [unroll]
    for (uint c = 0; c < C; ++c) {
        float s = 0.0;
        [unroll] for (uint j = 0; j < N; ++j) s += scores[j] * (float)s_V[w][j][c];
        ao[c] = s;
    }

    float po[C];
    [unroll]
    for (uint i = 0; i < C; ++i) {
        float a = (float)proj_b[i];
        [unroll] for (uint c = 0; c < C; ++c) a += ao[c] * (float)proj_w[i * C + c];
        po[i] = a;
    }

    float x_res2[C];
    [unroll] for (uint c = 0; c < C; ++c) { x[c] = x_orig[c] + po[c]; x_res2[c] = x[c]; }

    float sum2 = 0.0;
    [unroll] for (uint c = 0; c < C; ++c) sum2 += x[c];
    float mean2 = sum2 * (1.0 / (float)C);
    float vs2 = 0.0;
    [unroll] for (uint c = 0; c < C; ++c) { float d = x[c] - mean2; vs2 += d * d; }
    float invstd2 = rsqrt(vs2 * (1.0 / (float)C) + 1e-5);
    float xn2[C];
    [unroll] for (uint c = 0; c < C; ++c) {
        xn2[c] = (x[c] - mean2) * invstd2 * (float)norm2_w[c] + (float)norm2_b[c];
    }

    float hh[HID];
    [unroll]
    for (uint i = 0; i < HID; ++i) {
        float a = (float)fc1_b[i];
        [unroll] for (uint c = 0; c < C; ++c) a += xn2[c] * (float)s_fc1_w[i * C + c];
        float xv = a;
        float t1 = 0.7978845608 * (xv + 0.044715 * xv * xv * xv);
        hh[i] = 0.5 * xv * (1.0 + tanh(t1));
    }

    [unroll]
    for (uint i = 0; i < C; ++i) {
        float a = (float)fc2_b[i];
        [unroll] for (uint hi = 0; hi < HID; ++hi) a += hh[hi] * (float)fc2_w[i * HID + hi];
        out_y[base_in + i] = (float16_t)(x_res2[i] + a);
    }
}