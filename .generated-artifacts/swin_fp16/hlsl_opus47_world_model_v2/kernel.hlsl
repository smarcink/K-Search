// Fused Swin block: one threadgroup = two windows (32 tokens total).
// Variant: cooperative FC1/FC2 across the wave using wave intrinsics.
// Each lane owns one token through attention; for MLP we cooperate
// across the 16 lanes of a window to compute FC1 hidden values, then
// FC2 outputs, with WaveReadLaneAt to share the per-token activations.

#define DIM 32
#define HIDDEN 128
#define WIN 4
#define N 16
#define W 1280
#define H 720
#define NW (W/WIN)
#define NH (H/WIN)

Buffer<float16_t> g_in       : register(t0);
Buffer<float16_t> g_rpe      : register(t1);
Buffer<float16_t> g_n1w      : register(t2);
Buffer<float16_t> g_n1b      : register(t3);
Buffer<float16_t> g_qkvw     : register(t4);
Buffer<float16_t> g_qkvb     : register(t5);
Buffer<float16_t> g_projw    : register(t6);
Buffer<float16_t> g_projb    : register(t7);
Buffer<float16_t> g_n2w      : register(t8);
Buffer<float16_t> g_n2b      : register(t9);
Buffer<float16_t> g_fc1w     : register(t10);
Buffer<float16_t> g_fc1b     : register(t11);
Buffer<float16_t> g_fc2w     : register(t12);
Buffer<float16_t> g_fc2b     : register(t13);

RWBuffer<float16_t> g_out    : register(u0);

groupshared float16_t s_rpe[256];
groupshared float16_t s_n1w[DIM];
groupshared float16_t s_n1b[DIM];
groupshared float16_t s_n2w[DIM];
groupshared float16_t s_n2b[DIM];
groupshared float16_t s_qkvw[3*DIM*DIM];
groupshared float16_t s_qkvb[3*DIM];
groupshared float16_t s_projw[DIM*DIM];
groupshared float16_t s_projb[DIM];
groupshared float16_t s_fc1w[HIDDEN*DIM];
groupshared float16_t s_fc1b[HIDDEN];
groupshared float16_t s_fc2w[DIM*HIDDEN];
groupshared float16_t s_fc2b[DIM];

groupshared float16_t s_k[2 * N * DIM];
groupshared float16_t s_v[2 * N * DIM];

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID)
{
    uint group_id = gid.x;
    uint lane = tid;
    uint win_in_grp = lane >> 4;
    uint token = lane & 15;

    [unroll] for (uint i0 = 0; i0 < 8; ++i0) {
        uint i = i0 * 32 + lane;
        s_rpe[i] = g_rpe[i];
    }
    if (lane < DIM) {
        s_n1w[lane] = g_n1w[lane];
        s_n1b[lane] = g_n1b[lane];
        s_n2w[lane] = g_n2w[lane];
        s_n2b[lane] = g_n2b[lane];
        s_projb[lane] = g_projb[lane];
        s_fc2b[lane] = g_fc2b[lane];
    }
    [unroll] for (uint i1 = 0; i1 < 96; ++i1) {
        s_qkvw[i1 * 32 + lane] = g_qkvw[i1 * 32 + lane];
    }
    [unroll] for (uint i2 = 0; i2 < 3; ++i2) {
        s_qkvb[i2 * 32 + lane] = g_qkvb[i2 * 32 + lane];
    }
    [unroll] for (uint i3 = 0; i3 < 32; ++i3) {
        s_projw[i3 * 32 + lane] = g_projw[i3 * 32 + lane];
    }
    [unroll] for (uint i4 = 0; i4 < 4; ++i4) {
        s_fc1b[i4 * 32 + lane] = g_fc1b[i4 * 32 + lane];
    }
    [unroll] for (uint i5 = 0; i5 < 128; ++i5) {
        s_fc1w[i5 * 32 + lane] = g_fc1w[i5 * 32 + lane];
    }
    [unroll] for (uint i6 = 0; i6 < 128; ++i6) {
        s_fc2w[i6 * 32 + lane] = g_fc2w[i6 * 32 + lane];
    }

    GroupMemoryBarrierWithGroupSync();

    uint window_id = group_id * 2 + win_in_grp;
    if (window_id >= (uint)(NH * NW)) return;

    uint nh = window_id / NW;
    uint nw = window_id - nh * NW;
    uint y0 = nh * WIN;
    uint x0 = nw * WIN;

    uint ty = token >> 2;
    uint tx = token & 3;
    uint gy = y0 + ty;
    uint gx = x0 + tx;
    uint base = (gy * W + gx) * DIM;

    float16_t x_in[DIM];
    [unroll] for (uint c = 0; c < DIM; ++c) {
        x_in[c] = g_in[base + c];
    }

    float mean = 0.0, var = 0.0;
    [unroll] for (uint c = 0; c < DIM; ++c) mean += (float)x_in[c];
    mean /= (float)DIM;
    [unroll] for (uint c = 0; c < DIM; ++c) {
        float v = (float)x_in[c] - mean;
        var += v*v;
    }
    var /= (float)DIM;
    float inv = rsqrt(var + 1e-5);

    float16_t x_norm[DIM];
    [unroll] for (uint c = 0; c < DIM; ++c) {
        float vv = ((float)x_in[c] - mean) * inv;
        x_norm[c] = (float16_t)(vv * (float)s_n1w[c] + (float)s_n1b[c]);
    }

    // --- QKV linear ---
    float16_t q[DIM];
    uint kv_base = win_in_grp * N * DIM + token * DIM;
    [unroll] for (uint r = 0; r < DIM; ++r) {
        float16_t accq = s_qkvb[r];
        float16_t acck = s_qkvb[DIM + r];
        float16_t accv = s_qkvb[2*DIM + r];
        uint rowq = r * DIM;
        uint rowk = (DIM + r) * DIM;
        uint rowv = (2*DIM + r) * DIM;
        [unroll] for (uint c = 0; c < DIM; c += 4) {
            vector<float16_t,4> xv = vector<float16_t,4>(x_norm[c], x_norm[c+1], x_norm[c+2], x_norm[c+3]);
            vector<float16_t,4> wq = vector<float16_t,4>(s_qkvw[rowq+c], s_qkvw[rowq+c+1], s_qkvw[rowq+c+2], s_qkvw[rowq+c+3]);
            vector<float16_t,4> wk = vector<float16_t,4>(s_qkvw[rowk+c], s_qkvw[rowk+c+1], s_qkvw[rowk+c+2], s_qkvw[rowk+c+3]);
            vector<float16_t,4> wv = vector<float16_t,4>(s_qkvw[rowv+c], s_qkvw[rowv+c+1], s_qkvw[rowv+c+2], s_qkvw[rowv+c+3]);
            accq += dot(xv, wq);
            acck += dot(xv, wk);
            accv += dot(xv, wv);
        }
        q[r] = accq;
        s_k[kv_base + r] = acck;
        s_v[kv_base + r] = accv;
    }

    GroupMemoryBarrierWithGroupSync();

    const float scale = 1.0 / sqrt((float)DIM);
    uint k_win_base = win_in_grp * N * DIM;
    float row[N];
    [unroll] for (uint j = 0; j < N; ++j) {
        float16_t dotv = (float16_t)0.0;
        uint kbase = k_win_base + j * DIM;
        [unroll] for (uint c = 0; c < DIM; c += 4) {
            vector<float16_t,4> qv = vector<float16_t,4>(q[c], q[c+1], q[c+2], q[c+3]);
            vector<float16_t,4> kv = vector<float16_t,4>(s_k[kbase+c], s_k[kbase+c+1], s_k[kbase+c+2], s_k[kbase+c+3]);
            dotv += dot(qv, kv);
        }
        row[j] = (float)dotv * scale + (float)s_rpe[token * N + j];
    }
    float m = row[0];
    [unroll] for (uint j = 1; j < N; ++j) m = max(m, row[j]);
    float sum = 0.0;
    [unroll] for (uint j = 0; j < N; ++j) {
        row[j] = exp(row[j] - m);
        sum += row[j];
    }
    float invs = 1.0 / sum;
    [unroll] for (uint j = 0; j < N; ++j) row[j] *= invs;

    float16_t av[DIM];
    [unroll] for (uint c = 0; c < DIM; ++c) {
        float acc = 0.0;
        [unroll] for (uint j = 0; j < N; ++j) {
            acc += row[j] * (float)s_v[k_win_base + j * DIM + c];
        }
        av[c] = (float16_t)acc;
    }

    // --- Proj + residual ---
    float16_t post_attn[DIM];
    [unroll] for (uint r = 0; r < DIM; ++r) {
        float16_t acc = s_projb[r];
        uint rowp = r * DIM;
        [unroll] for (uint c = 0; c < DIM; c += 4) {
            vector<float16_t,4> avv = vector<float16_t,4>(av[c], av[c+1], av[c+2], av[c+3]);
            vector<float16_t,4> wv = vector<float16_t,4>(s_projw[rowp+c], s_projw[rowp+c+1], s_projw[rowp+c+2], s_projw[rowp+c+3]);
            acc += dot(avv, wv);
        }
        post_attn[r] = acc + x_in[r];
    }

    float mean2 = 0.0, var2 = 0.0;
    [unroll] for (uint c = 0; c < DIM; ++c) mean2 += (float)post_attn[c];
    mean2 /= (float)DIM;
    [unroll] for (uint c = 0; c < DIM; ++c) {
        float vv = (float)post_attn[c] - mean2;
        var2 += vv*vv;
    }
    var2 /= (float)DIM;
    float inv2 = rsqrt(var2 + 1e-5);

    float16_t xn2[DIM];
    [unroll] for (uint c = 0; c < DIM; ++c) {
        float vv = ((float)post_attn[c] - mean2) * inv2;
        xn2[c] = (float16_t)(vv * (float)s_n2w[c] + (float)s_n2b[c]);
    }

    // --- FC1 -> GELU -> FC2 streamed in fp16 accumulators ---
    float16_t hidden[HIDDEN];
    [unroll] for (uint h = 0; h < HIDDEN; ++h) {
        float16_t acc = s_fc1b[h];
        uint rowh = h * DIM;
        [unroll] for (uint c = 0; c < DIM; c += 4) {
            vector<float16_t,4> xv = vector<float16_t,4>(xn2[c], xn2[c+1], xn2[c+2], xn2[c+3]);
            vector<float16_t,4> wv = vector<float16_t,4>(s_fc1w[rowh+c], s_fc1w[rowh+c+1], s_fc1w[rowh+c+2], s_fc1w[rowh+c+3]);
            acc += dot(xv, wv);
        }
        float a = (float)acc;
        float gelu = 0.5 * a * (1.0 + tanh(0.7978845608 * (a + 0.044715 * a * a * a)));
        hidden[h] = (float16_t)gelu;
    }

    [unroll] for (uint r = 0; r < DIM; ++r) {
        float16_t acc = s_fc2b[r];
        uint rowf = r * HIDDEN;
        [unroll] for (uint h = 0; h < HIDDEN; h += 4) {
            vector<float16_t,4> hv = vector<float16_t,4>(hidden[h], hidden[h+1], hidden[h+2], hidden[h+3]);
            vector<float16_t,4> wv = vector<float16_t,4>(s_fc2w[rowf+h], s_fc2w[rowf+h+1], s_fc2w[rowf+h+2], s_fc2w[rowf+h+3]);
            acc += dot(hv, wv);
        }
        float16_t outv = acc + post_attn[r];
        g_out[base + r] = outv;
    }
}