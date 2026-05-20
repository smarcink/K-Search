Buffer<float16_t> input_0 : register(t0);
Buffer<float16_t> state_0 : register(t1);
Buffer<float16_t> state_1 : register(t2);
Buffer<float16_t> state_2 : register(t3);
Buffer<float16_t> state_3 : register(t4);
Buffer<float16_t> state_4 : register(t5);
Buffer<float16_t> state_5 : register(t6);
Buffer<float16_t> state_6 : register(t7);
Buffer<float16_t> state_7 : register(t8);
Buffer<float16_t> state_8 : register(t9);
Buffer<float16_t> state_9 : register(t10);
Buffer<float16_t> state_10 : register(t11);
Buffer<float16_t> state_11 : register(t12);
Buffer<float16_t> state_12 : register(t13);

RWBuffer<float16_t> output_0 : register(u0);

static const uint W = 1280;
static const uint C = 32;
static const float LN_EPS = 1.0e-5f;
static const float ATTN_SCALE = 0.1767766952966369f;

groupshared float16_t g_ln[512];
groupshared float16_t g_q[512];
groupshared float16_t g_k[512];
groupshared float16_t g_v[512];
groupshared float16_t g_attn[256];
groupshared float16_t g_tmp[2048];

groupshared float16_t g_wqkv[3072];
groupshared float16_t g_wfc1[4096];
groupshared float16_t g_wfc2[4096];

float gelu_fast(float x)
{
    float x2 = x * x;
    float u = 0.7978845608028654f * (x + 0.044715f * x * x2);
    return 0.5f * x * (1.0f + tanh(u));
}

[WaveSize(32)]
[numthreads(512, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    uint tid = group_thread_id.x;

    for (uint wi = tid; wi < 11264; wi += 512)
    {
        if (wi < 3072)
        {
            g_wqkv[wi] = state_3[wi];
        }
        else if (wi < 7168)
        {
            uint j = wi - 3072;
            g_wfc1[j] = state_9[j];
        }
        else
        {
            uint j = wi - 7168;
            g_wfc2[j] = state_11[j];
        }
    }

    uint wx = group_id.x;
    uint wy = group_id.y;

    uint token = tid >> 5;
    uint lane = tid & 31;

    uint ly = token >> 2;
    uint lx = token & 3;
    uint y = wy * 4 + ly;
    uint x = wx * 4 + lx;
    uint in_idx = ((y * W + x) * C + lane);

    float inv = float(input_0[in_idx]);

    float sum1 = WaveActiveSum(inv);
    float mean1 = sum1 * (1.0f / 32.0f);
    float d = inv - mean1;
    float varsum1 = WaveActiveSum(d * d);
    float rstd1 = rsqrt(varsum1 * (1.0f / 32.0f) + LN_EPS);

    g_ln[tid] = float16_t(d * rstd1 * float(state_1[lane]) + float(state_2[lane]));

    GroupMemoryBarrierWithGroupSync();

    for (uint idx = tid; idx < 1536; idx += 512)
    {
        uint tok = idx / 96;
        uint oc = idx - tok * 96;
        float acc = float(state_4[oc]);
        uint base_x = tok * 32;
        uint base_w = oc * 32;
        [unroll]
        for (uint ic = 0; ic < 32; ++ic)
        {
            acc += float(g_ln[base_x + ic]) * float(g_wqkv[base_w + ic]);
        }

        if (oc < 32)
        {
            g_q[tok * 32 + oc] = float16_t(acc);
        }
        else if (oc < 64)
        {
            g_k[tok * 32 + (oc - 32)] = float16_t(acc);
        }
        else
        {
            g_v[tok * 32 + (oc - 64)] = float16_t(acc);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint qi = tid >> 5;
    uint kj = tid & 31;
    float logit = -3.402823466e+38f;
    if (kj < 16)
    {
        float dot = 0.0f;
        uint qb = qi * 32;
        uint kb = kj * 32;
        [unroll]
        for (uint ic0 = 0; ic0 < 32; ++ic0)
        {
            dot += float(g_q[qb + ic0]) * float(g_k[kb + ic0]);
        }
        logit = dot * ATTN_SCALE + float(state_0[qi * 16 + kj]);
    }

    float row_max = WaveActiveMax(logit);
    float e = (kj < 16) ? exp(logit - row_max) : 0.0f;
    float row_sum = WaveActiveSum(e);
    if (kj < 16)
    {
        g_attn[qi * 16 + kj] = float16_t(e * rcp(row_sum));
    }

    GroupMemoryBarrierWithGroupSync();

    uint tok_av = tid >> 5;
    uint ch_av = tid & 31;
    float acc_av = 0.0f;
    [unroll]
    for (uint j = 0; j < 16; ++j)
    {
        acc_av += float(g_attn[tok_av * 16 + j]) * float(g_v[j * 32 + ch_av]);
    }
    g_tmp[tid] = float16_t(acc_av);

    for (uint pi = tid; pi < 1024; pi += 512)
    {
        if (pi < 512)
        {
            g_q[pi] = state_5[pi];
        }
        else
        {
            g_k[pi - 512] = state_5[pi];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint tok_p = tid >> 5;
    uint oc_p = tid & 31;
    float acc_p = float(state_6[oc_p]);
    uint base_x_p = tok_p * 32;
    if (oc_p < 16)
    {
        uint base_w_p0 = oc_p * 32;
        [unroll]
        for (uint icp0 = 0; icp0 < 32; ++icp0)
        {
            acc_p += float(g_tmp[base_x_p + icp0]) * float(g_q[base_w_p0 + icp0]);
        }
    }
    else
    {
        uint base_w_p1 = (oc_p - 16) * 32;
        [unroll]
        for (uint icp1 = 0; icp1 < 32; ++icp1)
        {
            acc_p += float(g_tmp[base_x_p + icp1]) * float(g_k[base_w_p1 + icp1]);
        }
    }

    float res_after_attn = inv + acc_p;

    float sum2 = WaveActiveSum(res_after_attn);
    float mean2 = sum2 * (1.0f / 32.0f);
    float rd = res_after_attn - mean2;
    float varsum2 = WaveActiveSum(rd * rd);
    float rstd2 = rsqrt(varsum2 * (1.0f / 32.0f) + LN_EPS);

    g_ln[tid] = float16_t(rd * rstd2 * float(state_7[lane]) + float(state_8[lane]));

    GroupMemoryBarrierWithGroupSync();

    for (uint idx1 = tid; idx1 < 2048; idx1 += 512)
    {
        uint tok1 = idx1 >> 7;
        uint oc1 = idx1 & 127;
        float acc1 = float(state_10[oc1]);
        uint base_x1 = tok1 * 32;
        uint base_w1 = oc1 * 32;
        [unroll]
        for (uint ic1 = 0; ic1 < 32; ++ic1)
        {
            acc1 += float(g_ln[base_x1 + ic1]) * float(g_wfc1[base_w1 + ic1]);
        }
        g_tmp[idx1] = float16_t(gelu_fast(acc1));
    }

    GroupMemoryBarrierWithGroupSync();

    uint tok2 = tid >> 5;
    uint oc2 = tid & 31;
    float acc2 = float(state_12[oc2]);
    uint base_x2 = tok2 * 128;
    uint base_w2 = oc2 * 128;
    [unroll]
    for (uint ic2 = 0; ic2 < 128; ++ic2)
    {
        acc2 += float(g_tmp[base_x2 + ic2]) * float(g_wfc2[base_w2 + ic2]);
    }

    float outv = res_after_attn + acc2;
    output_0[in_idx] = float16_t(outv);
}