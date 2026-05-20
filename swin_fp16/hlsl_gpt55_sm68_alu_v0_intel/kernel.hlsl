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

static const int H_PIX = 720;
static const int W_PIX = 1280;
static const int C_DIM = 32;
static const int WH = 4;
static const int WW = 4;
static const int TOKENS = 16;
static const int NUM_WIN_W = 320;
static const int NUM_WINDOWS = 57600;
static const int ROW_STRIDE = 40960;
static const float LN_EPS = 0.00001f;
static const float ATTN_SCALE = 0.1767766952966369f;

float gelu_approx(float x)
{
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

[WaveSize(32)]
[numthreads(128, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    uint lane = group_thread_id.x & 31;
    uint wave = group_thread_id.x >> 5;

    uint wx = (group_id.x << 2) + wave;
    uint win_base = group_id.y * 163840u + wx * 128u;

    uint row32 = lane << 5;
    uint qrow = row32;
    uint krow = 1024 + row32;
    uint vrow = 2048 + row32;
    uint fc2row = lane << 7;

    float norm1_w = (float)state_1[lane];
    float norm1_b = (float)state_2[lane];
    float q_bias = (float)state_4[lane];
    float k_bias = (float)state_4[32 + lane];
    float v_bias = (float)state_4[64 + lane];
    float proj_bias = (float)state_6[lane];
    float norm2_w = (float)state_7[lane];
    float norm2_b = (float)state_8[lane];
    float fc2_bias = (float)state_12[lane];

    float x0[16];
    float qv[16];
    float kv[16];
    float vv[16];
    float attv[16];
    float x1[16];

    [unroll]
    for (int t = 0; t < 16; ++t)
    {
        uint base = win_base + (uint)(t >> 2) * ROW_STRIDE + (uint)(t & 3) * C_DIM;
        float x = (float)input_0[base + lane];
        x0[t] = x;

        float sum_x = WaveActiveSum(x);
        float sum_x2 = WaveActiveSum(x * x);
        float mean = sum_x * 0.03125f;
        float var = max(sum_x2 * 0.03125f - mean * mean, 0.0f);
        float n = (x - mean) * rsqrt(var + LN_EPS);
        n = n * norm1_w + norm1_b;

        float aq = q_bias;
        float ak = k_bias;
        float av = v_bias;

        [unroll]
        for (uint i = 0; i < 32; ++i)
        {
            float ni = WaveReadLaneAt(n, i);
            aq += ni * (float)state_3[qrow + i];
            ak += ni * (float)state_3[krow + i];
            av += ni * (float)state_3[vrow + i];
        }

        qv[t] = aq;
        kv[t] = ak;
        vv[t] = av;
    }

    [unroll]
    for (int t = 0; t < 16; ++t)
    {
        float logits[16];
        float m = -3.402823466e+38f;

        [unroll]
        for (int j = 0; j < 16; ++j)
        {
            float dot_part = (qv[t] * ATTN_SCALE) * kv[j];
            float score = WaveActiveSum(dot_part) + (float)state_0[t * 16 + j];
            logits[j] = score;
            m = max(m, score);
        }

        float denom = 0.0f;
        float acc = 0.0f;

        [unroll]
        for (int j = 0; j < 16; ++j)
        {
            float e = exp(logits[j] - m);
            denom += e;
            acc += e * vv[j];
        }

        attv[t] = acc / denom;
    }

    [unroll]
    for (int t = 0; t < 16; ++t)
    {
        float p = proj_bias;

        [unroll]
        for (uint i = 0; i < 32; ++i)
        {
            float ai = WaveReadLaneAt(attv[t], i);
            p += ai * (float)state_5[row32 + i];
        }

        x1[t] = x0[t] + p;
    }

    [unroll]
    for (int t = 0; t < 16; ++t)
    {
        float x = x1[t];

        float sum_x = WaveActiveSum(x);
        float sum_x2 = WaveActiveSum(x * x);
        float mean = sum_x * 0.03125f;
        float var = max(sum_x2 * 0.03125f - mean * mean, 0.0f);
        float n = (x - mean) * rsqrt(var + LN_EPS);
        qv[t] = n * norm2_w + norm2_b;
        kv[t] = fc2_bias;
    }

    [loop]
    for (uint h = 0; h < 128; ++h)
    {
        float w9 = (float)state_9[h * 32 + lane];
        float b10 = (float)state_10[h];
        float w11 = (float)state_11[fc2row + h];

        [unroll]
        for (int t = 0; t < 16; ++t)
        {
            float fc1 = WaveActiveSum(qv[t] * w9) + b10;
            float g = gelu_approx(fc1);
            kv[t] += g * w11;
        }
    }

    [unroll]
    for (int t = 0; t < 16; ++t)
    {
        float y = x1[t] + kv[t];
        uint base = win_base + (uint)(t >> 2) * ROW_STRIDE + (uint)(t & 3) * C_DIM;
        output_0[base + lane] = (float16_t)y;
    }
}