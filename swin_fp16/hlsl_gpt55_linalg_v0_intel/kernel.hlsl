#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer input_0 : register(t0);
ByteAddressBuffer state_0 : register(t1);
ByteAddressBuffer state_1 : register(t2);
ByteAddressBuffer state_2 : register(t3);
ByteAddressBuffer state_3 : register(t4);
ByteAddressBuffer state_4 : register(t5);
ByteAddressBuffer state_5 : register(t6);
ByteAddressBuffer state_6 : register(t7);
ByteAddressBuffer state_7 : register(t8);
ByteAddressBuffer state_8 : register(t9);
ByteAddressBuffer state_9 : register(t10);
ByteAddressBuffer state_10 : register(t11);
ByteAddressBuffer state_11 : register(t12);
ByteAddressBuffer state_12 : register(t13);

RWByteAddressBuffer output_0 : register(u0);

#define LOAD_F16(buf, elem) f16tof32(((buf.Load<uint>((((elem) * 2u) & 0xffffffFCu)) >> (((((elem) * 2u) & 2u) == 0u) ? 0u : 16u)) & 0xffffu))
#define LOAD_H16(buf, elem) ((float16_t)LOAD_F16(buf, elem))
#define PACK_F16X2(a, b) ((f32tof16((float)(a)) & 0xffffu) | ((f32tof16((float)(b)) & 0xffffu) << 16u))

static const uint H_PIX = 720u;
static const uint W_PIX = 1280u;
static const uint C_DIM = 32u;
static const uint WIN = 4u;
static const uint N_TOK = 16u;
static const uint NH_WIN = 180u;
static const uint NW_WIN = 320u;
static const uint NUM_WIN = 57600u;

using W16x32 = Matrix<ComponentType::F16, 16, 32, MatrixUse::A, MatrixScope::Thread>;
using W16x64 = Matrix<ComponentType::F16, 16, 64, MatrixUse::A, MatrixScope::Thread>;

float gelu_tanh(float x)
{
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(k0 * (x + k1 * x3)));
}

[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 tid3 : SV_GroupThreadID)
{
    uint lane = WaveGetLaneIndex();
    if (lane >= 16u)
    {
        return;
    }

    uint winId = groupID.x;

    uint winY = winId / NW_WIN;
    uint winX = winId - winY * NW_WIN;

    uint tokR = lane >> 2;
    uint tokC = lane & 3u;
    uint pixY = winY * 4u + tokR;
    uint pixX = winX * 4u + tokC;
    uint baseElem = ((pixY * W_PIX + pixX) * C_DIM);

    float16_t xh[32];
    float sum0 = 0.0f;

    [unroll]
    for (uint c = 0u; c < 32u; ++c)
    {
        float16_t xv = LOAD_H16(input_0, baseElem + c);
        xh[c] = xv;
        sum0 += (float)xv;
    }

    float mean0 = sum0 * (1.0f / 32.0f);
    float var0 = 0.0f;
    [unroll]
    for (uint c0 = 0u; c0 < 32u; ++c0)
    {
        float d = (float)xh[c0] - mean0;
        var0 += d * d;
    }
    var0 *= (1.0f / 32.0f);
    float inv0 = rsqrt(var0 + 1.0e-5f);

    vector<float16_t, 32> n1vec;
    [unroll]
    for (uint c1 = 0u; c1 < 32u; ++c1)
    {
        float y = ((float)xh[c1] - mean0) * inv0;
        y = y * LOAD_F16(state_1, c1) + LOAD_F16(state_2, c1);
        n1vec[c1] = (float16_t)y;
    }

    float16_t qh[32];
    float16_t kh[32];
    float16_t vh[32];

    [unroll]
    for (uint qb = 0u; qb < 6u; ++qb)
    {
        uint outBase = qb * 16u;
        W16x32 wqkv = W16x32::Load<MatrixLayout::RowMajor>(state_3, outBase * 32u * 2u, 32u * 2u);
        vector<float16_t, 16> d16 = Multiply<float16_t>(wqkv, n1vec);
        [unroll]
        for (uint oi = 0u; oi < 16u; ++oi)
        {
            uint oc = outBase + oi;
            float16_t lv = (float16_t)((float)d16[oi] + LOAD_F16(state_4, oc));
            if (oc < 32u)
            {
                qh[oc] = lv;
            }
            else if (oc < 64u)
            {
                kh[oc - 32u] = lv;
            }
            else
            {
                vh[oc - 64u] = lv;
            }
        }
    }

    float attn[16];
    float rowMax = -3.402823466e+38f;
    [unroll]
    for (uint j = 0u; j < 16u; ++j)
    {
        float dot = 0.0f;
        [unroll]
        for (uint c2 = 0u; c2 < 32u; ++c2)
        {
            float qv = (float)qh[c2];
            float kv = WaveReadLaneAt((float)kh[c2], j);
            dot += qv * kv;
        }
        float s = dot * 0.1767766952966369f + LOAD_F16(state_0, lane * 16u + j);
        attn[j] = s;
        rowMax = max(rowMax, s);
    }

    float rowSum = 0.0f;
    [unroll]
    for (uint j0 = 0u; j0 < 16u; ++j0)
    {
        float e = exp(attn[j0] - rowMax);
        attn[j0] = e;
        rowSum += e;
    }
    float invRow = 1.0f / rowSum;

    vector<float16_t, 32> ctxvec;
    [unroll]
    for (uint c3 = 0u; c3 < 32u; ++c3)
    {
        float cv = 0.0f;
        [unroll]
        for (uint j1 = 0u; j1 < 16u; ++j1)
        {
            float av = attn[j1] * invRow;
            float vv = WaveReadLaneAt((float)vh[c3], j1);
            cv += av * vv;
        }
        ctxvec[c3] = (float16_t)cv;
    }

    float16_t projh[32];
    [unroll]
    for (uint pb = 0u; pb < 2u; ++pb)
    {
        uint outBaseP = pb * 16u;
        W16x32 wp = W16x32::Load<MatrixLayout::RowMajor>(state_5, outBaseP * 32u * 2u, 32u * 2u);
        vector<float16_t, 16> pd16 = Multiply<float16_t>(wp, ctxvec);
        [unroll]
        for (uint po = 0u; po < 16u; ++po)
        {
            uint ocp = outBaseP + po;
            projh[ocp] = (float16_t)((float)pd16[po] + LOAD_F16(state_6, ocp));
        }
    }

    float16_t res1h[32];
    float sum1 = 0.0f;
    [unroll]
    for (uint c4 = 0u; c4 < 32u; ++c4)
    {
        float16_t rv = (float16_t)((float)xh[c4] + (float)projh[c4]);
        res1h[c4] = rv;
        sum1 += (float)rv;
    }

    float mean1 = sum1 * (1.0f / 32.0f);
    float var1 = 0.0f;
    [unroll]
    for (uint c5 = 0u; c5 < 32u; ++c5)
    {
        float d1 = (float)res1h[c5] - mean1;
        var1 += d1 * d1;
    }
    var1 *= (1.0f / 32.0f);
    float inv1 = rsqrt(var1 + 1.0e-5f);

    vector<float16_t, 32> n2vec;
    [unroll]
    for (uint c6 = 0u; c6 < 32u; ++c6)
    {
        float y2 = ((float)res1h[c6] - mean1) * inv1;
        y2 = y2 * LOAD_F16(state_7, c6) + LOAD_F16(state_8, c6);
        n2vec[c6] = (float16_t)y2;
    }

    float16_t acth[128];

    [unroll]
    for (uint fb = 0u; fb < 8u; ++fb)
    {
        uint outBaseF = fb * 16u;
        W16x32 wf1 = W16x32::Load<MatrixLayout::RowMajor>(state_9, outBaseF * 32u * 2u, 32u * 2u);
        vector<float16_t, 16> fd16 = Multiply<float16_t>(wf1, n2vec);
        [unroll]
        for (uint fo = 0u; fo < 16u; ++fo)
        {
            uint ocf = outBaseF + fo;
            float fv = (float)fd16[fo] + LOAD_F16(state_10, ocf);
            float16_t ah = (float16_t)gelu_tanh((float)(float16_t)fv);
            acth[ocf] = ah;
        }
    }

    vector<float16_t, 64> act0;
    vector<float16_t, 64> act1;
    [unroll]
    for (uint a0 = 0u; a0 < 64u; ++a0)
    {
        act0[a0] = acth[a0];
        act1[a0] = acth[a0 + 64u];
    }

    float16_t fc2h[32];
    [unroll]
    for (uint ob = 0u; ob < 2u; ++ob)
    {
        uint outBase2 = ob * 16u;
        W16x64 w20 = W16x64::Load<MatrixLayout::RowMajor>(state_11, outBase2 * 128u * 2u, 128u * 2u);
        W16x64 w21 = W16x64::Load<MatrixLayout::RowMajor>(state_11, (outBase2 * 128u + 64u) * 2u, 128u * 2u);
        vector<float16_t, 16> d20 = Multiply<float16_t>(w20, act0);
        vector<float16_t, 16> d21 = Multiply<float16_t>(w21, act1);
        [unroll]
        for (uint oo = 0u; oo < 16u; ++oo)
        {
            uint occ = outBase2 + oo;
            float acc2 = (float)d20[oo] + (float)d21[oo] + LOAD_F16(state_12, occ);
            fc2h[occ] = (float16_t)acc2;
        }
    }

    [unroll]
    for (uint p = 0u; p < 16u; ++p)
    {
        uint cA = p * 2u;
        float16_t o0 = (float16_t)((float)res1h[cA] + (float)fc2h[cA]);
        float16_t o1 = (float16_t)((float)res1h[cA + 1u] + (float)fc2h[cA + 1u]);
        output_0.Store((baseElem + cA) * 2u, PACK_F16X2(o0, o1));
    }
}