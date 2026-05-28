#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer input_0 : register(t0);
ByteAddressBuffer state_0 : register(t1);
ByteAddressBuffer state_1 : register(t2);
ByteAddressBuffer state_2 : register(t3);

RWByteAddressBuffer output_0 : register(u0);

static const uint IN_C = 128;
static const uint OUT_C = 128;
static const uint IN_H = 360;
static const uint IN_W = 640;
static const uint OUT_H = 180;
static const uint OUT_W = 320;
static const uint K_TOTAL = 1152;
static const uint K_TILE = 64;
static const uint OC_TILE = 16;
static const uint OC_PAIR_TILE = 32;
static const uint OC_PAIR_BLOCKS = 4;
static const uint SPATIAL = OUT_H * OUT_W;

typedef Matrix<ComponentType::F16, 16, 64, MatrixUse::A, MatrixScope::Thread> WFrag16x64;

float16_t LoadInputHalf(uint c, uint y, uint x)
{
    uint elem = (c * IN_H + y) * IN_W + x;
    return input_0.Load<float16_t>(elem * 2);
}

vector<float16_t, 16> LoadBiasVec16(uint ocBase)
{
    return state_2.Load<vector<float16_t, 16> >(ocBase * 2);
}

float16_t InputForKNoBounds(uint kk, uint cy, uint cx)
{
    uint ic = kk / 9;
    uint rem = kk - ic * 9;
    uint kh = rem / 3;
    uint kw = rem - kh * 3;

    uint iy = cy + kh - 1;
    uint ix = cx + kw - 1;

    return LoadInputHalf(ic, iy, ix);
}

float16_t InputForK(uint kk, int cy, int cx)
{
    uint ic = kk / 9;
    uint rem = kk - ic * 9;
    uint kh = rem / 3;
    uint kw = rem - kh * 3;

    int iy = cy + (int)kh - 1;
    int ix = cx + (int)kw - 1;

    if (iy < 0 || iy >= (int)IN_H || ix < 0 || ix >= (int)IN_W)
    {
        return (float16_t)0.0h;
    }

    return LoadInputHalf(ic, (uint)iy, (uint)ix);
}

[WaveSize(32)]
[numthreads(32, 8, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint ox = groupID.x * 32 + groupThreadID.x;
    uint oy = groupID.y * 8 + groupThreadID.y;

    if (oy >= OUT_H)
    {
        return;
    }

    uint ocPairBlock = groupID.z;
    uint ocBase0 = ocPairBlock * OC_PAIR_TILE;
    uint ocBase1 = ocBase0 + OC_TILE;

    bool interior = (oy > 0 && oy < (OUT_H - 1) && ox > 0 && ox < (OUT_W - 1));

    vector<float16_t, 16> bias160 = LoadBiasVec16(ocBase0);
    vector<float16_t, 16> bias161 = LoadBiasVec16(ocBase1);

    vector<float, 16> poolAcc0;
    vector<float, 16> poolAcc1;
    [unroll]
    for (uint oi = 0; oi < 16; ++oi)
    {
        poolAcc0[oi] = 0.0f;
        poolAcc1[oi] = 0.0f;
    }

    [unroll]
    for (uint py = 0; py < 2; ++py)
    {
        [unroll]
        for (uint px = 0; px < 2; ++px)
        {
            uint ucy = oy * 2 + py;
            uint ucx = ox * 2 + px;
            int cy = (int)ucy;
            int cx = (int)ucx;

            vector<float, 16> acc0;
            vector<float, 16> acc1;
            [unroll]
            for (uint oi = 0; oi < 16; ++oi)
            {
                acc0[oi] = (float)bias160[oi];
                acc1[oi] = (float)bias161[oi];
            }

            [unroll]
            for (uint k0 = 0; k0 < K_TOTAL; k0 += K_TILE)
            {
                vector<float16_t, 64> xFrag;

                [branch]
                if (interior)
                {
                    [unroll]
                    for (uint ki = 0; ki < K_TILE; ++ki)
                    {
                        xFrag[ki] = InputForKNoBounds(k0 + ki, ucy, ucx);
                    }
                }
                else
                {
                    [unroll]
                    for (uint ki = 0; ki < K_TILE; ++ki)
                    {
                        xFrag[ki] = InputForK(k0 + ki, cy, cx);
                    }
                }

                uint weightByteOffset0 = ((ocBase0 * K_TOTAL) + k0) * 2;
                WFrag16x64 wFrag0 = WFrag16x64::Load<MatrixLayout::RowMajor>(
                    state_1,
                    weightByteOffset0,
                    K_TOTAL * 2);

                vector<float16_t, 16> delta160 = Multiply<float16_t>(wFrag0, xFrag);

                [unroll]
                for (uint oi = 0; oi < 16; ++oi)
                {
                    acc0[oi] += (float)delta160[oi];
                }

                uint weightByteOffset1 = ((ocBase1 * K_TOTAL) + k0) * 2;
                WFrag16x64 wFrag1 = WFrag16x64::Load<MatrixLayout::RowMajor>(
                    state_1,
                    weightByteOffset1,
                    K_TOTAL * 2);

                vector<float16_t, 16> delta161 = Multiply<float16_t>(wFrag1, xFrag);

                [unroll]
                for (uint oi = 0; oi < 16; ++oi)
                {
                    acc1[oi] += (float)delta161[oi];
                }
            }

            [unroll]
            for (uint oi = 0; oi < 16; ++oi)
            {
                poolAcc0[oi] += max(acc0[oi], 0.0f);
                poolAcc1[oi] += max(acc1[oi], 0.0f);
            }
        }
    }

    [unroll]
    for (uint oi = 0; oi < 16; ++oi)
    {
        uint oc0 = ocBase0 + oi;
        uint outElem0 = (oc0 * OUT_H + oy) * OUT_W + ox;
        float v0 = poolAcc0[oi] * 0.25f;
        output_0.Store<float16_t>(outElem0 * 2, (float16_t)v0);

        uint oc1 = ocBase1 + oi;
        uint outElem1 = (oc1 * OUT_H + oy) * OUT_W + ox;
        float v1 = poolAcc1[oi] * 0.25f;
        output_0.Store<float16_t>(outElem1 * 2, (float16_t)v1);
    }
}