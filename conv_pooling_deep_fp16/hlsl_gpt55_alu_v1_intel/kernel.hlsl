Buffer<float16_t> input_0 : register(t0);
Buffer<float16_t> state_0 : register(t1);
Buffer<float16_t> state_1 : register(t2);
Buffer<float16_t> state_2 : register(t3);

RWBuffer<float16_t> output_0 : register(u0);

static const uint IN_C = 128;
static const uint OUT_C = 128;
static const uint IN_H = 360;
static const uint IN_W = 640;
static const uint OUT_H = 180;
static const uint OUT_W = 320;
static const uint TERMS = 1152;
static const uint INPUT_HW = 230400;
static const uint OUTPUT_HW = 57600;

static const uint TILE_OW = 16;
static const uint TILE_OH = 4;
static const uint TILE_IW = 34;
static const uint TILE_IH = 10;
static const uint IC_BLK = 16;
static const uint OC_BLK = 16;
static const uint TILE_IN_ELEMS = 5440;

groupshared float gsIn[5440];

[WaveSize(32)]
[numthreads(32, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    uint lx = groupThreadId.x & 7;
    uint ly = groupThreadId.x >> 3;
    uint oc = groupId.z * OC_BLK + groupThreadId.y;
    uint tid = groupThreadId.y * 32 + groupThreadId.x;

    uint ox0 = groupId.x * TILE_OW + lx;
    uint ox1 = ox0 + 8;
    uint oy = groupId.y * TILE_OH + ly;

    int tileOriginX = (int)(groupId.x * 32) - 1;
    int tileOriginY = (int)(groupId.y * 8) - 1;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc30 = 0.0f;
    float acc31 = 0.0f;

    uint localBase = ly * 2 * TILE_IW + lx * 2;
    uint wbase = oc * TERMS;

    [unroll]
    for (uint icBlock = 0; icBlock < IN_C; icBlock += IC_BLK)
    {
        for (uint loadIdx = tid; loadIdx < TILE_IN_ELEMS; loadIdx += 512)
        {
            uint ci = loadIdx / (TILE_IH * TILE_IW);
            uint rem = loadIdx - ci * (TILE_IH * TILE_IW);
            uint ty = rem / TILE_IW;
            uint tx = rem - ty * TILE_IW;

            int gy = tileOriginY + (int)ty;
            int gx = tileOriginX + (int)tx;

            float v = 0.0f;
            if ((uint)gy < IN_H && (uint)gx < IN_W)
            {
                uint gic = icBlock + ci;
                v = (float)input_0[gic * INPUT_HW + (uint)gy * IN_W + (uint)gx];
            }
            gsIn[loadIdx] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint ci2 = 0; ci2 < IC_BLK; ++ci2)
        {
            uint inChanBase = ci2 * (TILE_IH * TILE_IW) + localBase;
            uint weightBase = wbase + (icBlock + ci2) * 9;

            [unroll]
            for (uint ky = 0; ky < 3; ++ky)
            {
                uint rowOff = inChanBase + ky * TILE_IW;

                [unroll]
                for (uint kx = 0; kx < 3; ++kx)
                {
                    float wlane = 0.0f;
                    if (groupThreadId.x == 0)
                    {
                        wlane = (float)state_1[weightBase + ky * 3 + kx];
                    }
                    float w = WaveReadLaneFirst(wlane);

                    uint p = rowOff + kx;
                    uint q = p + 16;

                    acc00 += gsIn[p] * w;
                    acc01 += gsIn[p + 1] * w;
                    acc10 += gsIn[p + TILE_IW] * w;
                    acc11 += gsIn[p + TILE_IW + 1] * w;

                    acc20 += gsIn[q] * w;
                    acc21 += gsIn[q + 1] * w;
                    acc30 += gsIn[q + TILE_IW] * w;
                    acc31 += gsIn[q + TILE_IW + 1] * w;
                }
            }
        }

        if (icBlock != 112)
        {
            GroupMemoryBarrierWithGroupSync();
        }
    }

    float blane = 0.0f;
    if (groupThreadId.x == 0)
    {
        blane = (float)state_2[oc];
    }
    float b = WaveReadLaneFirst(blane);

    float v00 = max(acc00 + b, 0.0f);
    float v01 = max(acc01 + b, 0.0f);
    float v10 = max(acc10 + b, 0.0f);
    float v11 = max(acc11 + b, 0.0f);

    float v20 = max(acc20 + b, 0.0f);
    float v21 = max(acc21 + b, 0.0f);
    float v30 = max(acc30 + b, 0.0f);
    float v31 = max(acc31 + b, 0.0f);

    uint outBase = oc * OUTPUT_HW + oy * OUT_W + ox0;
    output_0[outBase] = (float16_t)((v00 + v01 + v10 + v11) * 0.25f);
    output_0[outBase + 8] = (float16_t)((v20 + v21 + v30 + v31) * 0.25f);
}