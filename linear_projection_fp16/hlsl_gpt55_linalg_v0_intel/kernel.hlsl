ByteAddressBuffer input_0 : register(t0);
ByteAddressBuffer state_0 : register(t1);
ByteAddressBuffer state_1 : register(t2);
RWByteAddressBuffer output_0 : register(u0);

#define BATCH 32768u
#define IN_FEATURES 256u
#define OUT_FEATURES 256u
#define TILE_ROWS 64u
#define TILE_COL_PAIRS 16u
#define TILE_K 32u
#define SHX_WORDS 1024u
#define SHW_WORDS 512u

groupshared uint shX[2048];
groupshared uint shW[1024];

void LoadKTile(uint buf, uint k0, uint tx, uint ty,
               uint rowBase0, uint rowBase1, uint rowBase2, uint rowBase3,
               uint tileColBase)
{
    uint xOff = buf * SHX_WORDS;
    uint wOff = buf * SHW_WORDS;

    uint kx = k0 + tx * 4u;
    uint sx = tx * 2u;

    uint2 xWord = input_0.Load2((rowBase0 + kx) << 1);
    shX[xOff + ty * 16u + sx] = xWord.x;
    shX[xOff + ty * 16u + sx + 1u] = xWord.y;

    xWord = input_0.Load2((rowBase1 + kx) << 1);
    shX[xOff + (ty + 16u) * 16u + sx] = xWord.x;
    shX[xOff + (ty + 16u) * 16u + sx + 1u] = xWord.y;

    xWord = input_0.Load2((rowBase2 + kx) << 1);
    shX[xOff + (ty + 32u) * 16u + sx] = xWord.x;
    shX[xOff + (ty + 32u) * 16u + sx + 1u] = xWord.y;

    xWord = input_0.Load2((rowBase3 + kx) << 1);
    shX[xOff + (ty + 48u) * 16u + sx] = xWord.x;
    shX[xOff + (ty + 48u) * 16u + sx + 1u] = xWord.y;

    uint kEven = k0 + ty * 2u;

    uint p0 = tx;
    uint colA0 = tileColBase + p0 * 2u;
    uint wBaseA0 = colA0 << 8;
    uint wBaseA1 = wBaseA0 + 256u;
    uint wWordA0 = state_0.Load((wBaseA0 + kEven) << 1);
    uint wWordA1 = state_0.Load((wBaseA1 + kEven) << 1);
    shW[wOff + (ty * 2u) * TILE_COL_PAIRS + p0] = (wWordA0 & 0xFFFFu) | ((wWordA1 & 0xFFFFu) << 16);
    shW[wOff + (ty * 2u + 1u) * TILE_COL_PAIRS + p0] = (wWordA0 >> 16) | (wWordA1 & 0xFFFF0000u);

    uint p1 = tx + 8u;
    uint colB0 = tileColBase + p1 * 2u;
    uint wBaseB0 = colB0 << 8;
    uint wBaseB1 = wBaseB0 + 256u;
    uint wWordB0 = state_0.Load((wBaseB0 + kEven) << 1);
    uint wWordB1 = state_0.Load((wBaseB1 + kEven) << 1);
    shW[wOff + (ty * 2u) * TILE_COL_PAIRS + p1] = (wWordB0 & 0xFFFFu) | ((wWordB1 & 0xFFFFu) << 16);
    shW[wOff + (ty * 2u + 1u) * TILE_COL_PAIRS + p1] = (wWordB0 >> 16) | (wWordB1 & 0xFFFF0000u);
}

[numthreads(8, 16, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint tx = groupThreadID.x;
    uint ty = groupThreadID.y;

    uint row0 = groupID.x * TILE_ROWS + ty;
    uint row1 = row0 + 16u;
    uint row2 = row0 + 32u;
    uint row3 = row0 + 48u;

    uint rowBase0 = row0 << 8;
    uint rowBase1 = row1 << 8;
    uint rowBase2 = row2 << 8;
    uint rowBase3 = row3 << 8;

    uint tileColBase = groupID.y * (TILE_COL_PAIRS * 2u);
    uint col0 = tileColBase + tx * 4u;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;

    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;

    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;

    uint buf = 0u;
    LoadKTile(buf, 0u, tx, ty, rowBase0, rowBase1, rowBase2, rowBase3, tileColBase);
    GroupMemoryBarrierWithGroupSync();

    [unroll(8)]
    for (uint k0 = 0u; k0 < IN_FEATURES; k0 += TILE_K)
    {
        uint xOff = buf * SHX_WORDS;
        uint wOff = buf * SHW_WORDS;

        uint sxBase0 = xOff + ty * 16u;
        uint sxBase1 = sxBase0 + 256u;
        uint sxBase2 = sxBase0 + 512u;
        uint sxBase3 = sxBase0 + 768u;

        uint p0 = tx * 2u;
        uint p1 = p0 + 1u;

        [unroll]
        for (uint sx = 0u; sx < 16u; ++sx)
        {
            uint xWord0 = shX[sxBase0 + sx];
            uint xWord1 = shX[sxBase1 + sx];
            uint xWord2 = shX[sxBase2 + sx];
            uint xWord3 = shX[sxBase3 + sx];

            uint wWord01 = shW[wOff + (sx * 2u) * TILE_COL_PAIRS + p0];
            uint wWord23 = shW[wOff + (sx * 2u) * TILE_COL_PAIRS + p1];

            float wf0 = f16tof32(wWord01 & 0xFFFFu);
            float wf1 = f16tof32(wWord01 >> 16);
            float wf2 = f16tof32(wWord23 & 0xFFFFu);
            float wf3 = f16tof32(wWord23 >> 16);

            float x0 = f16tof32(xWord0 & 0xFFFFu);
            float x1 = f16tof32(xWord1 & 0xFFFFu);
            float x2 = f16tof32(xWord2 & 0xFFFFu);
            float x3 = f16tof32(xWord3 & 0xFFFFu);

            acc00 = mad(x0, wf0, acc00);
            acc01 = mad(x0, wf1, acc01);
            acc02 = mad(x0, wf2, acc02);
            acc03 = mad(x0, wf3, acc03);

            acc10 = mad(x1, wf0, acc10);
            acc11 = mad(x1, wf1, acc11);
            acc12 = mad(x1, wf2, acc12);
            acc13 = mad(x1, wf3, acc13);

            acc20 = mad(x2, wf0, acc20);
            acc21 = mad(x2, wf1, acc21);
            acc22 = mad(x2, wf2, acc22);
            acc23 = mad(x2, wf3, acc23);

            acc30 = mad(x3, wf0, acc30);
            acc31 = mad(x3, wf1, acc31);
            acc32 = mad(x3, wf2, acc32);
            acc33 = mad(x3, wf3, acc33);

            wWord01 = shW[wOff + (sx * 2u + 1u) * TILE_COL_PAIRS + p0];
            wWord23 = shW[wOff + (sx * 2u + 1u) * TILE_COL_PAIRS + p1];

            wf0 = f16tof32(wWord01 & 0xFFFFu);
            wf1 = f16tof32(wWord01 >> 16);
            wf2 = f16tof32(wWord23 & 0xFFFFu);
            wf3 = f16tof32(wWord23 >> 16);

            x0 = f16tof32(xWord0 >> 16);
            x1 = f16tof32(xWord1 >> 16);
            x2 = f16tof32(xWord2 >> 16);
            x3 = f16tof32(xWord3 >> 16);

            acc00 = mad(x0, wf0, acc00);
            acc01 = mad(x0, wf1, acc01);
            acc02 = mad(x0, wf2, acc02);
            acc03 = mad(x0, wf3, acc03);

            acc10 = mad(x1, wf0, acc10);
            acc11 = mad(x1, wf1, acc11);
            acc12 = mad(x1, wf2, acc12);
            acc13 = mad(x1, wf3, acc13);

            acc20 = mad(x2, wf0, acc20);
            acc21 = mad(x2, wf1, acc21);
            acc22 = mad(x2, wf2, acc22);
            acc23 = mad(x2, wf3, acc23);

            acc30 = mad(x3, wf0, acc30);
            acc31 = mad(x3, wf1, acc31);
            acc32 = mad(x3, wf2, acc32);
            acc33 = mad(x3, wf3, acc33);
        }

        uint nextK = k0 + TILE_K;
        if (nextK < IN_FEATURES)
        {
            uint nextBuf = buf ^ 1u;
            LoadKTile(nextBuf, nextK, tx, ty, rowBase0, rowBase1, rowBase2, rowBase3, tileColBase);
            GroupMemoryBarrierWithGroupSync();
            buf = nextBuf;
        }
    }

    uint2 bWords = state_1.Load2(col0 << 1);
    uint bWord01 = bWords.x;
    uint bWord23 = bWords.y;

    float b0 = f16tof32(bWord01 & 0xFFFFu);
    float b1 = f16tof32(bWord01 >> 16);
    float b2 = f16tof32(bWord23 & 0xFFFFu);
    float b3 = f16tof32(bWord23 >> 16);

    acc00 += b0;
    acc01 += b1;
    acc02 += b2;
    acc03 += b3;

    acc10 += b0;
    acc11 += b1;
    acc12 += b2;
    acc13 += b3;

    acc20 += b0;
    acc21 += b1;
    acc22 += b2;
    acc23 += b3;

    acc30 += b0;
    acc31 += b1;
    acc32 += b2;
    acc33 += b3;

    uint out01 = (f32tof16(acc00) & 0xFFFFu) | ((f32tof16(acc01) & 0xFFFFu) << 16);
    uint out23 = (f32tof16(acc02) & 0xFFFFu) | ((f32tof16(acc03) & 0xFFFFu) << 16);
    output_0.Store2((rowBase0 + col0) << 1, uint2(out01, out23));

    out01 = (f32tof16(acc10) & 0xFFFFu) | ((f32tof16(acc11) & 0xFFFFu) << 16);
    out23 = (f32tof16(acc12) & 0xFFFFu) | ((f32tof16(acc13) & 0xFFFFu) << 16);
    output_0.Store2((rowBase1 + col0) << 1, uint2(out01, out23));

    out01 = (f32tof16(acc20) & 0xFFFFu) | ((f32tof16(acc21) & 0xFFFFu) << 16);
    out23 = (f32tof16(acc22) & 0xFFFFu) | ((f32tof16(acc23) & 0xFFFFu) << 16);
    output_0.Store2((rowBase2 + col0) << 1, uint2(out01, out23));

    out01 = (f32tof16(acc30) & 0xFFFFu) | ((f32tof16(acc31) & 0xFFFFu) << 16);
    out23 = (f32tof16(acc32) & 0xFFFFu) | ((f32tof16(acc33) & 0xFFFFu) << 16);
    output_0.Store2((rowBase3 + col0) << 1, uint2(out01, out23));
}