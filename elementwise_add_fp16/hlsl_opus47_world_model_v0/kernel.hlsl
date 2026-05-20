ByteAddressBuffer x : register(t0);
ByteAddressBuffer y : register(t1);
RWByteAddressBuffer outBuf : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 grid : SV_GroupID)
{
    uint base = grid.x * (256u * 32u) + gtid.x * 16u;
    uint stride = 256u * 16u;

    uint4 a0 = x.Load4(base);
    uint4 b0 = y.Load4(base);
    uint4 a1 = x.Load4(base + stride);
    uint4 b1 = y.Load4(base + stride);

    uint4 r0, r1;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        vector<float16_t, 2> av, bv, sv;
        av.x = asfloat16((uint16_t)(a0[i] & 0xFFFFu));
        av.y = asfloat16((uint16_t)(a0[i] >> 16));
        bv.x = asfloat16((uint16_t)(b0[i] & 0xFFFFu));
        bv.y = asfloat16((uint16_t)(b0[i] >> 16));
        sv = av + bv;
        r0[i] = ((uint)asuint16(sv.x)) | (((uint)asuint16(sv.y)) << 16);

        av.x = asfloat16((uint16_t)(a1[i] & 0xFFFFu));
        av.y = asfloat16((uint16_t)(a1[i] >> 16));
        bv.x = asfloat16((uint16_t)(b1[i] & 0xFFFFu));
        bv.y = asfloat16((uint16_t)(b1[i] >> 16));
        sv = av + bv;
        r1[i] = ((uint)asuint16(sv.x)) | (((uint)asuint16(sv.y)) << 16);
    }

    outBuf.Store4(base, r0);
    outBuf.Store4(base + stride, r1);
}