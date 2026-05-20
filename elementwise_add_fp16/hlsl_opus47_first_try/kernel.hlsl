Buffer<float16_t> in0 : register(t0);
Buffer<float16_t> in1 : register(t1);
RWBuffer<float16_t> out0 : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint base = gid.x * 8;
    [unroll]
    for (uint i = 0; i < 8; ++i) {
        uint idx = base + i;
        out0[idx] = in0[idx] + in1[idx];
    }
}