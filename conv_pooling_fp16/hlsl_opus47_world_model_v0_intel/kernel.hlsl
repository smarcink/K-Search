// Fused Conv3x3 (IC=16, OC=32, pad=1, stride=1) + ReLU + AvgPool2x2
#define IC 16
#define OC 32
#define H_IN 720
#define W_IN 1280
#define H_OUT 360
#define W_OUT 640

#define POOL_TILE_X 8
#define POOL_TILE_Y 8
#define IN_TILE_X 18
#define IN_TILE_Y 18

Buffer<float16_t> input_buf : register(t0);
Buffer<float16_t> act_max_log : register(t1);
Buffer<float16_t> weights_buf : register(t2);
Buffer<float16_t> bias_buf : register(t3);

RWBuffer<float16_t> output_buf : register(u0);

groupshared float16_t gs_weights[OC * IC * 9];
groupshared float16_t gs_bias[OC];
groupshared float16_t gs_input[IC][IN_TILE_Y][IN_TILE_X];

#define THREADS_X 8
#define THREADS_Y 8
#define THREADS_Z 4
#define NUM_THREADS (THREADS_X * THREADS_Y * THREADS_Z)

[numthreads(THREADS_X, THREADS_Y, THREADS_Z)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint tIdx : SV_GroupIndex)
{
    int pool_x0 = int(gid.x) * POOL_TILE_X;
    int pool_y0 = int(gid.y) * POOL_TILE_Y;
    int conv_x0 = pool_x0 * 2;
    int conv_y0 = pool_y0 * 2;
    int in_x0 = conv_x0 - 1;
    int in_y0 = conv_y0 - 1;

    const uint W_TOTAL = OC * IC * 9;
    {
        [loop]
        for (uint i = tIdx; i < W_TOTAL; i += NUM_THREADS) {
            gs_weights[i] = weights_buf[i];
        }
    }
    if (tIdx < OC) {
        gs_bias[tIdx] = bias_buf[tIdx];
    }

    const uint IN_TOTAL = IC * IN_TILE_Y * IN_TILE_X;
    bool interior = (in_x0 >= 0) && (in_y0 >= 0) && (in_x0 + IN_TILE_X <= W_IN) && (in_y0 + IN_TILE_Y <= H_IN);

    if (interior) {
        [loop]
        for (uint j = tIdx; j < IN_TOTAL; j += NUM_THREADS) {
            uint tx = j % IN_TILE_X;
            uint ty = (j / IN_TILE_X) % IN_TILE_Y;
            uint c  = j / (IN_TILE_X * IN_TILE_Y);
            uint idx = (c * H_IN + uint(in_y0 + int(ty))) * W_IN + uint(in_x0 + int(tx));
            gs_input[c][ty][tx] = input_buf[idx];
        }
    } else {
        [loop]
        for (uint j = tIdx; j < IN_TOTAL; j += NUM_THREADS) {
            uint tx = j % IN_TILE_X;
            uint ty = (j / IN_TILE_X) % IN_TILE_Y;
            uint c  = j / (IN_TILE_X * IN_TILE_Y);
            int gx = in_x0 + int(tx);
            int gy = in_y0 + int(ty);
            float16_t v = (float16_t)0.0;
            if (gx >= 0 && gx < W_IN && gy >= 0 && gy < H_IN) {
                uint idx = (c * H_IN + uint(gy)) * W_IN + uint(gx);
                v = input_buf[idx];
            }
            gs_input[c][ty][tx] = v;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint px = gtid.x;
    uint py = gtid.y;
    uint pz = gtid.z;

    uint pool_x = uint(pool_x0) + px;
    uint pool_y = uint(pool_y0) + py;

    if (pool_x >= W_OUT || pool_y >= H_OUT) return;

    uint cx0 = px * 2;
    uint cy0 = py * 2;

    const uint OC_PER_THREAD = OC / THREADS_Z; // 8

    float acc00_0, acc00_1, acc00_2, acc00_3, acc00_4, acc00_5, acc00_6, acc00_7;
    float acc01_0, acc01_1, acc01_2, acc01_3, acc01_4, acc01_5, acc01_6, acc01_7;
    float acc10_0, acc10_1, acc10_2, acc10_3, acc10_4, acc10_5, acc10_6, acc10_7;
    float acc11_0, acc11_1, acc11_2, acc11_3, acc11_4, acc11_5, acc11_6, acc11_7;

    uint oc_base = pz * OC_PER_THREAD;
    {
        float b0 = (float)gs_bias[oc_base + 0];
        float b1 = (float)gs_bias[oc_base + 1];
        float b2 = (float)gs_bias[oc_base + 2];
        float b3 = (float)gs_bias[oc_base + 3];
        float b4 = (float)gs_bias[oc_base + 4];
        float b5 = (float)gs_bias[oc_base + 5];
        float b6 = (float)gs_bias[oc_base + 6];
        float b7 = (float)gs_bias[oc_base + 7];
        acc00_0=b0; acc01_0=b0; acc10_0=b0; acc11_0=b0;
        acc00_1=b1; acc01_1=b1; acc10_1=b1; acc11_1=b1;
        acc00_2=b2; acc01_2=b2; acc10_2=b2; acc11_2=b2;
        acc00_3=b3; acc01_3=b3; acc10_3=b3; acc11_3=b3;
        acc00_4=b4; acc01_4=b4; acc10_4=b4; acc11_4=b4;
        acc00_5=b5; acc01_5=b5; acc10_5=b5; acc11_5=b5;
        acc00_6=b6; acc01_6=b6; acc10_6=b6; acc11_6=b6;
        acc00_7=b7; acc01_7=b7; acc10_7=b7; acc11_7=b7;
    }

    [unroll]
    for (uint c = 0; c < IC; ++c) {
        float v00 = (float)gs_input[c][cy0 + 0][cx0 + 0];
        float v01 = (float)gs_input[c][cy0 + 0][cx0 + 1];
        float v02 = (float)gs_input[c][cy0 + 0][cx0 + 2];
        float v03 = (float)gs_input[c][cy0 + 0][cx0 + 3];
        float v10 = (float)gs_input[c][cy0 + 1][cx0 + 0];
        float v11 = (float)gs_input[c][cy0 + 1][cx0 + 1];
        float v12 = (float)gs_input[c][cy0 + 1][cx0 + 2];
        float v13 = (float)gs_input[c][cy0 + 1][cx0 + 3];
        float v20 = (float)gs_input[c][cy0 + 2][cx0 + 0];
        float v21 = (float)gs_input[c][cy0 + 2][cx0 + 1];
        float v22 = (float)gs_input[c][cy0 + 2][cx0 + 2];
        float v23 = (float)gs_input[c][cy0 + 2][cx0 + 3];
        float v30 = (float)gs_input[c][cy0 + 3][cx0 + 0];
        float v31 = (float)gs_input[c][cy0 + 3][cx0 + 1];
        float v32 = (float)gs_input[c][cy0 + 3][cx0 + 2];
        float v33 = (float)gs_input[c][cy0 + 3][cx0 + 3];

        [unroll]
        for (uint oi = 0; oi < OC_PER_THREAD; ++oi) {
            uint oc = oc_base + oi;
            uint w_base = ((oc * IC) + c) * 9;
            float w0 = (float)gs_weights[w_base + 0];
            float w1 = (float)gs_weights[w_base + 1];
            float w2 = (float)gs_weights[w_base + 2];
            float w3 = (float)gs_weights[w_base + 3];
            float w4 = (float)gs_weights[w_base + 4];
            float w5 = (float)gs_weights[w_base + 5];
            float w6 = (float)gs_weights[w_base + 6];
            float w7 = (float)gs_weights[w_base + 7];
            float w8 = (float)gs_weights[w_base + 8];

            float a00 = w0*v00 + w1*v01 + w2*v02 + w3*v10 + w4*v11 + w5*v12 + w6*v20 + w7*v21 + w8*v22;
            float a01 = w0*v01 + w1*v02 + w2*v03 + w3*v11 + w4*v12 + w5*v13 + w6*v21 + w7*v22 + w8*v23;
            float a10 = w0*v10 + w1*v11 + w2*v12 + w3*v20 + w4*v21 + w5*v22 + w6*v30 + w7*v31 + w8*v32;
            float a11 = w0*v11 + w1*v12 + w2*v13 + w3*v21 + w4*v22 + w5*v23 + w6*v31 + w7*v32 + w8*v33;

            if (oi == 0) { acc00_0 += a00; acc01_0 += a01; acc10_0 += a10; acc11_0 += a11; }
            else if (oi == 1) { acc00_1 += a00; acc01_1 += a01; acc10_1 += a10; acc11_1 += a11; }
            else if (oi == 2) { acc00_2 += a00; acc01_2 += a01; acc10_2 += a10; acc11_2 += a11; }
            else if (oi == 3) { acc00_3 += a00; acc01_3 += a01; acc10_3 += a10; acc11_3 += a11; }
            else if (oi == 4) { acc00_4 += a00; acc01_4 += a01; acc10_4 += a10; acc11_4 += a11; }
            else if (oi == 5) { acc00_5 += a00; acc01_5 += a01; acc10_5 += a10; acc11_5 += a11; }
            else if (oi == 6) { acc00_6 += a00; acc01_6 += a01; acc10_6 += a10; acc11_6 += a11; }
            else              { acc00_7 += a00; acc01_7 += a01; acc10_7 += a10; acc11_7 += a11; }
        }
    }

    #define EMIT(IDX, A00, A01, A10, A11) { \
        float a00 = max(A00, 0.0f); \
        float a01 = max(A01, 0.0f); \
        float a10 = max(A10, 0.0f); \
        float a11 = max(A11, 0.0f); \
        float avg = (a00 + a01 + a10 + a11) * 0.25f; \
        uint out_idx = ((oc_base + IDX) * H_OUT + pool_y) * W_OUT + pool_x; \
        output_buf[out_idx] = (float16_t)avg; \
    }

    EMIT(0, acc00_0, acc01_0, acc10_0, acc11_0);
    EMIT(1, acc00_1, acc01_1, acc10_1, acc11_1);
    EMIT(2, acc00_2, acc01_2, acc10_2, acc11_2);
    EMIT(3, acc00_3, acc01_3, acc10_3, acc11_3);
    EMIT(4, acc00_4, acc01_4, acc10_4, acc11_4);
    EMIT(5, acc00_5, acc01_5, acc10_5, acc11_5);
    EMIT(6, acc00_6, acc01_6, acc10_6, acc11_6);
    EMIT(7, acc00_7, acc01_7, acc10_7, acc11_7);
}