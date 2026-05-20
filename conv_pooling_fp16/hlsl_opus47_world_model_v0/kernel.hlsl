// Fused Conv2D(3x3, 16->32, stride1, pad1) + bias + ReLU + 2x2 AvgPool
// Reverting packed half2 attempt (failed correctness). Use base scalar fp16
// implementation with minor improvement: precompute output base index and
// fold 0.25 into accumulation.

#define IC 16
#define OC 32
#define H_IN 720
#define W_IN 1280
#define H_OUT 360
#define W_OUT 640

#define TILE_OY 8
#define TILE_OX 8
#define TILE_CY (TILE_OY*2)   // 16
#define TILE_CX (TILE_OX*2)   // 16
#define HALO_Y  (TILE_CY+2)   // 18
#define HALO_X  (TILE_CX+2)   // 18

#define OC_PER_THREAD 16
#define THREADS 128

Buffer<float16_t> input_buf : register(t0);
Buffer<float16_t> act_max_log : register(t1);
Buffer<float16_t> weight_buf : register(t2);
Buffer<float16_t> bias_buf   : register(t3);

RWBuffer<float16_t> output_buf : register(u0);

// Layout: [ic * 9 + k] * OC + oc
groupshared float16_t s_weights[IC * 9 * OC]; // 4608
groupshared float16_t s_in[HALO_Y * HALO_X];  // 324
groupshared float16_t s_bias[OC];

[numthreads(8, 8, 2)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uint lidx : SV_GroupIndex)
{
    uint tx = ltid.x;
    uint ty = ltid.y;
    uint tz = ltid.z;

    uint out_ox0 = gid.x * TILE_OX;
    uint out_oy0 = gid.y * TILE_OY;

    int conv_x0 = (int)(out_ox0 * 2);
    int conv_y0 = (int)(out_oy0 * 2);

    int halo_x0 = conv_x0 - 1;
    int halo_y0 = conv_y0 - 1;

    // Load weights cooperatively with layout transpose.
    [unroll]
    for (uint i = 0; i < 36; ++i) {
        uint idx = lidx + i * THREADS;
        uint oc = idx / (IC * 9);
        uint rem = idx - oc * (IC * 9);
        uint ic = rem / 9;
        uint k = rem - ic * 9;
        float16_t w = weight_buf.Load(idx);
        uint dst = (ic * 9 + k) * OC + oc;
        s_weights[dst] = w;
    }

    // Load bias cooperatively
    if (lidx < OC) {
        s_bias[lidx] = bias_buf.Load(lidx);
    }

    uint oc_base = tz * OC_PER_THREAD;

    float16_t acc[2][2][OC_PER_THREAD];
    [unroll] for (int py = 0; py < 2; ++py)
        [unroll] for (int px = 0; px < 2; ++px)
            [unroll] for (int oc = 0; oc < OC_PER_THREAD; ++oc)
                acc[py][px][oc] = (float16_t)0.0;

    uint ty2 = ty * 2;
    uint tx2 = tx * 2;

    bool full_inside = (halo_x0 >= 0) && (halo_y0 >= 0) &&
                       (halo_x0 + HALO_X <= W_IN) && (halo_y0 + HALO_Y <= H_IN);

    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint ic = 0; ic < IC; ++ic) {
        uint ic_base = ic * H_IN * W_IN;

        if (full_inside) {
            [unroll]
            for (uint i = 0; i < 3; ++i) {
                uint sidx = lidx + i * THREADS;
                if (sidx < HALO_Y * HALO_X) {
                    uint ly = sidx / HALO_X;
                    uint lx = sidx - ly * HALO_X;
                    uint gy = (uint)halo_y0 + ly;
                    uint gx = (uint)halo_x0 + lx;
                    uint gidx = ic_base + gy * W_IN + gx;
                    s_in[sidx] = input_buf.Load(gidx);
                }
            }
        } else {
            [unroll]
            for (uint i = 0; i < 3; ++i) {
                uint sidx = lidx + i * THREADS;
                if (sidx < HALO_Y * HALO_X) {
                    int ly = (int)(sidx / HALO_X);
                    int lx = (int)(sidx - (uint)ly * HALO_X);
                    int gy = halo_y0 + ly;
                    int gx = halo_x0 + lx;
                    float16_t v = (float16_t)0.0;
                    if (gy >= 0 && gy < H_IN && gx >= 0 && gx < W_IN) {
                        uint gidx = ic_base + (uint)gy * W_IN + (uint)gx;
                        v = input_buf.Load(gidx);
                    }
                    s_in[sidx] = v;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        float16_t patch[4][4];
        [unroll]
        for (int dy = 0; dy < 4; ++dy) {
            uint row = (ty2 + dy) * HALO_X + tx2;
            [unroll]
            for (int dx = 0; dx < 4; ++dx) {
                patch[dy][dx] = s_in[row + dx];
            }
        }

        uint w_ic_base = ic * 9 * OC;

        [unroll]
        for (int oc = 0; oc < OC_PER_THREAD; ++oc) {
            uint oc_g = oc_base + (uint)oc;
            float16_t w00 = s_weights[w_ic_base + 0 * OC + oc_g];
            float16_t w01 = s_weights[w_ic_base + 1 * OC + oc_g];
            float16_t w02 = s_weights[w_ic_base + 2 * OC + oc_g];
            float16_t w10 = s_weights[w_ic_base + 3 * OC + oc_g];
            float16_t w11 = s_weights[w_ic_base + 4 * OC + oc_g];
            float16_t w12 = s_weights[w_ic_base + 5 * OC + oc_g];
            float16_t w20 = s_weights[w_ic_base + 6 * OC + oc_g];
            float16_t w21 = s_weights[w_ic_base + 7 * OC + oc_g];
            float16_t w22 = s_weights[w_ic_base + 8 * OC + oc_g];

            float16_t s00 = acc[0][0][oc];
            float16_t s01 = acc[0][1][oc];
            float16_t s10 = acc[1][0][oc];
            float16_t s11 = acc[1][1][oc];

            s00 += patch[0][0] * w00;
            s00 += patch[0][1] * w01;
            s00 += patch[0][2] * w02;
            s00 += patch[1][0] * w10;
            s00 += patch[1][1] * w11;
            s00 += patch[1][2] * w12;
            s00 += patch[2][0] * w20;
            s00 += patch[2][1] * w21;
            s00 += patch[2][2] * w22;

            s01 += patch[0][1] * w00;
            s01 += patch[0][2] * w01;
            s01 += patch[0][3] * w02;
            s01 += patch[1][1] * w10;
            s01 += patch[1][2] * w11;
            s01 += patch[1][3] * w12;
            s01 += patch[2][1] * w20;
            s01 += patch[2][2] * w21;
            s01 += patch[2][3] * w22;

            s10 += patch[1][0] * w00;
            s10 += patch[1][1] * w01;
            s10 += patch[1][2] * w02;
            s10 += patch[2][0] * w10;
            s10 += patch[2][1] * w11;
            s10 += patch[2][2] * w12;
            s10 += patch[3][0] * w20;
            s10 += patch[3][1] * w21;
            s10 += patch[3][2] * w22;

            s11 += patch[1][1] * w00;
            s11 += patch[1][2] * w01;
            s11 += patch[1][3] * w02;
            s11 += patch[2][1] * w10;
            s11 += patch[2][2] * w11;
            s11 += patch[2][3] * w12;
            s11 += patch[3][1] * w20;
            s11 += patch[3][2] * w21;
            s11 += patch[3][3] * w22;

            acc[0][0][oc] = s00;
            acc[0][1][oc] = s01;
            acc[1][0][oc] = s10;
            acc[1][1][oc] = s11;
        }

        if (ic + 1 < IC) {
            GroupMemoryBarrierWithGroupSync();
        }
    }

    uint out_oy = out_oy0 + ty;
    uint out_ox = out_ox0 + tx;

    if (out_oy < H_OUT && out_ox < W_OUT) {
        uint out_yx = out_oy * W_OUT + out_ox;
        [unroll]
        for (int oc = 0; oc < OC_PER_THREAD; ++oc) {
            uint oc_g = oc_base + (uint)oc;
            float b = (float)s_bias[oc_g];
            float v00 = max((float)acc[0][0][oc] + b, 0.0f);
            float v01 = max((float)acc[0][1][oc] + b, 0.0f);
            float v10 = max((float)acc[1][0][oc] + b, 0.0f);
            float v11 = max((float)acc[1][1][oc] + b, 0.0f);
            float avg = (v00 + v01 + v10 + v11) * 0.25f;

            uint out_idx = oc_g * (H_OUT * W_OUT) + out_yx;
            output_buf[out_idx] = (float16_t)avg;
        }
    }
}