#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer InVectors : register(t0);
ByteAddressBuffer InMatrices : register(t1);
RWByteAddressBuffer OutVectors : register(u0);

[numthreads(1, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID) {
    MatrixRef<DATA_TYPE_FLOAT16, 16, 16, MATRIX_LAYOUT_ROW_MAJOR> matrix_ref = {
        InMatrices,
        0,
        16 * sizeof(float16_t)
    };

    vector<float16_t, 16> input_vector = InVectors.Load<vector<float16_t, 16> >(0);
    vector<float16_t, 16> result = Mul<float16_t>(
        matrix_ref,
        MakeInterpretedVector<DATA_TYPE_FLOAT16>(input_vector)
    );

    OutVectors.Store<vector<float16_t, 16> >(0, result);
}
