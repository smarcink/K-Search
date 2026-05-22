#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer InVectors : register(t0);
ByteAddressBuffer InMatrices : register(t1);
RWByteAddressBuffer OutVectors : register(u0);

[numthreads(1, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID) {
    MatrixRef<DATA_TYPE_FLOAT32, 1, 1, MATRIX_LAYOUT_ROW_MAJOR> matrix_ref = {
        InMatrices,
        0,
        sizeof(float)
    };

    vector<float, 1> input_vector = InVectors.Load<vector<float, 1> >(0);
    vector<float, 1> result = Mul<float>(
        matrix_ref,
        MakeInterpretedVector<DATA_TYPE_FLOAT32>(input_vector)
    );

    OutVectors.Store<vector<float, 1> >(0, result);
}
