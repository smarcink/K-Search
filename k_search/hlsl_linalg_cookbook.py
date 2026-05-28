"""Prompt cookbook for SM 6.10 Direct3D Linear Algebra generation."""

HLSL_LINALG_COOKBOOK = """## SM 6.10 Direct3D Linear Algebra Cookbook
Distilled from:
- https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md
- https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/
- https://devblogs.microsoft.com/directx/d3d12-linalg-preview/
- the local hlsl_probe FP16 thread-scope smoke test.

Use this as generation guidance when the task explicitly enables SM 6.10 Direct3D Linear Algebra; local compile/eval is still the source of truth.

Supported path to prefer in this repo when linalg is enabled:
- Include #include <dx/linalg.h> and use cs_6_10.
- Use raw ByteAddressBuffer/RWByteAddressBuffer resources and explicit byte offsets.
- Prefer Thread-scope matrix-vector linalg for dense FP16 inner K tiles:
  Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread> times vector<float16_t, K>.
- Thread-scope matrices are per-thread/read-only A matrices loaded from ByteAddressBuffer. They may be used from divergent control flow.
- The thread-scope K dimension must be a fixed fragment. The spec allows K in roughly [4,128], but hardware support can be narrower. On the local Intel Arc B580 stack, hlsl_probe shape sweep shows the strongest saturated FP16 thread matvec shape is M=16,K=64 with [WaveSize(32)] using Multiply<float16_t> plus FP32 register accumulation. Good fallback shapes are M=8,K=128, M=8,K=64, M=16,K=32, and M=4,K=128, all with [WaveSize(32)].
- Local caps currently report FP16 thread vector-matrix support with FP16 bias/result; direct FP32-output thread matvec is not assumed available.
- For Linear/GEMM reductions, prefer the locally validated compatibility path: vector<float16_t, M> delta16 = dx::linalg::Multiply<float16_t>(wFrag, xFrag), immediately widen each element to float, and keep the rest of the K reduction in FP32 registers. This still gives FP32 accumulation across K tiles; only the per-fragment linalg result is FP16.
- Treat direct vector<float, M> acc = dx::linalg::MultiplyAdd<float>(wFrag, xFrag, acc) as an optional diagnostic/probe path, not the default, unless hlsl_probe shape sweep or eval evidence shows it creates a valid PSO and runs on the target driver.
- For vector widths M > 4, do not use .x/.y/.z/.w swizzles; DXC rejects swizzles on vectors wider than 4. Use bracket indexing such as acc[0] and delta16[0].
- Do not use Wave or ThreadGroup matrix-matrix linalg, MatrixUse::B, MatrixUse::Accumulator, MultiplyAccumulate, Matrix::Length/Get/Set, or groupshared matrix load/store unless the task explicitly asks and the control flow/scope requirements are satisfied. This runner's proven path is Thread-scope matvec.
- Do not satisfy a linalg action with #include <dx/linalg.h> alone. The core dense fragment must contain a real dx::linalg::Matrix<..., MatrixScope::Thread> and dx::linalg::Multiply or dx::linalg::MultiplyAdd call.
- Avoid artificial, unused, or no-op linalg calls. If the dense fragment cannot be expressed with the supported API, use scalar HLSL and let the eval evidence update the world model.

Thread-scope FP32-register accumulating matvec pattern for Linear/GEMM:
using namespace dx::linalg;
using WFrag16x64 = Matrix<ComponentType::F16, 16, 64, MatrixUse::A, MatrixScope::Thread>;

vector<float, 16> acc;
[unroll]
for (uint oi = 0; oi < 16; ++oi) acc[oi] = 0.0f;

uint weightByteOffset = ((outBase * IN_FEATURES) + k0) * sizeof(float16_t);
uint xByteOffset = ((row * IN_FEATURES) + k0) * sizeof(float16_t);

WFrag16x64 wFrag = WFrag16x64::Load<MatrixLayout::RowMajor>(
    state_0,
    weightByteOffset,
    IN_FEATURES * sizeof(float16_t));
vector<float16_t, 64> xFrag = input_0.Load<vector<float16_t, 64> >(xByteOffset);
vector<float16_t, 16> delta16 = Multiply<float16_t>(wFrag, xFrag);
[unroll]
for (uint oi = 0; oi < 16; ++oi) acc[oi] += (float)delta16[oi];

Optional direct FP32-output diagnostic path if hlsl_probe says it is accepted on the target:
acc = MultiplyAdd<float>(wFrag, xFrag, acc);

Mapping note for nn.Linear:
- PyTorch weight is [OUT_FEATURES, IN_FEATURES] row-major. A linalg A-matrix row should correspond to one output channel and its K-fragment of weights.
- For a 16x64 fragment, M=16 output channels and K=64 input features. The matrix row stride is IN_FEATURES * sizeof(float16_t), and the x vector is the contiguous input row fragment at the same k0.
- The D3D12 examples page also describes D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL for preconverted production weights. The current hlsl_probe path uploads normal contiguous PyTorch buffers, so RowMajor is the practical starting point until runtime conversion support is wired into the evaluator.
"""
