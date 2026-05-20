import unittest

from hlsl_probe import BufferArg, BufferSpec, HlslProbeError, compile_hlsl_source, run_dxil

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment-dependent POC dependency
    torch = None
    nn = None


def _tensor_from_output(data: bytes, dtype):
    return torch.frombuffer(bytearray(data), dtype=dtype).clone()


if nn is not None:
    class Fp16Reference(nn.Module):
        def forward(self, tensor):
            return tensor * tensor.new_tensor(2.0) + tensor.new_tensor(1.0)


    class Int8Reference(nn.Module):
        def forward(self, tensor):
            return (tensor.to(torch.int16) + 2).to(torch.int8)
else:
    Fp16Reference = None
    Int8Reference = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TypedBufferTests(unittest.TestCase):
    def test_typed_fp16_buffer(self):
        shader = """
Buffer<float> Input : register(t0);
RWBuffer<float> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        Output[tid.x] = Input[tid.x] * 2.0f + 1.0f;
    }
}
"""
        x = (torch.arange(16, dtype=torch.float32) * 0.25).to(torch.float16).contiguous()
        expected = Fp16Reference()(x).contiguous()
        dxil = compile_hlsl_source(shader, target="cs_6_8")

        try:
            metadata, outputs = run_dxil(
                dxil,
                inputs=[BufferArg.from_tensor(x)],
                outputs=[BufferSpec.from_tensor(expected)],
                dispatch=(1, 1, 1),
            )
        except HlslProbeError as exc:
            if "not supported" in str(exc):
                self.skipTest(str(exc))
            raise

        actual = _tensor_from_output(outputs[0], torch.float16)
        self.assertEqual(metadata["inputs"][0]["format"], "float16")
        self.assertEqual(metadata["outputs"][0]["format"], "float16")
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-3, atol=1e-3), (actual, expected))

    def test_typed_int8_buffer(self):
        shader = """
Buffer<int> Input : register(t0);
RWBuffer<int> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        Output[tid.x] = Input[tid.x] + 2;
    }
}
"""
        x = torch.arange(-8, 8, dtype=torch.int16).to(torch.int8).contiguous()
        expected = Int8Reference()(x).contiguous()
        dxil = compile_hlsl_source(shader, target="cs_6_8")

        try:
            metadata, outputs = run_dxil(
                dxil,
                inputs=[BufferArg.from_tensor(x)],
                outputs=[BufferSpec.from_tensor(expected)],
                dispatch=(1, 1, 1),
            )
        except HlslProbeError as exc:
            if "not supported" in str(exc):
                self.skipTest(str(exc))
            raise

        actual = _tensor_from_output(outputs[0], torch.int8)
        self.assertEqual(metadata["inputs"][0]["format"], "int8")
        self.assertEqual(metadata["outputs"][0]["format"], "int8")
        self.assertTrue(torch.equal(actual, expected), (actual, expected))


if __name__ == "__main__":
    unittest.main()