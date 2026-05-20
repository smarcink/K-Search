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


    class ElementwiseReference(nn.Module):
        def forward(self, tensor):
            if tensor.dtype == torch.int8:
                values = tensor.to(torch.int16)
                return ((values + 3) * 2 - 1).to(torch.int8)
            return (tensor * tensor.new_tensor(1.5) + tensor.new_tensor(0.25)) - tensor * tensor.new_tensor(0.5)
else:
    Fp16Reference = None
    Int8Reference = None
    ElementwiseReference = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TypedBufferTests(unittest.TestCase):
    def test_typed_fp16_buffer(self):
        shader = """
Buffer<float16_t> Input : register(t0);
RWBuffer<float16_t> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        Output[tid.x] = Input[tid.x] * float16_t(2.0) + float16_t(1.0);
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

    def test_pytorch_module_elementwise_ops_across_dtypes(self):
        float_shader = """
Buffer<float> Input : register(t0);
RWBuffer<float> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        float value = Input[tid.x];
        Output[tid.x] = (value * 1.5f + 0.25f) - value * 0.5f;
    }
}
"""
        float16_shader = """
Buffer<float16_t> Input : register(t0);
RWBuffer<float16_t> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        float16_t value = Input[tid.x];
        Output[tid.x] = (value * float16_t(1.5) + float16_t(0.25)) - value * float16_t(0.5);
    }
}
"""
        int8_shader = """
// Metadata binds this as an R8_SINT typed view. HLSL exposes typed integer
// buffer elements as int values even though storage is one signed byte.
Buffer<int> Input : register(t0);
RWBuffer<int> Output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < 16) {
        int value = Input[tid.x];
        Output[tid.x] = ((value + 3) * 2) - 1;
    }
}
"""
        reference = ElementwiseReference()
        float_values = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
        cases = [
            ("float32", float_values.contiguous(), float_shader, 1e-6, 1e-6),
            ("float16", float_values.to(torch.float16).contiguous(), float16_shader, 1e-3, 1e-3),
            ("int8", torch.arange(-8, 8, dtype=torch.int16).to(torch.int8).contiguous(), int8_shader, 0.0, 0.0),
        ]

        for expected_format, x, shader, rtol, atol in cases:
            with self.subTest(dtype=expected_format):
                expected = reference(x).contiguous()
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

                actual = _tensor_from_output(outputs[0], x.dtype)
                self.assertEqual(metadata["inputs"][0]["format"], expected_format)
                self.assertEqual(metadata["outputs"][0]["format"], expected_format)
                if x.dtype == torch.int8:
                    self.assertTrue(torch.equal(actual, expected), (actual, expected))
                else:
                    self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected))


if __name__ == "__main__":
    unittest.main()