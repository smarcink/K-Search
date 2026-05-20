import struct
import unittest

from hlsl_probe import BufferArg, BufferSpec, compile_hlsl_source, probe_caps, run_dxil, self_test


class RawApiTests(unittest.TestCase):
    def test_python_self_test(self):
        result = self_test()

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["output_value"], 123)
        self.assertEqual(result["outputs"][0]["view_kind"], "raw")
        self.assertEqual(result["outputs"][0]["format"], "raw_u32")

    def test_probe_reports_modern_linalg_capabilities(self):
        caps = probe_caps()

        self.assertEqual(caps["status"], "passed")
        self.assertIn("linear_algebra_query_ok", caps)
        self.assertIn("linear_algebra_tier_name", caps)
        self.assertIn("linear_algebra_thread_vector_matrix_multiply", caps)
        self.assertIn("linear_algebra_wave_matrix_multiply", caps)
        self.assertIsInstance(caps["linear_algebra_thread_vector_matrix_multiply"], list)
        self.assertIsInstance(caps["linear_algebra_wave_matrix_multiply"], list)

    def test_raw_u32_add_kernel(self):
        shader = """
ByteAddressBuffer Input : register(t0);
RWByteAddressBuffer Output : register(u0);

[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint value = Input.Load(tid.x * 4);
    Output.Store(tid.x * 4, value + 7u);
}
"""
        dxil = compile_hlsl_source(shader, target="cs_6_8")
        input_bytes = struct.pack("4I", 10, 20, 30, 40)
        metadata, outputs = run_dxil(
            dxil,
            inputs=[BufferArg.raw_u32(input_bytes)],
            outputs=[BufferSpec.raw_u32(16)],
            dispatch=(1, 1, 1),
        )

        self.assertEqual(metadata["status"], "passed")
        self.assertEqual(metadata["inputs"][0]["format"], "raw_u32")
        self.assertEqual(metadata["outputs"][0]["format"], "raw_u32")
        self.assertEqual(list(struct.unpack("4I", outputs[0])), [17, 27, 37, 47])


if __name__ == "__main__":
    unittest.main()