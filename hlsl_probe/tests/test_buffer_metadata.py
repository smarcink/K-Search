import unittest

from hlsl_probe import BufferArg, BufferSpec


class BufferMetadataTests(unittest.TestCase):
    def test_typed_shape_infers_size(self):
        spec = BufferSpec("float16", shape=(2, 3))

        self.assertEqual(spec.dtype, "float16")
        self.assertEqual(spec.view, "typed")
        self.assertEqual(spec.element_count, 6)
        self.assertEqual(spec.size_bytes, 12)

    def test_raw_u32_helper(self):
        spec = BufferSpec.raw_u32(16)

        self.assertEqual(spec.dtype, "raw_u32")
        self.assertEqual(spec.view, "raw")
        self.assertEqual(spec.element_count, 4)
        self.assertEqual(spec.size_bytes, 16)

    def test_raw_u32_requires_alignment(self):
        with self.assertRaises(ValueError):
            BufferSpec.raw_u32(6)

    def test_rejects_invalid_raw_dtype(self):
        with self.assertRaises(ValueError):
            BufferSpec("float16", element_count=4, view="raw")

    def test_rejects_raw_u32_as_typed(self):
        with self.assertRaises(ValueError):
            BufferSpec("raw_u32", element_count=4, view="typed")

    def test_rejects_size_mismatch(self):
        with self.assertRaises(ValueError):
            BufferSpec("int8", element_count=4, size_bytes=8)

    def test_buffer_arg_size_matches_spec(self):
        with self.assertRaises(ValueError):
            BufferArg(b"abc", BufferSpec("uint8", element_count=4))


if __name__ == "__main__":
    unittest.main()