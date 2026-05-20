# HLSL Probe Runner

This is a focused Direct3D 12 probe used to validate the local DXC + Agility SDK setup before adding an HLSL backend to K-Search.

## Build

From the repository root:

```powershell
.\scripts\build_hlsl_probe.ps1 -Config Debug
```

The script enters the Visual Studio x64 developer environment internally, configures CMake, builds the native DLL and CLI, and copies the Agility SDK runtime DLLs into the output `D3D12` folder.

## Run Standalone

```powershell
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --probe
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --probe --compact-json
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --self-test
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --linalg-test
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --linalg-fp16-test
```

`--probe` prints adapter, shader model, Agility SDK, WaveMMA, and modern D3D12 linear algebra capability JSON in an indented terminal-readable form. Use `--probe --compact-json` for the original single-line machine output. `--self-test` compiles a tiny SM 6.8 compute shader with the downloaded DXC, dispatches it, reads back a 32-bit value, and reports GPU timestamp timing. `--linalg-test` compiles a tiny U32 `dx/linalg.h` matrix-vector shader for `cs_6_10` by default, then reports whether failure happened during compile, PSO creation, dispatch/readback, or output verification. `--linalg-fp16-test` does the same for a tiny FP16 thread matrix-vector multiply and verifies FP16 output values.

## Run From Python

```powershell
$env:PYTHONPATH = "$PWD\hlsl_probe\python"
python -m hlsl_probe --probe
python -m hlsl_probe --self-test
python -m hlsl_probe --linalg-test
python -m hlsl_probe --linalg-fp16-test
python -m hlsl_probe --linalg-fp16-shape-sweep --wave-sizes 0,16,32 --m-values 2,4,8,16 --k-values 4,8,16,32,64,128
```

The Python wrapper uses `ctypes` over `hlsl_probe_native.dll`. It stages buffers through CPU memory for the first POC; direct PyTorch GPU interop is intentionally deferred.

`--linalg-fp16-shape-sweep` empirically compiles, dispatches, verifies, and times candidate `Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread>` shapes. It reports per-shape compile/run/verify failures, GPU timestamp samples, `gfma_per_s_min_time`, `tfma_per_s_min_time`, `tflop_per_s_fma2_min_time`, `ns_per_matvec_min_time`, and fastest cases by raw time and throughput. Tiny sweeps are useful for shape ranking but are often launch/timestamp limited; use larger `--shape-reps` and `--shape-dispatch-groups` for roofline-style throughput checks. By default it uses `Multiply<float16_t>` with FP32 register accumulation outside LinAlg, matching the locally supported capability path. `--wave-sizes 0,16,32` compares omitted `[WaveSize]`, `[WaveSize(16)]`, and `[WaveSize(32)]`; `--shape-accumulators fp16,fp32 --shape-ops multiply,multiplyadd` can explicitly compare direct `MultiplyAdd` against the safer `Multiply` path.

## Buffer Metadata

The probe API uses explicit buffer metadata instead of guessing tensor formats from raw byte counts. Each input and output buffer declares:

- `view`: `raw` or `typed`
- `dtype`: `raw_u32`, `float16`, `float32`, `int8`, `uint8`, `int32`, or `uint32`
- `element_count`
- `size_bytes`

Raw buffers map to `ByteAddressBuffer` / `RWByteAddressBuffer` with `DXGI_FORMAT_R32_TYPELESS` and raw descriptor flags. Typed buffers map to typed DXGI views such as `DXGI_FORMAT_R16_FLOAT`, `DXGI_FORMAT_R8_SINT`, and `DXGI_FORMAT_R32_FLOAT` for `Buffer<T>` / `RWBuffer<T>` style HLSL.

Example raw call shape:

```python
from hlsl_probe import BufferArg, BufferSpec, run_dxil

metadata, outputs = run_dxil(
	dxil,
	inputs=[BufferArg.raw_u32(input_bytes)],
	outputs=[BufferSpec.raw_u32(16)],
	dispatch=(1, 1, 1),
)
```

Example typed call shape:

```python
from hlsl_probe import BufferArg, BufferSpec, run_dxil

metadata, outputs = run_dxil(
	dxil,
	inputs=[BufferArg.from_tensor(torch_tensor)],
	outputs=[BufferSpec.from_tensor(expected_tensor)],
	dispatch=(1, 1, 1),
)
```

## Tests

The probe has Python API tests under `hlsl_probe/tests/`. They exercise the native C ABI through the `ctypes` wrapper.

```powershell
$env:PYTHONPATH = "$PWD\hlsl_probe\python"
python -m unittest discover hlsl_probe/tests
```

The typed FP16 and int8 tests may report an explicit capability skip/failure if the current runtime does not expose the needed typed SRV/UAV format support.

## Linear Algebra Capabilities

Agility SDK 720 does not expose the older blog-era `D3D12CooperativeVectorExperiment` / `CooperativeVectorTier` names. The probe reports the newer preview capability surface instead:

- `linear_algebra_query_ok`
- `linear_algebra_tier_name`
- `linear_algebra_thread_vector_matrix_multiply`
- `linear_algebra_wave_matrix_multiply`

The thread-vector matrix multiply entries correspond to the current `dx/linalg.h` cooperative vector replacement API. A successful query with `linear_algebra_tier_name` set to `not_supported` means the runtime understands the Agility 720 query but the driver/device does not currently expose the feature. When FP16 thread-vector matrix multiply reports `supported: true`, run `--linalg-fp16-test --target cs_6_10` as the next proof that compile, PSO creation, dispatch, and readback all work for the actual FP16 path.

## Preview SDK Notes

The current configuration uses the preview DXC and Agility SDK packages in `thirdparty/dxc_preview_2026_04_22` and `thirdparty/microsoft.direct3d.d3d12.1.720.0-preview`. Windows Developer Mode must be enabled for `D3D12ExperimentalShaderModels`; without it, preview Agility device creation can fail before any shader is tested.

On the RTX 5090 Laptop GPU with driver `596.49`, Developer Mode allows Agility SDK 720 to load and reports WaveMMA tier `1_0`, but the runtime still reports highest shader model `6.9` and `linear_algebra_tier_name` as `not_supported`. The `--linalg-test` shader compiles as `cs_6_10` with preview DXC, then currently fails at PSO creation when `supports_sm_6_10` is false.
