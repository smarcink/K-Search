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
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --self-test
.\hlsl_probe\build\bin\Debug\hlsl_probe.exe --linalg-test
```

`--probe` prints adapter, shader model, Agility SDK, and WaveMMA capability JSON. `--self-test` compiles a tiny SM 6.8 compute shader with the downloaded DXC, dispatches it, reads back a 32-bit value, and reports GPU timestamp timing. `--linalg-test` compiles a tiny `dx/linalg.h` matrix-vector shader for `cs_6_10` by default, then reports whether failure happened during compile, PSO creation, dispatch/readback, or output verification.

## Run From Python

```powershell
$env:PYTHONPATH = "$PWD\hlsl_probe\python"
python -m hlsl_probe --probe
python -m hlsl_probe --self-test
python -m hlsl_probe --linalg-test
```

The Python wrapper uses `ctypes` over `hlsl_probe_native.dll`. It stages buffers through CPU memory for the first POC; direct PyTorch GPU interop is intentionally deferred.

## Preview SDK Notes

The current configuration uses the preview DXC and Agility SDK packages in `thirdparty/dxc_preview_2026_04_22` and `thirdparty/microsoft.direct3d.d3d12.1.720.0-preview`. Windows Developer Mode must be enabled for `D3D12ExperimentalShaderModels`; without it, preview Agility device creation can fail before any shader is tested.

On the RTX 5090 Laptop GPU with driver `596.49`, Developer Mode allows Agility SDK 720 to load and reports WaveMMA tier `1_0`, but the runtime still reports highest shader model `6.9`. The `--linalg-test` shader compiles as `cs_6_10` with preview DXC, then currently fails at PSO creation when `supports_sm_6_10` is false.
