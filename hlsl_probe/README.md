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
```

`--probe` prints adapter, shader model, Agility SDK, and WaveMMA capability JSON. `--self-test` compiles a tiny SM 6.8 compute shader with the downloaded DXC, dispatches it, reads back a 32-bit value, and reports GPU timestamp timing. Higher shader-model targets can still be passed explicitly when the runtime reports support for them.

## Run From Python

```powershell
$env:PYTHONPATH = "$PWD\hlsl_probe\python"
python -m hlsl_probe --probe
python -m hlsl_probe --self-test
```

The Python wrapper uses `ctypes` over `hlsl_probe_native.dll`. It stages buffers through CPU memory for the first POC; direct PyTorch GPU interop is intentionally deferred.
