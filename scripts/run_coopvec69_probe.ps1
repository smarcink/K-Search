param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release",
    [switch]$SkipShaderCompile,
    [switch]$SkipRun,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "vs_dev_env.ps1")

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DxcRoot = Join-Path $RepoRoot "thirdparty\dxc_2025_05_24"
$AgilityRoot = Join-Path $RepoRoot "thirdparty\microsoft.direct3d.d3d12.1.717.0-preview"
$ProbeSource = Join-Path $RepoRoot "hlsl_probe\tools\coopvec69_probe.cpp"
$ShaderSource = Join-Path $RepoRoot "hlsl_probe\tests\fixtures\coopvec69_smoke.hlsl"
$BuildRoot = Join-Path $RepoRoot "hlsl_probe\build\coopvec69"
$BuildDir = Join-Path $BuildRoot $Config
$DxilPath = Join-Path $RepoRoot "hlsl_probe\build\coopvec69_smoke.dxil"

$DxcExe = Join-Path $DxcRoot "bin\x64\dxc.exe"
$DxcInclude = Join-Path $DxcRoot "inc\hlsl"
$AgilityInclude = Join-Path $AgilityRoot "build\native\include"
$AgilityBin = Join-Path $AgilityRoot "build\native\bin\x64"

foreach ($Path in @($DxcExe, $DxcInclude, $AgilityInclude, $AgilityBin, $ProbeSource, $ShaderSource)) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

New-Item -ItemType Directory -Force $BuildDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $BuildDir "D3D12") | Out-Null

if (-not $SkipShaderCompile) {
    & $DxcExe $ShaderSource -T cs_6_9 -E main -HV 2021 -I $DxcInclude -enable-16bit-types -Fo $DxilPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Copy-Item (Join-Path $AgilityBin "D3D12Core.dll") (Join-Path $BuildDir "D3D12") -Force
Copy-Item (Join-Path $AgilityBin "d3d12SDKLayers.dll") (Join-Path $BuildDir "D3D12") -Force

$VsSetupBlock = New-VsDevCmdSetupBlock

$ProbeExe = Join-Path $BuildDir "coopvec69_probe.exe"
$ProbeObj = Join-Path $BuildDir "coopvec69_probe.obj"
$CompileFlags = if ($Config -eq "Release") { "/O2 /DNDEBUG /MD" } else { "/Od /Zi /D_DEBUG /MDd" }
$CmdFile = Join-Path ([System.IO.Path]::GetTempPath()) "build_coopvec69_probe_$PID.cmd"
@"
@echo off
$VsSetupBlock
cd /d "$RepoRoot"
cl /nologo /std:c++17 /EHsc $CompileFlags /I"$AgilityInclude" /Fo"$ProbeObj" /Fe"$ProbeExe" "$ProbeSource" d3d12.lib dxgi.lib
exit /b %errorlevel%
"@ | Set-Content -Path $CmdFile -Encoding ASCII

try {
    cmd.exe /d /s /c "`"$CmdFile`""
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Remove-Item $CmdFile -ErrorAction SilentlyContinue
}

Write-Host "Built: $ProbeExe"

if ($SkipRun) {
    return
}

Write-Host "=== shader-models-only ==="
& $ProbeExe --shader-models-only $DxilPath
Write-Host "shader_models_only_exit=$LASTEXITCODE"

Write-Host "=== coop-experiment ==="
& $ProbeExe $DxilPath
Write-Host "coop_experiment_exit=$LASTEXITCODE"
