param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Debug",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "vs_dev_env.ps1")

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SourceDir = Join-Path $RepoRoot "hlsl_probe"
$BuildDir = Join-Path $SourceDir "build"
$DxcRoot = Join-Path $RepoRoot "thirdparty\dxc_preview_2026_04_22"
$AgilityRoot = Join-Path $RepoRoot "thirdparty\microsoft.direct3d.d3d12.1.720.0-preview"
$AgilitySdkVersion = 720

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

$VsSetupBlock = New-VsDevCmdSetupBlock

$CmdFile = Join-Path ([System.IO.Path]::GetTempPath()) "build_hlsl_probe_$PID.cmd"
@"
@echo off
$VsSetupBlock
cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" -A x64 -DDXC_ROOT="$DxcRoot" -DAGILITY_ROOT="$AgilityRoot" -DAGILITY_SDK_VERSION=$AgilitySdkVersion
if errorlevel 1 exit /b %errorlevel%
cmake --build "$BuildDir" --config $Config --parallel
if errorlevel 1 exit /b %errorlevel%
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

$ExePath = Join-Path $BuildDir "bin\$Config\hlsl_probe.exe"
if (Test-Path $ExePath) {
    Write-Host "Built: $ExePath"
}
