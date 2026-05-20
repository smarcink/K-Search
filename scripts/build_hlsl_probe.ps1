param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Debug",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$SourceDir = Join-Path $RepoRoot "hlsl_probe"
$BuildDir = Join-Path $SourceDir "build"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
}

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe not found at $VsWhere"
}

$VsInstall = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsInstall) {
    throw "Visual Studio with VC x64 tools was not found."
}

$VsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
    throw "VsDevCmd.bat not found at $VsDevCmd"
}

$CmdFile = Join-Path ([System.IO.Path]::GetTempPath()) "build_hlsl_probe_$PID.cmd"
@"
@echo off
call "$VsDevCmd" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
cmake -S "$SourceDir" -B "$BuildDir" -G "Visual Studio 17 2022" -A x64
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
