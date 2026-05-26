function Get-VsWhereCandidates {
    $Candidates = @()
    $PathCommand = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($PathCommand) {
        $Candidates += $PathCommand.Source
    }

    foreach ($BasePath in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($BasePath) {
            $Candidates += Join-Path $BasePath "Microsoft Visual Studio\Installer\vswhere.exe"
        }
    }

    $Candidates | Where-Object { $_ } | Select-Object -Unique
}

function Get-VsInstallCandidates {
    $Candidates = @()
    if ($env:VSINSTALLDIR) {
        $Candidates += $env:VSINSTALLDIR
    }

    foreach ($BasePath in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $BasePath) {
            continue
        }

        foreach ($Edition in @("BuildTools", "Community", "Professional", "Enterprise", "Preview")) {
            $Candidates += Join-Path $BasePath "Microsoft Visual Studio\2022\$Edition"
        }
    }

    $Candidates | Where-Object { $_ } | Select-Object -Unique
}

function Get-VsDevCmdPath {
    foreach ($VsWhere in @(Get-VsWhereCandidates)) {
        if (-not (Test-Path -LiteralPath $VsWhere)) {
            continue
        }

        $InstallPaths = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($LASTEXITCODE -ne 0) {
            continue
        }

        foreach ($InstallPath in @($InstallPaths)) {
            if (-not $InstallPath) {
                continue
            }

            $Candidate = Join-Path $InstallPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path -LiteralPath $Candidate) {
                return $Candidate
            }
        }
    }

    foreach ($InstallPath in @(Get-VsInstallCandidates)) {
        $Candidate = Join-Path $InstallPath "Common7\Tools\VsDevCmd.bat"
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }

    return $null
}

function New-VsDevCmdSetupBlock {
    $VsDevCmd = Get-VsDevCmdPath
    if ($VsDevCmd) {
        return "call `"$VsDevCmd`" -arch=x64 -host_arch=x64`r`nif errorlevel 1 exit /b %errorlevel%"
    }

    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return "rem Visual Studio developer environment already active"
    }

    $VsWhereSearch = @(Get-VsWhereCandidates) -join "`n  "
    $VsInstallSearch = @(Get-VsInstallCandidates) -join "`n  "

    throw @"
Visual Studio 2022 C++ build tools were not found.

Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload, or run this script from an x64 Visual Studio developer command prompt.

Searched for vswhere.exe at:
  $VsWhereSearch

Searched for VsDevCmd.bat under:
  $VsInstallSearch
"@
}