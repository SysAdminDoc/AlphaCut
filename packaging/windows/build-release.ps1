#!/usr/bin/env pwsh
[CmdletBinding()]
param(
    [string]$Version,
    [string]$PyInstaller = "pyinstaller",
    [string]$InnoSetupCompiler = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    [switch]$SkipBuild,
    [switch]$NoClean,
    [switch]$SkipInstaller,
    [switch]$Sign,
    [string]$SignTool = $env:ALPHACUT_SIGNTOOL,
    [string]$SignCertThumbprint = $env:ALPHACUT_SIGN_CERT_SHA1,
    [string]$SignCertPath = $env:ALPHACUT_SIGN_CERT_PATH,
    [string]$SignCertPassword = $env:ALPHACUT_SIGN_CERT_PASSWORD,
    [string]$TimestampUrl = $(if ($env:ALPHACUT_TIMESTAMP_URL) { $env:ALPHACUT_TIMESTAMP_URL } else { "http://timestamp.digicert.com" })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$RepoRootPrefix = $RepoRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
$DistDir = Join-Path $RepoRoot "dist"
$InstallerDir = Join-Path $DistDir "installer"
$SpecPath = Join-Path $RepoRoot "AlphaCut-windows.spec"
$AppPath = Join-Path $RepoRoot "AlphaCut.py"

function Resolve-InRepoPath {
    param([Parameter(Mandatory)][string]$Path)
    $full = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
    if ($full -ne $RepoRoot -and -not $full.StartsWith($RepoRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside repository: $full"
    }
    return $full
}

function Remove-InRepoItem {
    param([Parameter(Mandatory)][string]$Path)
    $full = Resolve-InRepoPath $Path
    if (Test-Path -LiteralPath $full) {
        Remove-Item -LiteralPath $full -Recurse -Force
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    Write-Host ("> {0} {1}" -f $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

function Get-AlphaCutVersion {
    $source = [System.IO.File]::ReadAllText($AppPath)
    if ($source -match '__version__\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    throw "Unable to read __version__ from $AppPath"
}

function Find-SignTool {
    if (-not [string]::IsNullOrWhiteSpace($SignTool)) {
        $cmd = Get-Command $SignTool -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        if (Test-Path -LiteralPath $SignTool) { return (Resolve-Path -LiteralPath $SignTool).Path }
        throw "SignTool was configured but not found: $SignTool"
    }

    $pathCmd = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
    if ($pathCmd) { return $pathCmd.Source }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe",
        "${env:ProgramFiles}\Windows Kits\10\bin\*\x64\signtool.exe"
    )
    foreach ($pattern in $candidates) {
        $match = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) { return $match.FullName }
    }

    return $null
}

function Get-SignArguments {
    param([Parameter(Mandatory)][string]$Target)
    $args = @("sign", "/fd", "SHA256", "/tr", $TimestampUrl, "/td", "SHA256")
    if (-not [string]::IsNullOrWhiteSpace($SignCertThumbprint)) {
        $args += @("/sha1", $SignCertThumbprint)
    } elseif (-not [string]::IsNullOrWhiteSpace($SignCertPath)) {
        $args += @("/f", $SignCertPath)
        if (-not [string]::IsNullOrWhiteSpace($SignCertPassword)) {
            $args += @("/p", $SignCertPassword)
        }
    } else {
        $args += "/a"
    }
    $args += $Target
    return $args
}

function Quote-InnoSignPart {
    param([Parameter(Mandatory)][string]$Value)
    if ($Value -eq '$f') {
        return '"$f"'
    }
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Get-InnoSignCommand {
    param([Parameter(Mandatory)][string]$SignToolPath)
    $parts = @($SignToolPath, "sign", "/fd", "SHA256", "/tr", $TimestampUrl, "/td", "SHA256")
    if (-not [string]::IsNullOrWhiteSpace($SignCertThumbprint)) {
        $parts += @("/sha1", $SignCertThumbprint)
    } elseif (-not [string]::IsNullOrWhiteSpace($SignCertPath)) {
        $parts += @("/f", $SignCertPath)
        if (-not [string]::IsNullOrWhiteSpace($SignCertPassword)) {
            $parts += @("/p", $SignCertPassword)
        }
    } else {
        $parts += "/a"
    }
    $parts += '$f'
    return (($parts | ForEach-Object { Quote-InnoSignPart $_ }) -join " ")
}

function Invoke-CodeSign {
    param(
        [Parameter(Mandatory)][string]$SignToolPath,
        [Parameter(Mandatory)][string]$Target
    )
    Invoke-Native $SignToolPath (Get-SignArguments $Target)
}

function Get-Sha256Hex {
    param([Parameter(Mandatory)][string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $sha.ComputeHash($stream)
            return (($bytes | ForEach-Object { $_.ToString("x2") }) -join "")
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Write-ChecksumFiles {
    param([Parameter(Mandatory)][string[]]$Artifacts)
    $manifestPath = Join-Path $DistDir "SHA256SUMS.txt"
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($artifact in $Artifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "Artifact missing before checksum: $artifact"
        }
        $hash = Get-Sha256Hex $artifact
        $name = Split-Path -Leaf $artifact
        $line = "$hash  $name"
        $lines.Add($line)
        [System.IO.File]::WriteAllText("$artifact.sha256", "$line`n", [System.Text.UTF8Encoding]::new($false))
        Write-Host "SHA256 $name $hash"
    }
    [System.IO.File]::WriteAllText($manifestPath, (($lines -join "`n") + "`n"), [System.Text.UTF8Encoding]::new($false))
    Write-Host "Wrote $manifestPath"
}

Push-Location $RepoRoot
try {
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $Version = Get-AlphaCutVersion
    }

    $portableExe = Join-Path $DistDir "AlphaCut-windows.exe"
    $installerExe = Join-Path $InstallerDir "AlphaCut-Setup-$Version.exe"

    $signingRequested = $Sign.IsPresent -or
        (-not [string]::IsNullOrWhiteSpace($SignTool)) -or
        (-not [string]::IsNullOrWhiteSpace($SignCertThumbprint)) -or
        (-not [string]::IsNullOrWhiteSpace($SignCertPath)) -or
        ($env:ALPHACUT_SIGN -eq "1")

    $resolvedSignTool = $null
    if ($signingRequested) {
        $resolvedSignTool = Find-SignTool
        if (-not $resolvedSignTool) {
            throw "Signing was requested, but signtool.exe was not found."
        }
        Write-Host "Signing enabled with $resolvedSignTool"
    } else {
        Write-Warning "Signing not configured; unsigned artifacts and checksum files will be produced."
    }

    if (-not $SkipBuild.IsPresent) {
        if (-not $NoClean.IsPresent) {
            Remove-InRepoItem "build"
            Remove-InRepoItem "dist\AlphaCut-windows.exe"
            Remove-InRepoItem "dist\AlphaCut-windows.exe.sha256"
            Remove-InRepoItem "dist\SHA256SUMS.txt"
            Get-ChildItem -Path $InstallerDir -Filter "AlphaCut-Setup-*.exe" -ErrorAction SilentlyContinue | Remove-Item -Force
            Get-ChildItem -Path $InstallerDir -Filter "AlphaCut-Setup-*.exe.sha256" -ErrorAction SilentlyContinue | Remove-Item -Force
        }

        Invoke-Native $PyInstaller @("--noconfirm", "--clean", $SpecPath)
    }

    if (-not (Test-Path -LiteralPath $portableExe)) {
        throw "Portable executable was not produced: $portableExe"
    }

    if ($resolvedSignTool) {
        Invoke-CodeSign $resolvedSignTool $portableExe
    }

    $artifacts = @($portableExe)

    if (-not $SkipInstaller.IsPresent) {
        if (-not (Test-Path -LiteralPath $InnoSetupCompiler)) {
            throw "Inno Setup compiler not found: $InnoSetupCompiler"
        }
        New-Item -ItemType Directory -Path $InstallerDir -Force | Out-Null
        $innoArgs = @("/DAppVersion=$Version")
        if ($resolvedSignTool) {
            $innoArgs += "/Salphacut_signtool=$(Get-InnoSignCommand $resolvedSignTool)"
            $innoArgs += "/DInstallerSignTool=alphacut_signtool"
        }
        $innoArgs += (Join-Path $RepoRoot "packaging\windows\AlphaCut.iss")
        Invoke-Native $InnoSetupCompiler $innoArgs
        if (-not (Test-Path -LiteralPath $installerExe)) {
            throw "Installer was not produced: $installerExe"
        }
        $artifacts += $installerExe
    }

    Write-ChecksumFiles $artifacts
    Write-Host "Release artifacts ready for v$Version"
} finally {
    Pop-Location
}
