<#
.SYNOPSIS
    Build and deploy mcp-servers-for-revit to Revit 2026 Addins folder.

.DESCRIPTION
    This script automates the full build → deploy cycle for the
    mcp-servers-for-revit plugin. It:
      1. Kills Revit (if running) to release DLL locks
      2. Builds both RevitMCPPlugin and RevitMCPCommandSet
      3. Copies the output to the Revit Addins folder
      4. Optionally relaunches Revit

    Only Revit 2026 (R26, .NET 8) is supported on this machine.

.PARAMETER Configuration
    Build configuration: "Debug R26" (default, with .pdb + auto-deploy)
    or "Release R26" (optimized, requires manual copy).

.PARAMETER SkipKillRevit
    Skip the "kill Revit" step. Use if Revit is not running.

.PARAMETER SkipLaunch
    Don't relaunch Revit after deploy.

.PARAMETER Clean
    Rebuild from scratch (dotnet build --no-incremental).

.EXAMPLE
    .\scripts\build_deploy_revit.ps1
    # Debug build + deploy + relaunch Revit

.EXAMPLE
    .\scripts\build_deploy_revit.ps1 -Configuration "Release R26"
    # Release build + deploy

.EXAMPLE
    .\scripts\build_deploy_revit.ps1 -Clean
    # Full clean rebuild
#>

param(
    [string]$Configuration = "Debug R26",
    [switch]$SkipKillRevit,
    [switch]$SkipLaunch,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

$RevitYear   = "2026"
$RevitPath   = "F:\Software\AutoDesk\Revit 2026\Revit.exe"
$RepoRoot    = "G:\TJ\BIM\mcp-servers-for-revit"
$Solution    = "$RepoRoot\mcp-servers-for-revit.sln"
$ServerDir   = "$RepoRoot\server"
$AddinsDir   = "$env:APPDATA\Autodesk\Revit\Addins\$RevitYear"
$StagingDir  = "$RepoRoot\plugin\bin\AddIn $RevitYear $Configuration"

# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------

Write-Host "`n=== mcp-servers-for-revit Build & Deploy ===" -ForegroundColor Cyan
Write-Host "  Configuration : $Configuration"
Write-Host "  Revit Year    : $RevitYear"
Write-Host "  Staging       : $StagingDir"
Write-Host "  Addins target : $AddinsDir"
Write-Host ""

if (-not (Test-Path $Solution)) {
    Write-Error "Solution not found: $Solution"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 1: Kill Revit (release DLL locks)
# ---------------------------------------------------------------------------

if (-not $SkipKillRevit) {
    $revitProc = Get-Process -Name "Revit" -ErrorAction SilentlyContinue
    if ($revitProc) {
        Write-Host "[1/5] Stopping Revit (PID $($revitProc.Id))..." -ForegroundColor Yellow
        $revitProc | Stop-Process -Force
        Start-Sleep -Seconds 3
        Write-Host "      Done." -ForegroundColor Green
    } else {
        Write-Host "[1/5] Revit not running — skip." -ForegroundColor DarkGray
    }
} else {
    Write-Host "[1/5] Skip kill Revit (--SkipKillRevit)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Step 2a: Build C# (Plugin + CommandSet)
# ---------------------------------------------------------------------------

Write-Host "[2/5] Building C# $Configuration..." -ForegroundColor Yellow

# Always clean obj/ dirs to prevent stale WPF .g.cs files
Write-Host "      (cleaning obj/ dirs for WPF regeneration)" -ForegroundColor DarkYellow
foreach ($proj in @("plugin", "commandset")) {
    $objDir = "$RepoRoot\$proj\obj"
    if (Test-Path $objDir) { Remove-Item $objDir -Recurse -Force }
}
if ($Clean) {
    Write-Host "      (-Clean: also removing bin/ dirs)" -ForegroundColor DarkYellow
    foreach ($proj in @("plugin", "commandset")) {
        $binDir = "$RepoRoot\$proj\bin"
        if (Test-Path $binDir) { Remove-Item $binDir -Recurse -Force }
    }
}

$buildArgs = @("build", $Solution, "-c", $Configuration, "--no-incremental")
$buildStartTime = Get-Date
& dotnet @buildArgs
$buildExitCode = $LASTEXITCODE
$buildDuration = (Get-Date) - $buildStartTime

if ($buildExitCode -ne 0) {
    Write-Host "`n      BUILD FAILED (exit $buildExitCode, $($buildDuration.TotalSeconds.ToString('0.0'))s)" -ForegroundColor Red
    exit $buildExitCode
}

Write-Host "      C# build OK ($($buildDuration.TotalSeconds.ToString('0.0'))s)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 2b: Build TypeScript MCP Server
# ---------------------------------------------------------------------------

Write-Host "[3/5] Building TypeScript MCP Server..." -ForegroundColor Yellow

Push-Location $ServerDir
try {
    # Ensure node_modules exist
    if (-not (Test-Path "node_modules")) {
        Write-Host "      Installing npm dependencies..." -ForegroundColor DarkGray
        npm install --silent 2>&1 | Out-Null
    }

    $tsStartTime = Get-Date
    $tsResult = npm run build 2>&1
    $tsDuration = (Get-Date) - $tsStartTime
    $tsExit = $LASTEXITCODE

    if ($tsExit -ne 0) {
        Write-Host "      TS BUILD FAILED:" -ForegroundColor Red
        Write-Host $tsResult
        Pop-Location
        exit $tsExit
    }

    Write-Host "      TS build OK ($($tsDuration.TotalSeconds.ToString('0.0'))s)" -ForegroundColor Green
}
finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# Step 3: Deploy — copy staging → Addins folder
# ---------------------------------------------------------------------------

Write-Host "[4/5] Deploying to Addins..." -ForegroundColor Yellow

if (-not (Test-Path $StagingDir)) {
    Write-Error "Staging directory not found: $StagingDir`nBuild may have failed silently."
    exit 1
}

# Debug builds auto-copy to Addins via MSBuild targets, but we copy again
# to be safe (handles Release builds and partial failures).
$stagingItems = Get-ChildItem $StagingDir -Recurse
$copied = 0
$skipped = 0

foreach ($item in $stagingItems) {
    $relativePath = $item.FullName.Substring($StagingDir.Length)
    $targetPath = Join-Path $AddinsDir $relativePath

    if ($item.PSIsContainer) {
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        continue
    }

    # Create parent directory if needed
    $parentDir = Split-Path $targetPath -Parent
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    # Only copy if source is newer
    $needCopy = $true
    if (Test-Path $targetPath) {
        $sourceTime = $item.LastWriteTime
        $targetTime = (Get-Item $targetPath).LastWriteTime
        if ($sourceTime -le $targetTime) {
            $needCopy = $false
        }
    }

    if ($needCopy) {
        Copy-Item $item.FullName $targetPath -Force
        $copied++
    } else {
        $skipped++
    }
}

Write-Host "      Copied: $copied files, Skipped (up-to-date): $skipped files" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 4: Verify
# ---------------------------------------------------------------------------

Write-Host "[5/5] Verifying deployment..." -ForegroundColor Yellow

$pluginDll   = "$AddinsDir\revit_mcp_plugin\RevitMCPPlugin.dll"
$commandDll  = "$AddinsDir\revit_mcp_plugin\Commands\RevitMCPCommandSet\$RevitYear\RevitMCPCommandSet.dll"
$addinFile   = "$AddinsDir\mcp-servers-for-revit.addin"
$commandJson = "$AddinsDir\revit_mcp_plugin\Commands\RevitMCPCommandSet\command.json"

$checks = @(
    @{ Name = ".addin manifest";   Path = $addinFile },
    @{ Name = "Plugin DLL";         Path = $pluginDll },
    @{ Name = "CommandSet DLL";     Path = $commandDll },
    @{ Name = "command.json";       Path = $commandJson },
    @{ Name = "TS Server (index.js)"; Path = "$ServerDir\build\index.js" }
)

$allOk = $true
foreach ($check in $checks) {
    if (Test-Path $check.Path) {
        $size = (Get-Item $check.Path).Length
        $sizeKB = [math]::Round($size / 1024, 1)
        Write-Host "      OK  $($check.Name) ($sizeKB KB)" -ForegroundColor Green
    } else {
        Write-Host "      MISSING  $($check.Name)" -ForegroundColor Red
        $allOk = $false
    }
}

if ($allOk) {
    Write-Host "`n  Deployment successful!" -ForegroundColor Green
} else {
    Write-Host "`n  Deployment incomplete — some files missing." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Step 5: Relaunch Revit
# ---------------------------------------------------------------------------

if (-not $SkipLaunch) {
    if (Test-Path $RevitPath) {
        Write-Host "`n  Launching Revit..." -ForegroundColor Yellow
        Start-Process $RevitPath
        Write-Host "  Revit starting. Check the ribbon for mcp-servers-for-revit tab." -ForegroundColor Green
    } else {
        Write-Host "`n  Revit not found at $RevitPath — skip launch." -ForegroundColor DarkGray
    }
} else {
    Write-Host "`n  Skip launch (--SkipLaunch)" -ForegroundColor DarkGray
}

Write-Host "`nDone.`n" -ForegroundColor Cyan
