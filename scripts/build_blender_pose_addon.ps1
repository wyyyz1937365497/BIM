param(
    [string]$Blender = "F:\Blender\Blender4.3\blender.exe",
    [string]$OutputDir = "output\blender_extensions"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "blender_addons\bim_pose_annotation"
$Output = Join-Path $Root $OutputDir
New-Item -ItemType Directory -Force -Path $Output | Out-Null

& $Blender --command extension build --source-dir $Source --output-dir $Output
if ($LASTEXITCODE -ne 0) {
    throw "Blender extension build failed with exit code $LASTEXITCODE"
}
