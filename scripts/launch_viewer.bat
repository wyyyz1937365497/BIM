@echo off
REM === Standalone 3DGS Viewer ===
REM Usage: scripts\launch_viewer.bat <scene-name>

set SCENE=%~1
if "%SCENE%"=="" (
  echo Usage: scripts\launch_viewer.bat ^<scene-name^>
  exit /b 1
)

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set PYTHON=G:\Miniconda3\envs\bim-recon\python.exe

echo.
echo ==========================================
echo   Standalone 3DGS Viewer
echo   Scene: %SCENE%
echo   Viewer: http://127.0.0.1:18081
echo   Camera: http://127.0.0.1:18081/camera-state
echo ==========================================
echo.

%PYTHON% scripts\run_viewer.py --scene "%SCENE%" --port 18081
