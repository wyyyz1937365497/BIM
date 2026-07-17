@echo off
REM === 3DGS → BIM Gradio App Launcher ===
REM Launches the Gradio web UI with vcvars64 for gsplat JIT support.

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ==========================================
echo   3DGS to BIM Pipeline - Web UI
echo   URL: http://127.0.0.1:19255
echo ==========================================
echo.

set PYTHON=G:\Miniconda3\envs\bim-recon\python.exe
%PYTHON% scripts\gradio_app.py

pause
