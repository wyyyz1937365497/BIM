@echo off
REM === TRELLIS Mesh Generator Server Launcher ===
REM Runs the TRELLIS HTTP bridge in the trellis conda environment.
REM Auto-applies xformers Windows patches before starting.

echo.
echo ==========================================
echo   TRELLIS Mesh Generator Server
echo   URL: http://127.0.0.1:18391
echo ==========================================
echo.

REM --- Apply xformers patches (idempotent) ---
cd /d G:\TJ\BIM\TRELLIS
git apply --check ..\trellis_server\xformers_windows.patch 2>nul
if %ERRORLEVEL%==0 (
    echo Applying xformers Windows patches...
    git apply ..\trellis_server\xformers_windows.patch
    if %ERRORLEVEL%==0 (
        echo Patches applied successfully.
    ) else (
        echo WARNING: Patch apply failed, continuing anyway.
    )
) else (
    echo Patches already applied or not needed, skipping.
)

REM --- Launch server ---
cd /d G:\TJ\BIM
set PYTHON=G:\Miniconda3\envs\trellis\python.exe
set TRELLIS_MODEL=G:/TJ/BIM/TRELLIS/TRELLIS-image-large
%PYTHON% trellis_server\server.py --host 127.0.0.1 --port 18391 --model %TRELLIS_MODEL%

pause
