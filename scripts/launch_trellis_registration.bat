@echo off
REM === Focused TRELLIS Registration Lab ===
REM Separate from the main BIM Gradio page. Uses port 19256.

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set PYTHON=G:\Miniconda3\envs\bim-recon\python.exe
cd /d G:\TJ\BIM
%PYTHON% scripts\gradio_trellis_registration.py
pause
