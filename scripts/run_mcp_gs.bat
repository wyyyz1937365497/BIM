@echo off
REM Launcher for the bim-recon 3DGS MCP server.
REM Sets up MSVC (gsplat JIT), activates the conda env, and starts the server.
REM
REM Usage from opencode.json:
REM   "command": ["G:\\TJ\\BIM\\scripts\\run_mcp_gs.bat", "--demo"]
REM Replace "--demo" with "--ply path\\to\\splat.ply" when trained data exists.

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call G:\Miniconda3\Scripts\activate.bat bim-recon
cd /d G:\TJ\BIM
python -m bim_recon.mcp_gs %*
