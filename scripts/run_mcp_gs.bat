@echo off
REM Launcher for the bim-recon 3DGS MCP server.
REM Sets up MSVC (gsplat JIT), activates the conda env, and starts the server.
REM
REM All arguments are forwarded to the server (%*), so you can pass any
REM combination of the flags below. Environment variables act as defaults.
REM
REM Geometry source (pick ONE):
REM   --demo                      Synthetic room (no data needed, for wiring tests)
REM   --ply <path>                nerfstudio-exported splat.ply
REM   --data-dir <dir>            SceneSplat .npy dir (coord/color/opacity/scale/quat)
REM                               Use this for SceneSplat data — the .npy files hold
REM                               real RGB (uint8), unlike the PCA-colored PLY export.
REM
REM Semantic features (all three required together, or omit all three):
REM   --feat <path>               SceneSplat feat.pt (N,768) per-Gaussian features
REM   --text-emb <path>           bim_text_emb.pt (C,768) SigLIP2 text embeddings
REM   --class-names <path>        bim_class_names.json {class_name: index}
REM
REM Optional:
REM   --cameras <path>            transforms.json for training camera list
REM   --width 800 --height 600   Default render resolution
REM
REM Equivalent environment variables (lower priority than CLI flags):
REM   GS_PLY_PATH, GS_DATA_DIR, GS_FEAT_PATH, GS_TEXT_EMB_PATH,
REM   GS_CLASS_NAMES_PATH, GS_CAMERAS_JSON
REM
REM Example — SceneSplat .npy data with semantics:
REM   run_mcp_gs.bat --data-dir G:\TJ\BIM\data ^
REM     --feat output\data_feat.pt ^
REM     --text-emb data\bim_text_emb.pt ^
REM     --class-names data\bim_class_names.json

call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call G:\Miniconda3\Scripts\activate.bat bim-recon
cd /d G:\TJ\BIM
python -m bim_recon.mcp_gs %*
