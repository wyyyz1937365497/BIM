"""3DGS → BIM Pipeline — Gradio Web UI.

Launch from bim-recon environment with vcvars64:
    cmd /c "call \"...\\vcvars64.bat\" && python scripts/gradio_app.py"

Tabs:
  1. 3DGS Viewer  — nerfview iframe (free scene exploration)
  2. Pipeline      — run pipeline, configure detection
  3. Results       — wall top-down, VLM gallery, seg overlay gallery, JSON report
  4. Seg Editor    — manual bbox adjustment for Falcon segmentation
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bim_recon.pipeline_api import PipelineResults, load_results

VIEWER_PORT = 8081


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def find_latest_output(scene_name: str) -> str:
    base = ROOT / "output" / scene_name
    if not base.exists():
        return ""
    dirs = sorted(base.iterdir(), reverse=True)
    return str(dirs[0]) if dirs else ""


def start_viewer(scene_name: str) -> str:
    subprocess.run(
        f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{VIEWER_PORT} ^| findstr LISTENING\') do taskkill /f /pid %a',
        shell=True, capture_output=True,
    )
    input_root = str(ROOT / "data" / scene_name / "preprocessed")
    feat_path = str(ROOT / "output" / scene_name / f"{scene_name}_feat.pt")
    subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_viewer.py"),
         "--input-root", input_root, "--feature-path", feat_path,
         "--port", str(VIEWER_PORT)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return (
        f'<iframe src="http://127.0.0.1:{VIEWER_PORT}" '
        f'style="width:100%;height:600px;border:none;"></iframe>'
        f'<p>Viewer starting on port {VIEWER_PORT}... Refresh if blank.</p>'
    )


def run_pipeline_cb(scene: str, doors: bool, windows: bool,
                    falcon: bool, skip_vlm: bool) -> tuple[str, str]:
    elems = []
    if doors:
        elems.append("door")
    if windows:
        elems.append("window")
    if not elems:
        return "Error: select at least one element type", ""
    args = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
            "--name", scene, "--elements", *elems]
    if skip_vlm:
        args.append("--skip-vlm")
    if not falcon:
        args.append("--no-falcon")
    try:
        subprocess.run(args, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return f"Pipeline failed: {e}", ""
    out = find_latest_output(scene)
    return f"Done. Output: {out}", out


def load_results_cb(out_dir: str) -> tuple:
    if not out_dir or not Path(out_dir).exists():
        return None, "No output found", [], [], None, None, gr.update(choices=[])

    res = load_results(Path(out_dir))
    topdown = res.wall_topdown_image or None
    vlm_imgs = [(e.image_path, f"{e.element_class} #{i}")
                for i, e in enumerate(res.doors + res.windows)
                if Path(e.image_path).exists()]
    seg_imgs = [(e.overlay_image, f"{e.element_class} #{i}")
                for i, e in enumerate(res.doors + res.windows)
                if e.overlay_image and Path(e.overlay_image).exists()]
    summary = (f"### Results\n- Walls: {len(res.walls)}\n"
               f"- Doors: {len(res.doors)} | Windows: {len(res.windows)}")
    labels = [f"{e.element_class} #{i}" for i, e in
              enumerate(res.doors + res.windows)]
    return res, summary, vlm_imgs, seg_imgs, topdown, res.report, gr.update(choices=labels)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="3DGS → BIM Pipeline") as app:
        gr.Markdown("# 3DGS → BIM Pipeline")
        results_state = gr.State(None)

        # --- Tab 1: 3DGS Viewer ---
        with gr.Tab("3DGS Viewer"):
            gr.Markdown("### 3DGS Scene Viewer")
            with gr.Row():
                viewer_scene = gr.Textbox(value="room0", label="Scene")
                viewer_btn = gr.Button("Start Viewer", variant="primary")
            viewer_html = gr.HTML("<p>Click 'Start Viewer'.</p>")
            viewer_btn.click(fn=start_viewer, inputs=viewer_scene, outputs=viewer_html)

        # --- Tab 2: Pipeline ---
        with gr.Tab("Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    pipe_scene = gr.Textbox(value="room0", label="Scene name")
                    cb_doors = gr.Checkbox(True, label="Doors")
                    cb_windows = gr.Checkbox(True, label="Windows")
                    cb_falcon = gr.Checkbox(True, label="Falcon Seg")
                    cb_skipvlm = gr.Checkbox(False, label="Skip VLM")
                    run_btn = gr.Button("Run Pipeline", variant="primary")
                with gr.Column(scale=1):
                    pipe_status = gr.Markdown("Ready.")
                    out_dir_box = gr.Textbox(label="Output dir", interactive=False)
                    load_btn = gr.Button("Load Results →")
            run_btn.click(
                fn=run_pipeline_cb,
                inputs=[pipe_scene, cb_doors, cb_windows, cb_falcon, cb_skipvlm],
                outputs=[pipe_status, out_dir_box],
            )

        # --- Tab 3: Results ---
        with gr.Tab("Results"):
            summary_md = gr.Markdown("Run pipeline → click 'Load Results'.")
            gr.Markdown("### Wall Lines (top-down)")
            wall_img = gr.Image(height=500)
            gr.Markdown("### VLM Verification Images")
            vlm_gallery = gr.Gallery(columns=4, height=400)
            gr.Markdown("### Seg Overlay Images")
            seg_gallery = gr.Gallery(columns=3, height=400)
            gr.Markdown("### Pipeline Report")
            report_json = gr.JSON()

        # --- Tab 4: Seg Editor ---
        with gr.Tab("Seg Editor"):
            gr.Markdown("### Manual BBox Adjustment\n"
                        "Select an element, adjust the normalized bbox, recalculate.")
            with gr.Row():
                elem_sel = gr.Dropdown(label="Element", choices=[], allow_custom_value=True)
                cx_sld = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Center X")
                cy_sld = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Center Y")
                w_sld = gr.Slider(minimum=0.01, maximum=1, value=0.3, step=0.01, label="Width")
                h_sld = gr.Slider(minimum=0.01, maximum=1, value=0.5, step=0.01, label="Height")
            remap_btn = gr.Button("Recalculate Wall Coords")
            remap_out = gr.JSON(label="Recalculated dimensions")

            def do_remap(elem_label: str, cx: float, cy: float,
                         w: float, h: float, results: PipelineResults | None):
                if results is None or not elem_label:
                    return {"error": "Load results and select an element first"}
                bbox = {"cx": cx, "cy": cy, "w": w, "h": h}
                # Find the element's original detection for comparison
                idx = int(elem_label.split("#")[-1]) if "#" in elem_label else 0
                all_elems = results.doors + results.windows
                if idx >= len(all_elems):
                    return {"error": f"Invalid index {idx}"}
                elem = all_elems[idx]
                orig = elem.height_detection or {}
                return {
                    "manual_bbox": bbox,
                    "original_method": orig.get("method", "N/A"),
                    "original_sill_m": orig.get("sill_height"),
                    "original_header_m": orig.get("header_height"),
                    "original_width_m": orig.get("width_m"),
                    "note": "Manual bbox in normalized coords (0-1). "
                            "Full wall-coord remap requires ElevationParams from pipeline.",
                }

            remap_btn.click(
                fn=do_remap,
                inputs=[elem_sel, cx_sld, cy_sld, w_sld, h_sld, results_state],
                outputs=remap_out,
            )

        # --- Wire load button ---
        load_btn.click(
            fn=load_results_cb,
            inputs=out_dir_box,
            outputs=[results_state, summary_md, vlm_gallery, seg_gallery,
                     wall_img, report_json, elem_sel],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_port=7860, server_name="127.0.0.1")
