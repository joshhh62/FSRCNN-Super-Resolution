from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from utils.image_processing import (
    ImageProcessingError,
    nearest_upscale,
    build_download_name,
    decode_uploaded_image,
    encode_png,
    post_enhance,
    run_fsrcnn_inference,
)
from utils.loader import ModelLoadError, get_default_model_path, load_fsrcnn_model
from utils.logging_utils import get_logger
from utils.metrics import compute_metrics


PAGE_TITLE = "FSRCNN Super-Resolution Studio"
SCALE_OPTIONS = (2, 4)
logger = get_logger()


def inject_styles() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>FSRCNN Image Super-Resolution</h1>
            <p>
                Upload a low-resolution image, enhance it with a pre-trained FSRCNN neural network,
                inspect sharpness and quality metrics, and compare the result side-by-side with the original input.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_card(title: str, badge: str, image, caption: str) -> None:
    st.markdown(
        f"""
        <div class="image-card">
            <div class="card-header">
                <p class="card-title">{title}</p>
                <span class="card-badge">{badge}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if image is None:
        st.markdown('<div class="placeholder">Run FSRCNN to generate this view.</div>', unsafe_allow_html=True)
    else:
        st.image(image, use_container_width=True)
    st.markdown(f'<p class="card-meta">{caption}</p></div>', unsafe_allow_html=True)


def format_psnr(value: float) -> str:
    return "∞ dB" if value == float("inf") else f"{value:.2f} dB"


def render_metric_grid(metrics: dict[str, float]) -> None:
    st.markdown('<div class="section-title">Quality Metrics</div>', unsafe_allow_html=True)

    metric_cards = [
        {
            "label": "PSNR",
            "value": format_psnr(metrics["psnr"]),
            "caption": "Peak signal-to-noise ratio of FSRCNN output vs. nearest-neighbour upscaled input.",
        },
        {
            "label": "SSIM",
            "value": f'{metrics["ssim"]:.4f}',
            "caption": "Structural similarity index — how much structure FSRCNN has changed vs. the input.",
        },
        {
            "label": "Sharpness Gain",
            "value": f'{metrics["sharpness_ratio"]:.2f}×',
            "caption": "Laplacian variance ratio: FSRCNN ÷ input sharpness. Values >1 mean enhanced detail.",
        },
        {
            "label": "FSRCNN Sharpness",
            "value": f'{metrics["sharpness_fsrcnn"]:.0f}',
            "caption": "Absolute Laplacian variance of the FSRCNN output. Higher = more fine detail recovered.",
        },
    ]

    columns = st.columns(4)
    for column, card in zip(columns, metric_cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-label">{card["label"]}</p>
                    <p class="metric-value">{card["value"]}</p>
                    <p class="metric-caption">{card["caption"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "PSNR and SSIM are computed against the nearest-neighbour upscaled input — not a ground-truth HR image. "
        "Sharpness Gain is a no-reference metric: values above 1× confirm FSRCNN recovered high-frequency detail."
    )


def render_before_after_slider(before_image_bytes: bytes, after_image_bytes: bytes) -> None:
    before_b64 = base64.b64encode(before_image_bytes).decode("utf-8")
    after_b64 = base64.b64encode(after_image_bytes).decode("utf-8")
    with Image.open(io.BytesIO(before_image_bytes)) as _im:
        iw, ih = _im.size
    ar = iw / ih if ih else 1.0
    frame_height = min(660, int(900 / ar) + 90)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:transparent;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
body{{padding:6px 6px 0}}

.stage{{display:flex;flex-direction:column;gap:10px}}

.shell{{
  position:relative;
  width:100%;
  aspect-ratio:{ar:.6f};
  max-height:560px;
  margin:0 auto;
  border-radius:14px;
  overflow:hidden;
  cursor:ew-resize;
  touch-action:none;
  user-select:none;
  background:#0a0f1a;
  box-shadow:0 0 0 1px rgba(255,255,255,0.09),0 20px 60px rgba(0,0,0,0.6);
}}

.cmp-img{{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  object-fit:contain;
  pointer-events:none;
  display:block;
}}
.cmp-before{{
  z-index:2;
  clip-path:inset(0 50% 0 0);
  will-change:clip-path;
}}

.lbl{{
  position:absolute;
  top:12px;
  z-index:6;
  padding:4px 10px;
  border-radius:6px;
  font-size:10px;
  font-weight:700;
  letter-spacing:.1em;
  text-transform:uppercase;
  pointer-events:none;
  transition:opacity .2s;
  backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);
}}
.lbl-l{{
  left:12px;
  background:rgba(15,23,42,.82);
  color:#94a3b8;
  border:1px solid rgba(148,163,184,.22);
}}
.lbl-r{{
  right:12px;
  background:rgba(6,54,42,.78);
  color:#6ee7b7;
  border:1px solid rgba(110,231,183,.32);
}}

.divider{{
  position:absolute;
  top:0;bottom:0;
  left:50%;
  width:0;
  z-index:10;
  pointer-events:none;
}}
.div-line{{
  position:absolute;
  top:0;bottom:0;
  left:0;
  width:2px;
  transform:translateX(-50%);
  background:rgba(255,255,255,.9);
  box-shadow:0 0 6px rgba(255,255,255,.55),0 0 20px rgba(255,255,255,.18);
}}
.div-pct{{
  position:absolute;
  top:10px;
  left:0;
  transform:translateX(-50%);
  background:rgba(8,14,28,.88);
  color:#e2e8f0;
  font-size:10px;
  font-weight:700;
  letter-spacing:.05em;
  padding:3px 8px;
  border-radius:5px;
  border:1px solid rgba(255,255,255,.16);
  white-space:nowrap;
  opacity:0;
  transition:opacity .15s;
  backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);
}}
.handle{{
  position:absolute;
  top:50%;left:0;
  transform:translate(-50%,-50%);
  width:48px;height:48px;
  border-radius:50%;
  background:rgba(8,15,30,.88);
  border:2px solid rgba(255,255,255,.88);
  display:flex;align-items:center;justify-content:center;
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  box-shadow:
    0 0 0 4px rgba(255,255,255,.07),
    0 8px 28px rgba(0,0,0,.55),
    0 0 16px rgba(255,255,255,.07);
  transition:box-shadow .12s,transform .12s;
  pointer-events:auto;
  cursor:ew-resize;
  outline:none;
}}
.handle:focus-visible{{
  box-shadow:0 0 0 3px rgba(99,179,237,.7),0 8px 28px rgba(0,0,0,.55);
}}
.handle svg{{width:22px;height:14px;pointer-events:none;flex-shrink:0}}

.shell.active .handle{{
  transform:translate(-50%,-50%) scale(1.13);
  box-shadow:
    0 0 0 4px rgba(255,255,255,.14),
    0 12px 40px rgba(0,0,0,.65),
    0 0 28px rgba(255,255,255,.13);
}}
.shell.active .div-pct{{opacity:1}}

.hint{{
  text-align:center;
  color:#475569;
  font-size:11.5px;
  padding-bottom:2px;
  letter-spacing:.01em;
  transition:opacity .5s;
}}
</style>
</head>
<body>
<div class="stage">
  <div class="shell" id="shell">

    <!-- FSRCNN output — full-size base layer (right/after side) -->
    <img class="cmp-img cmp-after"
         src="data:image/png;base64,{after_b64}"
         alt="FSRCNN output" />

    <!-- Original input — clipped by clip-path (left/before side) -->
    <img class="cmp-img cmp-before" id="cmp-before"
         src="data:image/png;base64,{before_b64}"
         alt="Original input" />

    <!-- Side labels -->
    <div class="lbl lbl-l" id="lbl-l">Original</div>
    <div class="lbl lbl-r" id="lbl-r">FSRCNN</div>

    <!-- Divider + handle -->
    <div class="divider" id="divider">
      <div class="div-line"></div>
      <div class="div-pct" id="div-pct">50%</div>
      <div class="handle" id="handle"
           tabindex="0" role="slider"
           aria-label="Original vs FSRCNN comparison"
           aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
        <svg viewBox="0 0 22 14" fill="none"
             stroke="rgba(255,255,255,.92)" stroke-width="2.3"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="7,1 2,7 7,13"/>
          <polyline points="15,1 20,7 15,13"/>
        </svg>
      </div>
    </div>

  </div>
  <p class="hint" id="hint">Drag left / right to compare &nbsp;&middot;&nbsp; Arrow keys also work &nbsp;&middot;&nbsp; Shift + Arrow = 10 %</p>
</div>

<script>
(function(){{
  const shell   = document.getElementById("shell");
  const imgB    = document.getElementById("cmp-before");
  const divider = document.getElementById("divider");
  const handle  = document.getElementById("handle");
  const pct     = document.getElementById("div-pct");
  const hint    = document.getElementById("hint");
  const lblL    = document.getElementById("lbl-l");
  const lblR    = document.getElementById("lbl-r");

  let pos = 50, dragging = false, raf = null, interacted = false;
  const clamp = (v,a,b) => Math.min(Math.max(v,a),b);

  function apply(p) {{
    pos = clamp(p, 0, 100);
    const right = (100 - pos).toFixed(3);
    imgB.style.clipPath        = `inset(0 ${{right}}% 0 0)`;
    divider.style.left         = `${{pos.toFixed(3)}}%`;
    pct.textContent            = `${{Math.round(pos)}}%`;
    handle.setAttribute("aria-valuenow", String(Math.round(pos)));
    lblL.style.opacity = pos < 14 ? (pos / 14).toFixed(3) : "1";
    lblR.style.opacity = pos > 86 ? ((100 - pos) / 14).toFixed(3) : "1";
  }}

  function fromEvent(cx) {{
    const r = shell.getBoundingClientRect();
    apply(((cx - r.left) / r.width) * 100);
  }}

  shell.addEventListener("pointerdown", (e) => {{
    shell.setPointerCapture(e.pointerId);
    dragging = true;
    shell.classList.add("active");
    fromEvent(e.clientX);
    if (!interacted) {{ interacted = true; hint.style.opacity = "0"; }}
  }});

  shell.addEventListener("pointermove", (e) => {{
    if (!dragging) return;
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => fromEvent(e.clientX));
  }});

  function stopDrag(e) {{
    if (!dragging) return;
    dragging = false;
    shell.classList.remove("active");
    if (e && e.pointerId != null) shell.releasePointerCapture(e.pointerId);
  }}
  shell.addEventListener("pointerup",     stopDrag);
  shell.addEventListener("pointercancel", stopDrag);

  handle.addEventListener("keydown", (e) => {{
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft")  {{ e.preventDefault(); apply(pos - step); }}
    if (e.key === "ArrowRight") {{ e.preventDefault(); apply(pos + step); }}
    if (e.key === "Home")       {{ e.preventDefault(); apply(0);   }}
    if (e.key === "End")        {{ e.preventDefault(); apply(100); }}
  }});

  apply(50);
}})();
</script>
</body>
</html>"""
    components.html(html, height=frame_height)


def render_model_status(model_path: str | None, device_name: str | None) -> None:
    st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)

    if not model_path:
        st.info("FSRCNN was not executed for this run.")
        return

    st.markdown(
        f"""
        <div class="download-card">
            <strong>Checkpoint</strong><br/>
            <code>{model_path}</code><br/><br/>
            <strong>Device</strong>: {device_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("### Controls")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Drag and drop a JPG or PNG image here.",
        )
        scale_factor = st.selectbox("Scale factor", SCALE_OPTIONS, index=0)
        post_process = st.checkbox(
            "Post-processing (CLAHE + Sharpen)",
            value=True,
            help="Applies adaptive contrast enhancement and unsharp masking to recover detail softened by the model.",
        )
        process_clicked = st.button("Enhance Image", use_container_width=True)

        st.markdown(
            """
            <div class="notice-card">
                <strong>Model notes</strong><br/>
                FSRCNN is a verified 2x checkpoint. The 4x option chains the same model twice
                with intermediate sharpening between passes.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return uploaded_file, scale_factor, post_process, process_clicked


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🖼️", layout="wide")
    inject_styles()
    render_header()

    uploaded_file, scale_factor, post_process, process_clicked = render_sidebar()

    st.markdown('<div class="section-title">Comparison Workspace</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown(
            """
            <div class="notice-card">
                Drag and drop a JPG or PNG image in the sidebar, choose a scale factor, and run enhancement.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if process_clicked or "latest_result" not in st.session_state:
        try:
            file_bytes = uploaded_file.getvalue()
            logger.info(
                "Enhancement requested for '%s' scale=%dx",
                uploaded_file.name,
                scale_factor,
            )
            with st.spinner("Running FSRCNN inference..."):
                processed_upload = decode_uploaded_image(file_bytes, uploaded_file.name, scale_factor)
                original_image = processed_upload.image
                processed_notice = processed_upload.notice
                logger.info(
                    "Image ready: name=%s shape=%s notice=%s",
                    uploaded_file.name,
                    original_image.shape,
                    processed_notice or "none",
                )

                model, device = load_fsrcnn_model()
                inference_start = time.perf_counter()
                fsrcnn_image = run_fsrcnn_inference(
                    image=original_image,
                    model=model,
                    device=device,
                    scale_factor=scale_factor,
                    post_process=post_process,
                )
                inference_duration = time.perf_counter() - inference_start
                logger.info(
                    "FSRCNN inference done in %.3fs on %s, output shape=%s",
                    inference_duration,
                    device.type.upper(),
                    fsrcnn_image.shape,
                )

                original_upscaled = nearest_upscale(original_image, scale_factor)
                metrics = compute_metrics(original_upscaled, fsrcnn_image)

                post_label = " + post-processing" if post_process else ""
                timing_label = (
                    f"FSRCNN inference{post_label} completed in {inference_duration:.3f}s on {device.type.upper()}"
                )

                st.session_state["latest_result"] = {
                    "original": original_image,
                    "original_upscaled": original_upscaled,
                    "fsrcnn": fsrcnn_image,
                    "metrics": metrics,
                    "scale_factor": scale_factor,
                    "processed_notice": processed_notice,
                    "timing": timing_label,
                    "download_bytes": encode_png(fsrcnn_image),
                    "download_name": build_download_name(uploaded_file.name),
                    "model_path": str(get_default_model_path()),
                    "device": device.type.upper(),
                }
            logger.info(
                "Enhancement completed for '%s'. Download artifact: %s",
                uploaded_file.name,
                build_download_name(uploaded_file.name),
            )
            st.success("Enhancement completed successfully.")
        except (ImageProcessingError, ModelLoadError) as exc:
            st.session_state.pop("latest_result", None)
            logger.exception("Enhancement failed for '%s'", uploaded_file.name)
            st.error(str(exc))
            return
        except Exception as exc:
            st.session_state.pop("latest_result", None)
            logger.exception("Unexpected runtime error for '%s'", uploaded_file.name)
            st.error(f"Unexpected runtime error during enhancement: {exc}")
            return

    latest_result = st.session_state.get("latest_result")
    if latest_result is None:
        return

    if latest_result["processed_notice"]:
        st.warning(latest_result["processed_notice"])

    st.markdown(
        f"""
        <div class="download-card">
            <strong>Processing summary</strong><br/>
            Scale: {latest_result["scale_factor"]}x &nbsp;&middot;&nbsp; {latest_result["timing"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        render_image_card(
            title="Original Input",
            badge="Input",
            image=latest_result["original"],
            caption="Validated upload after safe RGB conversion and optional memory guard.",
        )
    with col2:
        render_image_card(
            title="FSRCNN Output",
            badge=f'{latest_result["scale_factor"]}x',
            image=latest_result["fsrcnn"],
            caption="Neural super-resolution output from the pre-trained FSRCNN model.",
        )

    st.markdown('<div class="section-title">Before / After Slider</div>', unsafe_allow_html=True)
    render_before_after_slider(
        before_image_bytes=encode_png(latest_result["original_upscaled"]),
        after_image_bytes=encode_png(latest_result["fsrcnn"]),
    )

    render_metric_grid(metrics=latest_result["metrics"])

    render_model_status(
        model_path=latest_result["model_path"],
        device_name=latest_result["device"],
    )

    st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="download-card">
            Export the FSRCNN enhanced result as a PNG file.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        label="Download Enhanced Image",
        data=latest_result["download_bytes"],
        file_name=latest_result["download_name"],
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
