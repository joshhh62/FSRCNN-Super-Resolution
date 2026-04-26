# FSRCNN Super-Resolution Studio

A Streamlit web app that upscales low-resolution images using a pre-trained **FSRCNN** (Fast Super-Resolution Convolutional Neural Network) model. Upload an image, pick a scale factor, and get an enhanced result with quality metrics and an interactive before/after slider.

---

## Demo

| Original Input | FSRCNN 2× Output |
|:-:|:-:|
| ![input](outputs/model_check_input.png) | ![fsrcnn](outputs/model_check_fsrcnn.png) |

---

## Features

- **2× and 4× upscaling** — the verified 2× checkpoint is used directly; 4× chains two passes with intermediate sharpening.
- **Optional post-processing** — bilateral filter → CLAHE → unsharp mask → saturation boost, recovering visual quality lost in PSNR-biased training.
- **Interactive before/after slider** — drag the divider to compare original vs enhanced at full resolution.
- **Quality metrics panel** — PSNR, SSIM, Sharpness Gain, and absolute Laplacian sharpness.
- **Memory guard** — large inputs are automatically downscaled to keep inference safe (cap: 12 MP output).
- **One-click download** — export the enhanced result as a lossless PNG.

---

## Project Structure

```
fsrcnn-super-resolution/
│
├── app.py                      # Streamlit entry point — UI layout, state machine, render functions
│
├── model/
│   ├── fsrcnn.py               # FSRCNN PyTorch model definition (feature → shrink → map → expand → deconv)
│   └── fsrcnn1.pth             # Pre-trained 2× checkpoint (d=56, s=12, m=4)
│
├── utils/
│   ├── image_processing.py     # Full pipeline: decode → memory guard → pre-sharpen → inference → post-enhance
│   ├── loader.py               # Model loader with checkpoint format normalisation and device selection
│   ├── metrics.py              # PSNR, SSIM, and Laplacian sharpness computation
│   └── logging_utils.py        # Structured stdout logger (shared across modules)
│
├── assets/
│   └── styles.css              # Custom CSS for cards, metrics grid, hero banner, and badges
│
├── outputs/                    # Sample reference images
│   ├── model_check_input.png
│   ├── model_check_bicubic.png
│   └── model_check_fsrcnn.png
│
├── .streamlit/
│   └── config.toml             # Disables file-watcher (avoids inotify exhaustion in some environments)
│
├── requirements.txt
└── .gitignore
```

---

## How It Works — End-to-End Flow

```
User uploads image (JPG / PNG)
        │
        ▼
┌─────────────────────────────┐
│  1. Validate & Decode       │  extension check, size < 20 MB,
│     (image_processing.py)   │  EXIF rotation, force RGB
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. Memory Guard            │  if projected output > 12 MP,
│     (image_processing.py)   │  downscale input proportionally
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. Load FSRCNN Model       │  cached via @st.cache_resource;
│     (loader.py)             │  auto-selects CUDA or CPU
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  4. Pre-sharpen Input       │  light unsharp mask so the model
│     (image_processing.py)   │  sees crisper edges before inference
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. FSRCNN Inference        │  normalise → CHW tensor → model →
│     (fsrcnn.py)             │  clamp → HWC uint8
│                             │  (for 4×: run twice with
│                             │   CLAHE + unsharp between passes)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐   (skipped if checkbox is off)
│  6. Post-Processing         │  bilateral filter → CLAHE →
│     (image_processing.py)   │  unsharp mask → saturation boost
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  7. Compute Quality Metrics │  PSNR & SSIM vs nearest-neighbour
│     (metrics.py)            │  baseline; Laplacian sharpness gain
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  8. Render in Streamlit UI  │  side-by-side cards, drag slider,
│     (app.py)                │  metrics grid, download button
└─────────────────────────────┘
```

---

## FSRCNN Architecture

```
Input (LR image)
    │
    ▼  Conv2d(3→56, 5×5) + PReLU       — feature extraction
    ▼  Conv2d(56→12, 1×1) + PReLU      — shrink (dimensionality reduction)
    ▼  4× [Conv2d(12→12, 3×3) + PReLU] — non-linear mapping
    ▼  Conv2d(12→56, 1×1) + PReLU      — expand
    ▼  ConvTranspose2d(56→3, 9×9, stride=scale) — sub-pixel deconvolution
    │
Output (HR image — 2× or 4× the input dimensions)
```

Parameters: `d=56` feature maps, `s=12` shrink maps, `m=4` mapping layers.
The checkpoint (`fsrcnn1.pth`) is a verified 2× model. 4× is achieved by chaining the same weights twice.

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/dinu2oo6/fsrcnn-super-resolution.git
cd fsrcnn-super-resolution
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Requirements

| Package | Version |
|---|---|
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| streamlit | 1.44.1 |
| numpy | 1.26.4 |
| opencv-python | 4.10.0.84 |
| pillow | 10.4.0 |
| scikit-image | 0.25.2 |

Python 3.10+ recommended. GPU (CUDA) is automatically used when available; CPU fallback requires no configuration.

---

## Usage

1. Open the sidebar and upload a `.jpg`, `.jpeg`, or `.png` file.
2. Choose a **scale factor** — `2x` (native FSRCNN pass) or `4x` (two chained passes).
3. Toggle **Post-processing** on or off (recommended: on).
4. Click **Enhance Image**.
5. Inspect the side-by-side image cards and drag the **Before / After Slider** to compare.
6. Review the **Quality Metrics** panel for PSNR, SSIM, and Sharpness Gain.
7. Click **Download Enhanced Image** to save the result as a lossless PNG.

---

## Quality Metrics Explained

| Metric | What it measures |
|---|---|
| **PSNR** | Peak signal-to-noise ratio of the FSRCNN output vs. the nearest-neighbour upscaled baseline. Higher = less pixel-level error. |
| **SSIM** | Structural Similarity Index — how much structural information changed. Closer to 1.0 = more structure preserved. |
| **Sharpness Gain** | Laplacian variance of FSRCNN ÷ Laplacian variance of the baseline. Values > 1× confirm FSRCNN recovered high-frequency detail. |
| **FSRCNN Sharpness** | Absolute Laplacian variance of the FSRCNN output. Higher = finer detail. |

> PSNR and SSIM are computed against the nearest-neighbour upscaled input, **not** a ground-truth HR image. Sharpness Gain is a no-reference metric and is the most meaningful signal for perceived quality improvement.

---

## Post-Processing Pipeline

When the **Post-processing** checkbox is enabled, the FSRCNN output is passed through four stages designed to compensate for artefacts introduced by PSNR-biased training:

1. **Bilateral filter** — smooths ringing and banding while preserving hard edges.
2. **CLAHE** — Contrast Limited Adaptive Histogram Equalisation restores local contrast that MSE loss tends to flatten.
3. **Unsharp mask** — sharpens fine edges and textures.
4. **Saturation boost** — corrects mild colour desaturation caused by pixel-averaging in MSE optimisation.

---

## File Details

| File | Role |
|---|---|
| `app.py` | Streamlit entry point. Renders the sidebar controls, image cards, before/after slider (custom HTML/JS), metric grid, model status panel, and download button. Manages `st.session_state` for result caching across re-runs. |
| `model/fsrcnn.py` | Pure PyTorch `nn.Module`. No app-level imports — can be used standalone. |
| `model/fsrcnn1.pth` | Pre-trained 2× checkpoint. Expected keys: `feature`, `shrink`, `map`, `expand`, `deconv`. |
| `utils/image_processing.py` | Validates uploads, enforces the memory cap, runs pre-sharpening, drives model inference (including the chained 4× pass), and applies the post-processing pipeline. |
| `utils/loader.py` | Loads the checkpoint with format normalisation (`state_dict` key unwrapping, `module.` prefix stripping). Cached with `@st.cache_resource`. |
| `utils/metrics.py` | Stateless functions: `compute_psnr`, `compute_ssim`, `_laplacian_sharpness`, `compute_metrics`. |
| `utils/logging_utils.py` | Returns a named `logging.Logger` with a single stdout handler; safe to call multiple times (idempotent handler setup). |
| `assets/styles.css` | Scoped CSS for the hero card, image cards, metric cards, download card, notice card, and section titles. |
