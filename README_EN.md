# AlphaGen Gizmo

<p align="center">
  <a href="README.md">한국어</a> | <b>English</b>
</p>

**AlphaGen** is a next-generation controllable video alpha generation system that integrates Vision-Language Model (VLM) object grounding, SAM2 high-precision segmentation, LTX-2.5 video diffusion (IC-LoRA), and CompMatte-based core/edge matte refinement.

Designed for production environments, AlphaGen runs natively as a Foundry Nuke Gizmo and features a dual-runtime architecture easily extensible to Web UIs (FastAPI/Gradio) and research benchmarks.

---

## 🏗️ Pipeline Architecture

$$\begin{matrix}
\text{Initial Frame + Natural Language Prompt} \\
\downarrow \\
\textbf{VLM Object Grounding (Gemini Flash)} - \text{2D BBox [ymin, xmin, ymax, xmax] \& semantic target label} \\
\downarrow \\
\textbf{Local SAM2.1 (Hiera-Large)} - \text{Reference frame 1.2x auto-expanded pixel-level initial target mask} \\
\downarrow \\
\textbf{Nuke Viewer Pre-verification} -
 \text{Instant preview in 'Target Selection (VLM)' view within 1–2 seconds} \\
\downarrow \\
\textbf{Target Conditioning (LTX Alpha)} - \text{Spatial Attention Mask / ROI BBox Crop + Prompt Embedding} \\
\downarrow \\
\textbf{LTX-2.5 IC-LoRA Inference} - \text{Multi-frame raw alpha generation (49-frame chunks / 17-frame cosine overlap)} \\
\downarrow \\
\textbf{CompMatte CPU Refinement} - \text{Density Curve + Core Solidification + Edge Restoration} \\
\downarrow \\
\textbf{Final Compositing Alpha \& RGBA Output}
\end{matrix}$$

---

## 📁 Project Structure

All obsolete prototype scripts and temporary experiment caches have been audited and removed, leaving a clean, production-ready codebase:

```text
AlphaGen/
├── gizmos/
│   └── AlphaGen.gizmo              # Refactored Nuke Gizmo (4-step unified single tab, native dividers, Stop button)
├── alphagen_nuke.py                # Nuke API & GUI bridge (async worker queue, frame I/O, process tree termination)
├── alphagen_vlm.py                 # VLM (Gemini Flash) object grounding & SAM2 segmentation module
├── alphagen_worker.py              # Standalone Python AI worker (isolated from Nuke GIL, 49-frame chunking, ROI stitch)
├── alphagen_matte.py               # CPU core/edge matte refinement engine (pure math / OpenCV)
├── load_alphagen.py                # One-click execution script for Nuke Script Editor
├── init.py / menu.py               # Automatic Nuke plugin path registration
├── experiment_config.json          # Model checkpoint and external Python runtime configuration
├── NUKE_README.md                  # Comprehensive manual for Nuke compositing artists
├── README.md                       # Project technical documentation (Korean)
├── README_EN.md                    # Project technical documentation (English - this file)
│
├── tools/                          # Essential execution and validation utilities
│   ├── run_experiment.py           # LTX inference runner (automated 49-frame chunk CLI argument injection)
│   ├── ic_lora_video_only.py       # Video-only IC-LoRA pipeline script
│   ├── ic_lora_video_only.provenance.json
│   └── check_gizmo_structure.py    # Headless Gizmo Tcl syntax and node connection validator
│
└── tests/                          # 100% headless unit test suite
    ├── test_matte.py               # Core/edge morphology and density curve validation
    ├── test_vlm.py                 # VLM coordinate scaling and SAM2 inference tests
    └── test_worker.py              # IPC, process tree cancellation, per-node isolation, and chunking tests
```

---

## 🎛️ Unified Linear Workflow in Nuke Gizmo

All tab-switching and collapsible toggle groups have been removed in favor of an intuitive, top-to-bottom **unified single-tab workflow** separated by visual native dividers:

1. **Step 1. TARGET SELECTION (VLM & SAM2)**:
   - **Target Prompt**: Natural language description of the subject (e.g., `"person in the center wearing brown jacket"`).
   - **Ref Frame**: Best reference frame where the subject is clearly visible (default: `1`).
   - **Mask Expansion**: Multiplier to automatically dilate the SAM mask boundaries (default: `1.2x`, ensuring full coverage for motion blur and fine hair).
   - **[🔍 Analyze Target (VLM)]**: Completes BBox grounding and SAM2 mask extraction in 1–2 seconds with 1.2x auto-expansion (automatically switches viewer on completion).
2. **Step 2. LTX ALPHA VIDEO GENERATION**:
   - **First / Last**: Frame range to generate. Generates full sequences (200, 300, 1000+ frames) without missing frames. The pipeline automatically divides long shots into 49-frame chunks with 17-frame equal-power cosine blending and handles boundary padding/cropping.
   - **[🚀 Generate Alpha Video]**: Dispatches background LTX video diffusion inference.
   - **[⏹ Stop Queue]**: Immediately terminates the background process tree (`taskkill /T /F`) and completely releases GPU VRAM.
   - **Live Progress Bar**: Displays real-time progress `[████████░░░░] 65% (LTX Denoising step 5/8)` and automatically connects output on completion.
3. **Step 3. MATTE REFINEMENT (CORE & EDGE)**:
   - Interactive sliders exposed directly on node properties:
     - `Density & Alpha Curve`: Background Cutoff, White Level, Alpha Gamma
     - `Core Solidification`: Core Threshold, Inset (px), Strength, Fill Small Holes
     - `Edge & Detail Restoration`: Safe Radius (px), Edge Restoration, Edge Feather
   - **[⚡ Update Matte (CPU)]**: Millisecond-level CPU re-calculation without re-running heavy GPU models.
4. **Step 4. OUTPUT & VIEW MODE**:
   - 7 real-time view modes (`Final Alpha`, `Raw LTX Alpha`, `Target Selection (VLM)`, `Core`, `Edge`, `Source with Alpha`, `Premultiplied RGBA`).
   - **[📂 Open Alpha Folder]**: Opens the referenced per-node alpha directory (`.../alpha/<NodeName>/`) directly in the OS file explorer.

---

## 👁️ View Modes

- `Final alpha`: Fully composited matte with core solidification, edge restoration, and garbage matte logic applied.
- `Raw LTX alpha`: Raw, unrefined alpha sequence generated directly by the LTX diffusion model.
- `Target Selection (VLM)`: Single-frame initial target mask from VLM + SAM2 (**for verification prior to generation**).
- `Core`: Solid core mask generated via erosion and hole-filling to eliminate interior transparency.
- `Edge`: Soft edge band retaining fine details and high-frequency hair strands with Gaussian falloff.
- `Source with alpha`: Original RGB footage combined with the computed alpha channel.
- `Premultiplied RGBA`: Premultiplied result ready for downstream compositing.

---

## 📥 Installation & Setup Guide

AlphaGen can be permanently installed into the standard Nuke plugin directory (`~/.nuke/`) or loaded dynamically.

### 1. Permanent Nuke Installation (Recommended)

1. Open **`init.py`** in your `.nuke` folder (`C:\Users\<Username>\.nuke\init.py`) and add:

   ```python
   # ~/.nuke/init.py
   import nuke
   nuke.pluginAddPath('./AlphaGen')
   ```

   *(Or use an absolute path: `nuke.pluginAddPath(r"C:\Users\<Username>\.nuke\AlphaGen")`)*

2. When Nuke starts:
   - [init.py](init.py) automatically adds the `gizmos/` folder and Python modules to Nuke's search path.
   - [menu.py](menu.py) creates a **✦ AlphaGen** entry in the left Nodes toolbar.
   - Clicking the toolbar icon instantly creates an `AlphaGen` Gizmo node.

> [!TIP]
> **Keyboard Shortcut (Optional)**:
> Register a hotkey (e.g., `Ctrl+Alt+A`) in `~/.nuke/menu.py`:
>
> ```python
> # ~/.nuke/menu.py
> toolbar = nuke.menu("Nodes")
> m = toolbar.findItem("AlphaGen") or toolbar.addMenu("AlphaGen")
> m.addCommand("AlphaGen", "import alphagen_nuke; alphagen_nuke.create()", "ctrl+alt+a")
> ```

---

### 2. Requirements & Dependencies

In your external Python environment (system Python or existing AI virtual environment), install the required packages:

```powershell
cd C:\Users\<Username>\.nuke\AlphaGen
pip install -r requirements.txt
```

> [!TIP]
> **Reusing Existing AI Environments**:
> If you already have PyTorch and CUDA configured (e.g., ComfyUI at `D:/AI/ComfyUI_DATA/.venv`), you can point AlphaGen to it without setting up a new virtual environment.

Key dependency summary:
- **Deep Learning**: `torch>=2.1.0`, `torchvision>=0.16.0` (CUDA enabled)
- **Vision & SAM2 Models**: `transformers>=4.45.0` (with `Sam2Model`), `accelerate>=0.26.0`, `safetensors>=0.4.0`
- **Video & Image I/O**: `opencv-python>=4.8.0` (lossless FFV1 codec support), `Pillow>=10.0.0`, `numpy>=1.24.0`
- **API & Networking**: `requests>=2.28.0`

---

### 3. Environment Variables

Configure environment variables so AlphaGen can locate external APIs and GPU runtimes:

| Variable | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| **`GEMINI_API_KEY`** | **Required** | Google AI Studio API key for VLM 2D bounding-box spatial grounding | `AIzaSyD...` |
| **`ALPHAGEN_PYTHON`** | Optional | Path to external Python executable running AI models (auto-detects ComfyUI venvs or system Python) | `D:\AI\ComfyUI_DATA\.venv\Scripts\python.exe` |
| **`ALPHAGEN_MODELS_DIR`** | Optional | Directory containing LTX checkpoints (auto-scans common drive paths if unset) | `D:\AI\Models` |
| **`ALPHAGEN_CACHE_DIR`** | Optional | Custom cache root (defaults to Nuke's `localCachePath` / `NUKE_TEMP_DIR` / OS temp) | `D:\NukeCache\AlphaGen` |

> [!NOTE]
> **Setting the API Key**:
> Obtain a free key from [Google AI Studio](https://aistudio.google.com/), and add it to Windows System Environment Variables or `~/.nuke/init.py`:
>
> ```python
> import os
> os.environ["GEMINI_API_KEY"] = "AIzaSy..."
> ```

---

### 4. Verification & Testing

Run the automated test suite in a terminal to confirm full engine compatibility:

```powershell
python -m unittest discover -s tests -v
# Result: Ran 27 tests ... OK
```

---

## 📁 Output File Storage & Node Isolation

Generated alpha sequences are saved in an **`alpha/<NodeName>/` subfolder alongside the original source media**, ensuring full isolation between multiple `AlphaGen` nodes within the same Nuke script:

```text
[Source Media Directory]/
├── source_plate.%04d.exr           # Original plate sequence (or clip.mp4)
│
└── alpha/                          # Root alpha folder created alongside source
    └── <NodeName>/                 # ★ Isolated per AlphaGen node instance (e.g. AlphaGen1)
        ├── final/
        │   └── alpha.%06d.png      # ★ [Final Alpha] 16-bit lossless matte (auto-loaded by Read_final)
        ├── raw/
        │   └── alpha.%06d.png      # Raw LTX diffusion matte sequence (auto-loaded by Read_raw)
        ├── vlm_initial_mask.png    # SAM2 1.2x expanded target mask (for pre-verification)
        ├── vlm_selection.json      # Bounding box coordinates and confidence metadata
        └── result.json             # Execution job manifest
```

> [!NOTE]
> - **Clean 2-Folder Scheme (`final/` & `raw/`)**: Project directories stay clean with only the essential raw and final mattes.
> - **Per-Node Isolation**: Renaming or duplicating `AlphaGen` nodes in Nuke (e.g., `AlphaGen_Hero`, `AlphaGen_BG`) routes outputs to separate subdirectories, preventing file collisions.
> - **Cache Isolation**: Heavy intermediate tensors (`*.decoded.pt`) and attention videos are routed to Nuke's official cache directory (`%TEMP%/nuke/AlphaGen/<job_id>/`), leaving zero footprint in project or plugin folders.
> - **Automatic Reload**: Internal Read nodes automatically point to the corresponding files upon generation completion. Virtual nodes (e.g., `Constant`) safely fallback to Nuke cache storage.
> - **Windows File Lock Defense**: Implements atomic write and lock-retry logic to prevent Windows access violations (`WinError 5/32`) during concurrent polling.

---

## ⚙️ Hardware & Model Specifications

- **Recommended GPU**: NVIDIA RTX 4080 (16GB VRAM) or higher
- **Runtime**: Python 3.10 ~ 3.12 (Verified: Python 3.12, PyTorch 2.10+cu130)
- **LTX Model Checkpoints (Searched under `<MODELS_DIR>/`)**:
  - `LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
  - `LTX-2.5/vae/ltx-2.5-video-vae-bf16.safetensors`
  - `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
  - `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`
  - `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors`
- **VLM & Segmentation Weights**:
  - Gemini Flash API (`GEMINI_API_KEY`)
  - Local HuggingFace cache: SAM2.1 (`facebook/sam2.1-hiera-large`)
