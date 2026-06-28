# AlphaCut v1.6.1

**Video & image background removal + compositing.**

AlphaCut uses ONNX segmentation models to isolate subjects from video and image backgrounds, with built-in compositing, mixed batch processing, quick previews, and detected hardware encoding.

![Version](https://img.shields.io/badge/Version-v1.6.1-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-blue?style=flat-square)

---

## Features

**AI & Processing**
- 10 AI Models — U2Net, ISNet, BiRefNet (6 variants) — from fast drafts to cinema-quality edges
- 11 Software Output Formats — ProRes 4444+Alpha, WebM VP9+Alpha, H.264, HEVC, AV1, Animated WebP, Animated GIF, PNG sequences, green screen, grayscale matte, FG+Alpha pair
- Hardware Encoder Options — detected NVENC/QSV H.264 and HEVC formats appear automatically when FFmpeg exposes them
- Single & Batch Image Mode — drop one image or mixed image/video batches for transparent PNG export
- Chroma-Key Fallback — auto-detect green/blue screens; FFmpeg chroma-key for 10x speed boost
- Pipelined I/O — parallel decode/infer/save threads for higher throughput
- Frame Skip — process every Nth frame with mask reuse (up to 10x speedup)
- Benchmark Mode — test 10 samples to estimate total processing time
- Memory Monitoring — live RAM % during processing with warnings
- Resume Interrupted Jobs — progress saved every 50 frames, auto-resumes on restart
- Pipe Mode — `--pipe` streams raw RGBA to stdout for FFmpeg/scriptable pipelines
- Model Registry — `models.json` declares model name, URL, hash, input size, and license

**Advanced Compositing**
- ROI Selection — crop AI inference to a preview-selected subject region
- Manual Mask Painting — foreground/background brush corrections applied to preview, single export, and batch export
- Mask Preview Modes — compare original/result, inspect binary masks, gray masks, and red/green overlays
- Background Replacement — solid color (6 presets + custom picker) or image file
- Color Spill Suppression — reduce green/blue/red spill along mask edges
- Shadow Preservation — detect and keep ground shadows via luminance analysis
- Mask Inversion — remove subject instead of background

**Batch & Workflow**
- Batch Queue — drop multiple videos, images, folders, or mixed sets and process sequentially
- Job Table — per-file thumbnails, per-job model/format overrides, status, progress %, and output path
- Thumbnail Grid — visual grid preview of all batch files before processing
- Quick Preview — render a 10-second sample clip before committing to full processing
- Drag-Out Export — drag output file directly into NLE timelines
- CLI Mode — full argparse interface for headless/scripted operation (with auto chroma-key detection)
- Auto-Naming — configurable output patterns ({name}, {model}, {format}, {date})
- Export Presets — save/load all settings including compositing
- Recent Files — last 20 files quick-access menu

**UI/UX**
- Before/After Split View with draggable divider
- Smart Model Picker, Toast Notifications, Animated Progress
- System Tray integration, Scrubable Preview
- Menu Bar — File, Tools (Model Manager, Settings Folder), Help (Updates, About)
- About Dialog — version, system info, GitHub links
- Update Checker — checks GitHub releases on startup and on demand
- Model Manager — view/delete cached ONNX models with size info
- Settings Persistence — all preferences saved between sessions with failure logging
- Localization Ready — expanded i18n coverage with `locale_template.json` for community translations

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.9+ |
| **FFmpeg** | On PATH |
| **NVIDIA GPU** | Optional — CUDA 12 + cuDNN 9 |

## Installation

Windows releases include `AlphaCut-Setup-<version>.exe`, an Inno Setup installer that installs AlphaCut per user and creates Start Menu shortcuts. The portable `AlphaCut-windows.exe` is still published for users who prefer a standalone executable.

### Docker

```bash
docker build -t alphacut .
docker run --rm -v $(pwd):/data alphacut -i /data/video.mp4 -f webm -o /data/output.webm
```

GPU-enabled Docker builds use the NVIDIA CUDA runtime and `onnxruntime-gpu`:

```bash
docker build -f Dockerfile.gpu -t alphacut:gpu .
docker run --rm --gpus all -v $(pwd):/data alphacut:gpu -i /data/video.mp4 -f webm -o /data/output.webm --gpu-device 0
```

## Quick Start

```bash
# GUI
python AlphaCut.py

# CLI — single image (outputs transparent PNG)
python AlphaCut.py -i photo.jpg -m BiRefNet

# CLI — video
python AlphaCut.py -i video.mp4 -f prores -m BiRefNet

# CLI — background replacement
python AlphaCut.py -i video.mp4 -f webm --bg-image beach.jpg

# CLI — green spill + shadow preservation
python AlphaCut.py -i greenscreen.mp4 -f prores --spill 60 --shadow 40

# CLI — batch with frame skip
python AlphaCut.py -i vid1.mp4 vid2.mp4 vid3.mp4 -f webm --frame-skip 3

# CLI — batch images (outputs transparent PNG files)
python AlphaCut.py -i photo1.jpg photo2.png photo3.webp -m BiRefNet -y

# CLI — invert mask + custom background
python AlphaCut.py -i video.mp4 -f prores --invert --bg-color 0,0,0

# CLI — FG + Alpha pair (for NLEs without alpha support)
python AlphaCut.py -i video.mp4 -f fg_alpha -m BiRefNet

# CLI — pipe raw RGBA into FFmpeg
python AlphaCut.py -i video.mp4 --pipe 2>nul | ffmpeg -f rawvideo -pix_fmt rgba -s 1920x1080 -r 30 -i - -c:v libvpx-vp9 out.webm
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | — | Input video/image file(s) |
| `-o`, `--output` | Auto-named | Output path |
| `-m`, `--model` | `u2net_human_seg` | Model name (partial match) |
| `-f`, `--format` | `mp4` | mp4, hevc, av1, prores, webm, webp_anim, gif_anim, png_seq, greenscreen, matte, fg_alpha, mp4_nvenc, hevc_nvenc, mp4_qsv, hevc_qsv |
| `--gpu-device` | -1 (auto) | GPU device index for inference |
| `--fp16` | — | Use half-precision inference on GPU |
| `--max-res` | 0 (original) | Max resolution |
| `--edge` | 0 | Edge softness (0-100) |
| `--shift` | 0 | Mask shift (-20 to +20) |
| `--temporal` | 0 | Temporal smooth (0-7) |
| `--frame-skip` | 1 | Process every Nth frame |
| `--invert` | — | Invert mask |
| `--spill` | 0 | Spill suppression (0-100) |
| `--spill-color` | green | Spill color: green, blue, red |
| `--shadow` | 0 | Shadow preservation (0-100) |
| `--bg-color` | — | Background R,G,B |
| `--bg-image` | — | Background image path |
| `--no-audio` | — | Strip audio |
| `-y`, `--overwrite` | — | Overwrite output without confirmation |
| `--chroma-key` | — | Use FFmpeg chroma-key when a green/blue screen is detected |
| `--pipe` | — | Stream raw RGBA to stdout for FFmpeg pipelines |
| `--json` | — | Emit newline-delimited JSON progress/status/error events; failures exit non-zero and end with a `failed` event |

## AI Models

| Model | Speed | Quality | Best For |
|---|---|---|---|
| `u2netp` | Fast | Good | Quick previews |
| `u2net_human_seg` | Fast | Great | **People (default)** |
| `silueta` | Fast | Good | People (small) |
| `u2net` | Medium | Great | General subjects |
| `isnet-general-use` | Medium | Great | General subjects |
| `isnet-anime` | Medium | Great | Anime / illustrations |
| `BiRefNet-portrait` | Slow | Excellent | Portraits, hair detail |
| `BiRefNet-general` | Slow | Excellent | **Best overall** |
| `BiRefNet-general-lite` | Medium | Excellent | Fast + high quality |
| `BiRefNet-matting` | Slow | Excellent | Hair and transparent edges |

## Project Planning

- `ROADMAP.md` - active and proposed work.
- `Roadmap_Blocked.md` - items blocked by external dependencies or research.
- `RESEARCH.md` - research notes that inform future roadmap candidates.

## Architecture

```
AlphaCut.py (single file, ~4,850 lines)
├── Crash Handler + Bootstrap (auto-installs all deps)
├── AlphaCutEngine — ONNX inference + mask refinement
│   ├── Edge refinement, temporal smoothing
│   ├── Spill suppression, shadow preservation
│   ├── Mask inversion, background compositing
│   ├── SHA-256 model integrity verification
│   └── Engine cache (singleton)
├── ProcessingWorker — Pipelined frame processing
│   ├── Reader thread → AI inference → Saver thread
│   ├── Frame skip with mask reuse
│   └── Resume from last saved frame
├── BatchWorker — Sequential multi-file processing
├── ImageProcessWorker — Single and batch image background removal
├── BenchmarkWorker — 10-frame speed estimation
├── PreviewFrameWorker — Single-frame preview
├── QuickPreviewWorker — 10-second sample render workflow
├── UpdateChecker — GitHub releases API
├── AboutDialog — Version/system info
├── ModelManagerDialog — Download/delete models
├── SplitPreviewWidget — Before/after comparison
│   └── ROI selection + manual mask paint overlays + mask viewer modes
├── DragOutButton — Drag output file to NLE
├── ThumbnailGrid — Visual batch file preview grid
├── ToastWidget — Floating notifications
├── CLI — Full argparse headless interface
├── Pipe Mode — Raw RGBA stdout for FFmpeg pipelines
├── i18n — _tr() scaffold + locale_template.json
├── models.json — External model registry
└── Settings/Presets — JSON persistence
```

## Local Build

```bash
pyinstaller --onefile --windowed --runtime-hook packaging/pyinstaller_runtime_hook.py --collect-all onnxruntime --collect-submodules scipy AlphaCut.py
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\AlphaCut.iss
```
