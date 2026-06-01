# AlphaCut Completed Work

This file preserves shipped roadmap history. Active work lives in `ROADMAP.md`.

## v1.2.0

- [x] **Animated GIF export** - `gif_anim` format with a 256-color palette,
  binary transparency, and PIL `disposal=2` restoration.
- [x] **Built-in chroma-key fallback** - `detect_chroma_background()` detects
  green/blue screens from corner patches, `ChromaDetectWorker` runs detection
  in the background, and `ChromaKeyWorker` uses FFmpeg chroma-key filters for
  faster synthetic-background processing.
- [x] **Batch thumbnail previews** - `ThumbnailLoader` extracts 80px thumbnails
  and `JobTable` displays thumbnail, file, status, progress, and output path.
- [x] **CLI chroma-key support** - `--chroma-key` flag and headless
  auto-detection.

## v1.1.0

- [x] **PyInstaller single-exe packaging** - `.github/workflows/build.yml`
  builds Windows, Linux, and macOS artifacts and can publish GitHub releases.
- [x] **Animated WebP transparent export** - `webp_anim` format with RGBA
  transparency and quality control.
- [x] **SHA-256 model integrity verification** - `.sha256` sidecars are written
  after download and verified before model load.
- [x] **requirements.txt** - pinned dependency set for clean installs.

## v1.0.0

### Phase 1: Core Engine and Quality

- Temporal smoothing, edge refinement, engine caching
- Audio passthrough, matte output, scrubable preview, settings persistence

### Phase 2: UI/UX Overhaul

- Split preview, smart picker, toasts, animated progress, system tray

### Phase 3: Batch and Workflow

- Batch queue, job table, CLI, auto-naming, presets, recent files

### Phase 4: Performance

- Pipelined I/O, frame skip, benchmark, memory monitoring

### Phase 5: Advanced Compositing

- Background replacement, spill suppression, shadow preservation, mask inversion

### Phase 6: Polish and Distribution

- Menu bar, About dialog, update checker, model manager, and interrupted-job
  resume support
