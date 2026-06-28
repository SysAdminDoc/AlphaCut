# Research - AlphaCut

## Executive Summary
AlphaCut v1.6.0 is a local-first Python/PyQt6 desktop and CLI tool for ONNX-based video and image background removal, with FFmpeg encoding, batch jobs, model downloads, quick previews, manual mask edits, Docker CLI images, and Windows installer packaging. Its strongest shape is the privacy-preserving offline workflow: it already covers the mainstream creator/NLE export formats better than most small OSS tools. The highest-value direction is trust and automation hardening before new model work: make CLI failures observable, make resume deterministic across restarts, pin first-download model integrity, avoid pipe-mode deadlocks, and turn the current static tests into real CLI/FFmpeg integration coverage. Next priorities are signed/checksummed release artifacts, accelerator installation profiles, responsive/accessibility polish, and temporal-quality options that can be added without abandoning the ONNX/local design.

Top opportunities:
- P0: Make CLI/JSON mode fail non-zero and emit error events when image/video workers fail (`AlphaCut.py:4500`, `AlphaCut.py:4640`, `AlphaCut.py:4662`).
- P0: Replace salted `hash(self.input_path)` resume IDs with stable content/path hashing so interrupted jobs actually resume after app restart (`AlphaCut.py:1292`).
- P0: Add expected model hashes to `models.json` and verify first downloads instead of trusting the first downloaded sidecar (`models.json`, `AlphaCut.py:887`).
- P0: Prevent pipe-mode FFmpeg stderr pipe deadlock on noisy/long inputs (`AlphaCut.py:4750`).
- P1: Promote the real CLI parser to a reusable function and stop duplicating argparse choices in tests (`AlphaCut.py:4791`, `tests/test_cli.py:8`).
- P1: Add generated-media integration tests for CLI video/image paths, JSON events, overwrite behavior, and encode failure handling.
- P1: Add signed installer/portable build support plus SHA-256 checksums for release artifacts (`packaging/windows/AlphaCut.iss`, `AlphaCut-windows.spec`).
- P1: Add CPU/CUDA/DirectML/CoreML install profiles and clearer runtime detection for Windows/macOS acceleration (`requirements*.txt`, `AlphaCut.py:669`).
- P2: Improve large-clip animated WebP/GIF handling with hard memory estimates and early refusal instead of warnings only (`AlphaCut.py:1685`, `AlphaCut.py:1716`).
- P2: Add accessibility and layout validation for fixed-width/high-DPI UI states (`AlphaCut.py:3364`, `AlphaCut.py:3598`).

## Product Map
- Core workflows: single video background removal; single image transparent PNG export; mixed batch image/video queue; quick 10-second preview; CLI/Docker processing; raw RGBA pipe output.
- User personas: solo video editors, content creators, VFX/compositing users needing alpha exports, batch automation users, privacy-sensitive users avoiding cloud background removal.
- Platforms and distribution: Python 3.9+ source on Windows/Linux/macOS; Windows PyInstaller portable exe; Inno Setup per-user installer; CPU and NVIDIA GPU Docker CLI images.
- Key integrations and data flows: FFmpeg decode/encode; ONNX Runtime inference; local model cache in `~/.alphacut/models`; GitHub release update check; model registry from `models.json`; optional Docker volume-mounted inputs/outputs.

## Competitive Landscape
- rembg: Strongest OSS integration reference; offers CLI, library, HTTP server, Docker, stdin/stdout frame workflows, backend-specific install extras, and folder watch mode. Learn from its single parser/API surface, server/binary stream discipline, and hardware install profiles; avoid importing its heavier dependency tree because AlphaCut's value is direct ONNX plus GUI/NLE workflows.
- transparent-background: Strong image/video quality reference through InSPyReNet-style matting. Learn from model-specific preprocessing and simple CLI ergonomics; avoid PyTorch-only integration unless exported ONNX models and preprocessing metadata are available.
- BiRefNet upstream: Best fit for AlphaCut's current ONNX registry direction, especially high-quality portrait/matting/DIS variants already represented in `models.json`. Learn from dynamic-resolution and task-specific variants; avoid adding huge models without size warnings, expected hashes, and download cancellation tests.
- RobustVideoMatting and MatAnyone: Best evidence that video matting quality depends on temporal memory, not just per-frame segmentation. Learn from temporal-state concepts and jitter metrics; avoid treating RVM/MatAnyone as drop-in registry entries because their recurrent/memory architecture differs from AlphaCut's current per-frame ONNX pipeline.
- backgroundremover: Closest simple OSS video/image CLI competitor. Learn from CLI-first positioning and simple commands; avoid weaker GUI/distribution story because AlphaCut can win on local desktop workflow.
- remove.bg and Unscreen: Commercial benchmark for polished API contracts, predictable outputs, and paid batch/video workflows. Learn from strict API/status semantics, confidence/preview affordances, and production download links; avoid cloud-only processing and per-asset pricing because AlphaCut's differentiator is offline processing.
- HandBrake: Best adjacent queue/preset/encoder UX reference. Learn from per-job queues, encoder availability feedback, presets, logs, and recoverable batch failures; avoid overwhelming AlphaCut's focused background-removal UI.
- DaVinci Resolve/Adobe/Runway/CapCut: Professional and consumer references for object cutout, preview, and NLE handoff. Learn from quality inspection, mask/refine controls, and export affordances; avoid full editor/timeline expansion.

## Security, Privacy, and Reliability
- Verified: `models.json` contains model URLs and licenses but no expected hashes, while `_ensure_model()` creates a trusted `.sha256` sidecar only after first download (`models.json`, `AlphaCut.py:887-931`). This does not protect against a compromised first download.
- Verified: Resume IDs use Python's salted `hash(self.input_path)` (`AlphaCut.py:1292`), so a crash/restart can compute a different progress file/WIP directory and miss saved work.
- Verified: CLI mode runs `ProcessingWorker` and `ChromaKeyWorker` synchronously but does not connect `error`/`finished` to an outcome accumulator (`AlphaCut.py:4640-4678`); `_cli_process_image()` catches exceptions, prints an error, and returns no failure signal (`AlphaCut.py:4500-4538`). Automation can receive a final `"complete"` JSON event and exit 0 after failures.
- Verified: `run_pipe()` starts FFmpeg with `stderr=subprocess.PIPE` and never drains stderr (`AlphaCut.py:4750-4783`), which can hang on long/noisy inputs once the pipe fills.
- Verified: Batch worker relays progress/status/log/preview but not the worker's error signal (`AlphaCut.py:1920-1946`), so table rows can show "No output produced" without the real underlying FFmpeg/model/image error.
- Verified: Windows installer config does not define `SignTool`, and `AlphaCut-windows.spec` has `codesign_identity=None`; release artifacts are not wired for signing/checksum generation.
- Verified: existing Docker images are CLI-only and avoid PyQt6 through `requirements-cli.txt`, which preserves the local/headless direction. GPU Docker guidance should also document NVIDIA Container Toolkit expectations.
- Likely: animated WebP/GIF paths load all frames into memory (`AlphaCut.py:1685-1745`); warnings exist, but there is no hard estimate/refusal when RAM is insufficient.

## Architecture Assessment
- `AlphaCut.py` is 4,857 lines and still workable, but the natural seams are now clear: parser/CLI, engine/model registry, FFmpeg encoding, workers, and GUI widgets. Splitting should wait until parser/error/test hardening lands so behavior is pinned first.
- CLI should expose a single `build_parser()` used by `main()` and `tests/test_cli.py`; current test parser duplication already requires manual sync for every new flag/format.
- Encoding would benefit from a result object rather than inferred success from output-file existence; this would make GUI, batch, and CLI share the same success/error contract.
- Model registry should validate schema, expected hash, file basename, model size, license, preprocessing fields, and URL host before a download starts.
- Tests are better than the previous static-only state, but they still avoid importing the application and do not exercise FFmpeg, worker error signals, pipe mode, or JSON CLI outcomes.
- UI has accessible names for many controls, but the fixed 1200x820 window and 400px left panel need high-DPI, small-screen, keyboard focus, color-contrast, and text-overflow checks against Qt/WCAG guidance.

## Rejected Ideas
- Full plugin system: Rejected because `models.json` already provides the extensibility AlphaCut needs now; a plugin ABI would add maintenance cost without evidence of user demand.
- Cloud processing/API relay: Rejected because it contradicts the local-first privacy positioning that differentiates AlphaCut from remove.bg/Unscreen.
- Full NLE/timeline editor: Rejected because DaVinci Resolve, Premiere, CapCut, and Runway already own that workflow; AlphaCut should remain a focused extraction/compositing tool.
- PyTorch model bundling: Rejected unless an ONNX export and preprocessing metadata are available; PyTorch would undermine the lightweight direct-ONNX architecture.
- RVM/MatAnyone as simple registry entries: Rejected for now because they require recurrent/memory inference semantics, not a single per-frame mask call.
- NCNN/Vulkan backend: Rejected here because it is already parked in blocked planning and would add a second inference stack.
- Mobile app: Rejected because the Python/PyQt6 desktop stack does not target mobile distribution.
- BRIA RMBG-2.0 bundling: Rejected because the available model license is not a clean fit for AlphaCut's MIT/offline redistribution story.

## Sources
Competitors and models:
- https://github.com/danielgatis/rembg
- https://github.com/plemeri/transparent-background
- https://github.com/ZhengPeng7/BiRefNet
- https://github.com/PeterL1n/RobustVideoMatting
- https://github.com/pq-yang/MatAnyone
- https://github.com/nadermx/backgroundremover
- https://github.com/MCG-NJU/MODNet
- https://huggingface.co/briaai/RMBG-2.0
- https://www.remove.bg/api
- https://www.unscreen.com/api
- https://runwayml.com/ai-tools/green-screen
- https://www.capcut.com/tools/remove-background-from-video

ONNX, FFmpeg, and packaging:
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- https://github.com/microsoft/onnxruntime/releases
- https://github.com/microsoft/onnxruntime/security/advisories
- https://ffmpeg.org/ffmpeg-codecs.html
- https://trac.ffmpeg.org/wiki/Encode/VP9
- https://trac.ffmpeg.org/wiki/Encode/H.265
- https://pyinstaller.org/en/stable/runtime-information.html
- https://jrsoftware.org/ishelp/index.php?topic=setup_signtool
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Accessibility and UX:
- https://doc.qt.io/qt-6/accessible.html
- https://www.w3.org/TR/WCAG22/
- https://handbrake.fr/docs/
- https://documents.blackmagicdesign.com/UserManuals/DaVinci-Resolve-20-Reference-Manual.pdf
- https://helpx.adobe.com/premiere-pro/using/masking-tracking.html

## Open Questions
- Which release-signing certificate, if any, is available on the build machine for Windows exe/installer signing?
- Should expected model hashes be sourced only from upstream release assets, or should AlphaCut publish its own reviewed hash manifest per release?
