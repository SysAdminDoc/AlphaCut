# AlphaCut Roadmap

This file tracks active and proposed work. Blocked items live in
`Roadmap_Blocked.md`. Completed delivery history lives in git history
and `CHANGELOG.md`.

## Active Items

No active roadmap items.

## Research-Driven Additions

### P0

### P1

- [ ] P1 - Add generated-media integration tests for CLI and FFmpeg paths
  Why: Current tests validate helpers and static source patterns but do not exercise real output creation, JSON events, overwrite behavior, or encode failures.
  Evidence: `tests/`; `AlphaCut.py:1605`; `AlphaCut.py:4544`
  Touches: `tests/`, `AlphaCut.py`
  Acceptance: Tests generate tiny image/video fixtures, run image CLI and at least one video encode path with a lightweight mocked/session model, and verify output, errors, and JSON event order.
  Complexity: L

- [ ] P1 - Preserve real worker errors in batch rows
  Why: Batch jobs can collapse model/FFmpeg/image exceptions into "No output produced", slowing recovery.
  Evidence: `AlphaCut.py:1920`; `AlphaCut.py:4417`; HandBrake queue/log UX.
  Touches: `AlphaCut.py`, `locale_template.json`
  Acceptance: Batch rows display the worker's last error summary, full detail remains in the log, and a failed job does not hide the original exception.
  Complexity: S

- [ ] P1 - Add signed release and checksum packaging path
  Why: Windows installer and portable exe builds are not wired for signing or checksum manifests.
  Evidence: `packaging/windows/AlphaCut.iss`; `AlphaCut-windows.spec`; Inno Setup SignTool docs.
  Touches: `packaging/windows/AlphaCut.iss`, `AlphaCut-windows.spec`, `README.md`
  Acceptance: Local release build can sign when a certificate is configured, emits SHA-256 checksum files for installer/exe, and documents unsigned fallback behavior when no certificate exists.
  Complexity: M

- [ ] P1 - Add accelerator install profiles and runtime diagnostics
  Why: Code probes CUDA, DirectML, and CoreML providers, but requirements/docs only make CPU and CUDA Docker paths obvious.
  Evidence: `AlphaCut.py:669`; `requirements.txt`; ONNX Runtime DirectML/CoreML docs; rembg CPU/GPU extra pattern.
  Touches: `requirements.txt`, `requirements-cli.txt`, `README.md`, `AlphaCut.py`
  Acceptance: Users can choose CPU, CUDA, DirectML, or macOS CoreML install profiles, and startup/about diagnostics explain unavailable providers and required packages.
  Complexity: M

### P2

- [ ] P2 - Enforce memory limits for animated WebP and GIF exports
  Why: Animated WebP/GIF builds load all frames into RAM and currently only warn for long clips.
  Evidence: `AlphaCut.py:1685`; `AlphaCut.py:1716`; README WebM recommendation.
  Touches: `AlphaCut.py`, `locale_template.json`, `tests/test_real_functions.py`
  Acceptance: Export estimates memory before loading frames, refuses or requires explicit override when projected RAM is unsafe, and recommends WebM/PNG sequence alternatives.
  Complexity: M

- [ ] P2 - Add UI accessibility and responsive-layout validation
  Why: The GUI has accessible names, but fixed window/panel dimensions and dense dark controls need high-DPI, contrast, focus, and text-overflow verification.
  Evidence: `AlphaCut.py:3364`; `AlphaCut.py:3598`; Qt accessibility docs; WCAG 2.2.
  Touches: `AlphaCut.py`, `locale_template.json`, optional UI smoke test tooling
  Acceptance: Main workflows fit at 125 percent DPI and smaller desktop widths, focus order is predictable, non-color cues exist for status/errors, and contrast/text clipping issues are fixed.
  Complexity: M

- [ ] P2 - Add mask quality inspection metrics
  Why: Commercial and pro tools emphasize confidence/edge inspection; AlphaCut has mask views but no numeric or localized problem indicators.
  Evidence: `AlphaCut.py:3536`; remove.bg/Unscreen workflow expectations; DaVinci/Adobe mask review patterns.
  Touches: `AlphaCut.py`, `locale_template.json`
  Acceptance: Preview computes lightweight edge/jitter/transparent-pixel summaries, surfaces warnings in the preview/log, and helps users choose model/refinement settings before full export.
  Complexity: L

### P3

- [ ] P3 - Add optional local watch-folder/server mode
  Why: rembg's folder watch and HTTP server are useful integration patterns for automation, while AlphaCut already has CLI/Docker building blocks.
  Evidence: rembg README; `Dockerfile`; `AlphaCut.py:4544`
  Touches: `AlphaCut.py`, `Dockerfile`, `README.md`, tests
  Acceptance: A local-only watch-folder or localhost server mode processes new files with saved presets, reports structured status, and remains disabled by default.
  Complexity: XL
