# AlphaCut Roadmap

This file tracks active and proposed work. Blocked items live in
`Roadmap_Blocked.md`. Completed delivery history lives in git history
and `CHANGELOG.md`.

## Active Items

No active roadmap items.

## Research-Driven Additions

### P0

### P1

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
