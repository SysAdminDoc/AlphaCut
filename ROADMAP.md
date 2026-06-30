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

### P3

- [ ] P3 - Add optional local watch-folder/server mode
  Why: rembg's folder watch and HTTP server are useful integration patterns for automation, while AlphaCut already has CLI/Docker building blocks.
  Evidence: rembg README; `Dockerfile`; `AlphaCut.py:4544`
  Touches: `AlphaCut.py`, `Dockerfile`, `README.md`, tests
  Acceptance: A local-only watch-folder or localhost server mode processes new files with saved presets, reports structured status, and remains disabled by default.
  Complexity: XL
