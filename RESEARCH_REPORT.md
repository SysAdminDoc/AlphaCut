# AlphaCut Research Report

This report captures research findings used to shape future roadmap candidates.

## Related Open-Source Projects

- **danielgatis/rembg** - U2Net, ISNet, and BiRefNet runner with CLI, HTTP,
  Python, Docker, and FFmpeg-pipe RGB24 stdin workflows.
- **nadermx/backgroundremover** - CLI-first image and video removal reference
  for checkpoint auto-download UX.
- **Ckrest/rvm-virtual-camera** - real-time Robust Video Matting into a virtual
  camera output.
- **PeterL1n/RobustVideoMatting** - upstream RVM model with temporal coherence
  for fast-motion footage.
- **Akascape/Rembg-Fuse** - DaVinci Resolve Fusion plugin reference for
  node-graph compositing UX.
- **Shlok-crypto/Background-Remover-RealTime** - webcam background replacement
  and blur pipeline.
- **sunwood-ai-labs/video-background-remover-cli** - rembg/OpenCV pipeline with
  transparent WebP/GIF output and MatAnyone foreground plus alpha exports.

## Findings

- **Virtual camera output** is a strong distribution feature because it brings
  AlphaCut into OBS, Zoom, Meet, and other live-video tools.
- **FFmpeg-pipe RGB24 mode** would let long processing chains avoid temporary
  frame directories and compose cleanly with existing FFmpeg filters.
- **Foreground plus matte export** broadens NLE compatibility where alpha-aware
  codecs are unreliable.
- **Temporal-coherent models** such as RVM can reduce edge flicker on live or
  interview footage compared with frame-by-frame masks.
- **Interactive mask refinement** would address high-value one-frame defects
  that automatic segmentation cannot reliably resolve.
- **Optical-flow propagation** is a better frame-skip companion than straight
  mask reuse for fast edges.
- **Dual ONNX/NCNN Vulkan backend support** could reduce CUDA dependence and
  improve portability for AMD and Apple Silicon users.
- **Model registry metadata** would make downloadable model management more
  auditable by tracking URL, hash, size, and license in one structured file.
