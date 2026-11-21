# Changelog

## 1.0.0 — 2026-11-21

First release, extracted from NPSS-Enhancer's `GUI/still_merge.py` and rebuilt as a
standalone package.

- `fuse()` — align a frame stack and fuse it: median reference, Gaussian weights, weighted mean.
- `align_stack()` — block phase-correlation affine registration (4×4 grid) plus an SSD
  sub-pixel refinement (+0.7 dB over the raw parabolic fit in testing).
- `from_video()` / `read_video_frames()` / `auto_count()` — ffmpeg frame extraction, with the
  frame count chosen from inter-frame motion.
- Temporal-consistency blend now uses an **absolute** threshold (`var < 2σ²`). The original
  used the 60th percentile, which forces a fixed 40% of pixels back to the single frame —
  on a static scene that alone cost 1.6 dB.
- `benchmark.py` reports the oracle-alignment column so registration error is visible.
- `demo.py` — CLI for a video or a folder of frames.
