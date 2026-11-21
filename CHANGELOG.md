# Changelog

## 1.1.0 — 2026-11-21

Replaced the global-affine core with the **trajectory accumulation** implementation from
NPSS-Enhancer (`GUI/traj_denoise.py`) — block-matching flow chained along the trajectory.

Three real bugs found and fixed while porting:

1. **Flow only at 1/4 resolution** — a 1–2 px jitter produced Δ = 0.00 px, i.e. no correction
   at all. Added a full-resolution stage and a coarse-to-fine pyramid (reaches ±24 px *and*
   sub-pixel). Registration on a 1 px shift: 33.47 → **39.58 dB**.
2. **Flat blocks win the first search offset** — blocks without texture have zero error at
   every offset (aperture problem), so `-R` wins and smears across the field. A zero
   displacement pair reported **−4.65 px**. Fixed with a variance gate + field inpainting:
   34.16 → **108.00 dB** on a zero-shift pair.
3. **Block error not normalised by overlap** — border blocks had no valid score.

Benchmarks are honest about their limits: synthetic resampled bursts cap at ~31 dB regardless
of frame count, so the historical 40.66 dB (static regions) cannot be reproduced without real
footage. The harness reports the ceiling explicitly.

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
