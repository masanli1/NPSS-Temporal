# Changelog

## 1.8.0 — 2026-11-22

`denoise_temporal_parallel()` — frames are independent, so the work splits across processes
(fork, shared memory via copy-on-write). Measured on 12 frames (1088×1440):

| preset | workers | s/frame | total |
|---|---|---|---|
| fast | 1 | 1.23 | 15 s |
| **fast** | **4** | **0.82** | **10 s** |
| balanced | 4 | 1.67 | 20 s |

Speedup is 1.5× at 4 workers on a machine already carrying load 4.6; 8 workers is slower
(contention). The presets remain the strongest lever: `fast` at 1.3 s/frame versus the
original multi-scale at 31.6 s/frame is a 24× gain.

### Two findings from this round

- **Batching the phases into one big matmul is slower** (0.82×) — the concatenate copy costs
  more than the BLAS efficiency gains at this size.
- **Padding once per scale instead of once per phase** made no measurable difference; profiling
  puts the cost in the block-transform data movement, not in the padding.

### Why sigma matters more than it looks

Alignment is bilinear, and that resampling lowers the noise: a frame synthesised at σ=12/255
measures **8.3/255** after alignment. Passing the *pre*-alignment σ (12) over-thresholds and
costs 3 dB (32.91 vs 35.89). Auto-estimation on the aligned frames is therefore the right
default for this pipeline.

## 1.7.0 — 2026-11-22

Temporal 3D is now **3.4× faster** (and up to 24× with the fast preset) at identical output.

- `temporal_3d_fast()` — batched BLAS matmuls instead of einsum paths, all channels in one
  batch, float32, and the temporal transform done as a single large matmul.
- `denoise_temporal(seq, preset=...)` with `fast` / `balanced` / `best`.

### Measured (real clip, 1088×1440, σ=12 added, 12 frames, ground truth)

| config | PSNR | SSIM | s/frame |
|---|---|---|---|
| noisy | 29.31 | 0.8003 | — |
| single-frame NPSS | 34.54 | 0.9442 | 3.7 |
| **fast** (phases 4, B=8) | **35.89** | 0.9588 | **1.3** |
| **balanced** (phases 9, B=8) | **36.22** | 0.9623 | **2.8** |
| **best** (phases 9, 4+8+16) | **36.56** | **0.9651** | 9.4 |
| previous slow multi-scale | 36.66 | 0.9660 | 31.6 |

The fix that made the fast path match the reference exactly: the temporal axis had been left
at position 3 while `reshape(-1, T)` assumed it was last, so the "time transform" was silently
mixing block columns — 0.18 max pixel error and 1.9 dB lost. Verified: fast now reproduces the
reference bit-for-bit up to float32 rounding (36.42 vs 36.42 at 5.0 s/frame).

## 1.6.0 — 2026-11-22

The temporal axis is now **inside** the NPSS spatial framework, and it finally beats the
single-frame denoiser.

`temporal_3d(aligned, K=3, blocks=(4,8,16), phases=16, k0=3.0, k0_ac=6.0)`:

- per frame, per block size, per cycle-spinning phase: build a (T, B, B) tensor
- 2D DCT in space, **1D DCT in time**, hard threshold, invert, take the temporal centre frame
- average over phases and scales
- **separate threshold for the temporal ACs** (`k0_ac`, ≈6σ): the temporal DC carries the
  static part and is thresholded like a spatial coefficient, while the temporal ACs carry
  motion — only strong ones survive, so moving detail is kept instead of averaged away

### Measured (real iPhone clip, 1088×1440, σ=12 added, 12 frames, ground truth)

| method | PSNR | SSIM |
|---|---|---|
| noisy | 29.31 | 0.8003 |
| single-frame NPSS (same aligned data) | 34.54 | 0.9442 |
| **temporal_3d K=3, B=8** | **36.42** | **0.9641** |
| **temporal_3d K=3, 4+8+16** | **36.66** | **0.9660** |

+2.12 dB and +0.022 SSIM over the single-frame denoiser. Cost: 8.1 s/frame single scale,
31.6 s/frame multi-scale at 1.57 MP.

### Bugs fixed on the way (all in the transform plumbing)

- block layout: the padded array's first axis is time, not rows
- inverse transforms: two einsums used `Ks` where `Ksᵀ` was needed (and one the reverse),
  which silently produced 12–19 dB output
- the temporal inverse had the same label swap as the spatial one

## 1.5.0 — 2026-11-22

`idm_temporal()` — the three-layer mechanism from the source document
(`3 时域处理的核心机制.md`), applied to the temporal axis and made block-adaptive:

- 蒸馏: a block counts as a critical point over time only if most frames agree about it
  (deviation from the temporal median < τσ).
- 升华: for those blocks take the temporal median (the robust complement).
- 背景: everything else keeps the frame's own content, so moving detail is never averaged.
- No transform to invert; the decision is per 8×8 block, which is what makes it completion
  rather than fusion.

Measured (σ=12 added, 12 frames, ground truth):

| method | PSNR | SSIM |
|---|---|---|
| noisy | 29.3 | 0.80 |
| idm_temporal K=3 / 5 / 9 | 31.2 / 31.9 / 32.0 | 0.87 / 0.89 / 0.89 |
| temporal median over the *whole* clip | 33.8 | 0.94 |
| whole-clip median + spatial denoise | 35.0 | 0.96 |
| **single-frame NPSS** | **36.0** | **0.96** |

Still short of the single-frame denoiser. The reason is structural, not a bug: a K-frame window
borrows 3–9 observations, while the whole-clip estimators borrow 12, and the spatial prior in
NPSS is worth about +9 dB on its own — more than any amount of temporal averaging on this clip.

## 1.4.0 — 2026-11-22

Added `flow_split()` — optical-flow alignment, background subtraction, subject compositing
(the "背景重建 + 人物合成" route), and measured it honestly.

| route (σ=12 added, 12 frames, ground truth) | PSNR | SSIM |
|---|---|---|
| noisy | 26.7 | 0.69 |
| **single-frame NPSS** | **36.0** | **0.96** |
| temporal median background (aligned space) | 33.8 | 0.94 |
| temporal median bg + spatial denoise (aligned space) | 35.0 | 0.96 |
| flow_split, thr 1.5σ / 2σ / 3σ | 29.8 / 27.7 / 26.9 | 0.91 / 0.88 / 0.86 |

`flow_split` loses to the plain background estimate for two reasons:

1. the background is reconstructed in the reference frame's coordinates and then **warped back
   per frame**, so it is resampled twice — that alone costs several dB;
2. the mask threshold trades subject leakage against background loss, and neither setting wins.

**The original implementation of this route does not exist in the repository.** A full-history
search (`git rev-list --all | xargs git grep`) finds "背景重建" only in
`docs/history/2026-10-trajectory-video/README.md`, and no .py in any commit contains
sparse / RPCA / low-rank / background code. The 40.66 dB figure was recorded in prose as a
rejected experiment; its code was never committed.

## 1.3.0 — 2026-11-22

Added the other three routes recorded for video, plus noise injection for evaluation.

- `add_noise(seq, sigma)` — inject AWGN so a clean clip can serve as ground truth.
- `sparse_3d(seq, K=3)` — 3D transform-domain sparse reconstruction: 2D DCT (space) ×
  1D DCT (time), threshold, invert. Static content piles into the temporal DC; motion
  produces large temporal coefficients that survive, so it does not average motion away.
- `rpca(seq)` / `denoise_rpca()` — low-rank background + sparse subject decomposition,
  the "背景重建 + 人物合成" route recorded in NPSS-Enhancer's history.

### Measured against real ground truth (iPhone still-life, σ=12 added, 12 frames)

| route | PSNR | SSIM |
|---|---|---|
| noisy | 26.7 | 0.69 |
| **single-frame NPSS** | **36.0** | **0.96** |
| 3D sparse, K=3 | 28.9 | 0.78 |
| 3D sparse, K=7 | 29.9 | 0.82 |
| flow accumulation, r=1 | 31.3 | 0.88 |
| RPCA low-rank part (unaligned / aligned) | 20.5 / 26.9 | — |
| temporal median background | 33.8 | 0.94 |
| **temporal median background + spatial denoise** | **35.0** | **0.96** |

**Honest conclusion: no temporal route here beats the single-frame denoiser on this material.**
The temporal/low-rank routes buy +4–5 dB (matching √N theory for 12 frames) while the
single-frame DCT route buys +9.3 dB, so the spatial prior is worth more than the temporal one
at this clip length. `rpca` also needs a static camera: with 15 px of drift the background is
not low-rank (20.5 dB), and even after alignment its ALM implementation is numerically fragile.

## 1.2.0 — 2026-11-22

The method is a **continuous time-series complement**, not a burst-to-one-still fusion.
Renamed the project to NPSS-Temporal and made the sequence path the main entry.

- `denoise_sequence(frames, radius=3)` — every output frame is accumulated from a ±radius
  window of its temporal neighbours along their motion trajectories. Output length and fps
  match the input.
- `write_video()` — encode the result to H.264 with ffmpeg.
- `fuse()` remains as the single-target special case.

### Measured on a real iPhone clip (1088×1440, 28.5 fps, 24 frames, drift ≈15 px)

| | single | sequence (radius 3) |
|---|---|---|
| noise σ | 1.72 | **1.42** (−17%) |
| per-frame | — | 12–23% |

Cost 4.8 s/frame at 1.57 MP on CPU. The gap to the theoretical √9 (−67%) is registration
residual caused by the handheld drift.

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
