# NPSS-Temporal

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Burst](https://github.com/masanli1/NPSS-Burst)

[中文说明](README.zh-CN.md)

> **Denoise a video continuously by complementing each frame with its temporal neighbours.**
> Consecutive frames are *complementary*: same scene, independent noise. Align them and the
> noise averages down by √N while the detail stays — something no single-frame denoiser can do.

Pure numpy for the core; ffmpeg only to pull frames out of a video.

## Pipeline

Extracted from NPSS-Enhancer's `GUI/traj_denoise.py` (trajectory accumulation):

```
video frames (continuous)
  ↓  block-matching flow between consecutive frames (coarse-to-fine pyramid)
  ↓  for EVERY frame: chain the flow over a ±radius window, warp the neighbours onto it
  ↓  robust weighted merge (median reference → Gaussian weights)
  ↓  optional single-frame denoise of each result
denoised sequence (same length, same fps)
```

`denoise_sequence()` is the main entry; `fuse()` is the same machinery specialised to one
target frame, for when you want a single still out of a burst.

### Three bugs fixed relative to the original

| bug | effect | fix |
|---|---|---|
| flow computed only at 1/4 resolution | a 1–2 px jitter is **invisible** (measured Δ = 0.00 px, no correction at all) | full-resolution pass added; coarse-to-fine pyramid covers ±24 px *and* sub-pixel |
| flat blocks win at the first search offset | blocks with no texture have zero error at every offset (aperture problem), so the border offset `-R` wins and smears over the whole field — a **zero-displacement** pair reported **−4.65 px** | variance gate rejects flat blocks, then the field is inpainted from valid neighbours |
| block error not normalised by overlap | border blocks (no overlap) always score worst/best arbitrarily | error divided by the number of overlapping pixels |

Measured registration quality on noise-free integer crops:

| true shift | before | after |
|---|---|---|
| 0 px | 34.16 dB (flow −4.65 px) | **108.00 dB** (flow 0.00 px) |
| 1 px | 33.47 dB | **39.58 dB** |
| 3,−2 px | 33.79 dB | **36.81 dB** |

## Honest status of the numbers

`benchmark.py` synthesises bursts by resampling and by cropping, and **neither can
reproduce a 40 dB fusion result**:

- resampled frames are blurred by the interpolation itself — the ceiling is ~31 dB no matter
  how many frames are fused;
- 1/f random textures do not share content between crops, so averaging different crops is not
  the same signal.

The **40.66 dB static-region figure** recorded in NPSS-Enhancer's history came from **real
footage**, where frames are captured rather than resampled. Reproducing it needs a real clip;
the harness here cannot substitute for one.

## Measured on real footage

iPhone still-life clip, 1088×1440, 28.5 fps, 24 frames, handheld drift up to ±15 px,
noise estimated at σ ≈ 1.7/255:

| | single frame | `denoise_sequence(radius=3)` |
|---|---|---|
| noise σ | 1.72 | **1.42** (−17%) |
| per-frame drop | — | 12% … 23% |

Visually the 500ML label text and the gold grain on the bottle become resolvable where the
single frame has grain mixing into them. Cost: **4.8 s/frame** at 1.57 MP on CPU (flow
dominates).

Why only −17% instead of the −67% that √9 suggests: the camera drifts ~15 px over the clip and
the registration residual eats most of the averaging.

## Measured on synthetic bursts

`python3 benchmark.py` — synthetic burst: a textured 384×384 scene, per-frame random shifts
up to ±2 px, AWGN σ=20.

| frames | noisy | single-frame denoise | naive average | **aligned fusion** | oracle alignment |
|---|---|---|---|---|---|
| 4 | 21.34 | 24.60 | 23.89 | **25.57** (+0.97) | 26.90 |
| 8 | 19.02 | 22.71 | 23.25 | **23.00** (+0.29) | 27.17 |

`oracle alignment` uses the known synthetic shifts, so the gap to the fused column is exactly
what registration error costs: **1.3 dB at 4 frames, 4.2 dB at 8 frames**.

### Honest reading

- On this synthetic test the fusion only beats the single-frame denoiser by **0.3–1.0 dB**,
  and at 8 frames it stops improving — residual misalignment blurs the average.
- With perfect alignment the same frames would give **+2.5 to +4.5 dB** over the single-frame
  denoiser, so **registration accuracy, not the fusion idea, is the bottleneck**.
- Real handheld bursts have smaller, smoother motion than this test; results there should sit
  between the two columns, but that has not been measured here.

## Usage

```python
import numpy as np
from PIL import Image
import npss_burst as B

frames = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
                   for f in sorted(paths)])       # (N, H, W, 3)
out, info = B.fuse(frames)                        # sigma=None → estimated
Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save("clean.png")
print(info["frames"], info["merged_fraction"])
```

CLI:

```bash
python3 demo.py --video clip.mp4 --time 3.0      # 从视频里取一段连续帧融合
python3 demo.py --frames burst/                  # 或者直接给一个文件夹
python3 benchmark.py --frames 4 8 16 --sigma 20 50
```

## What it is not

- **Not multi-frame super-resolution.** It fuses to the resolution of the input frames.
- **Not a video denoiser.** It produces one still from a short burst; there is no temporal
  filtering across the whole clip.
- **Not registration-free.** If the scene moves more than a few pixels between frames, the
  affine model breaks down and the fusion degrades toward the single-frame result.

## License

Dual-licensed: free for academic / non-commercial use ([LICENSE-ACADEMIC.md](LICENSE-ACADEMIC.md)),
paid license for commercial use ([LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)).
Contact `2459551532@qq.com`.
