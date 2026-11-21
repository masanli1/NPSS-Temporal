# NPSS-Burst

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Burst](https://github.com/masanli1/NPSS-Burst)

[中文说明](README.zh-CN.md)

> **Fuse a burst of consecutive frames into one clean photo.**
> Consecutive frames are *complementary*: same scene, independent noise. Align them and the
> noise averages down by √N while the detail stays — something no single-frame denoiser can do.

Pure numpy for the core; ffmpeg only to pull frames out of a video.

## Pipeline

```
N consecutive frames
  ↓  per-frame affine registration (block phase correlation, sub-pixel + SSD refine)
  ↓  robust weighted fusion (median reference → Gaussian weights → weighted mean)
  ↓  single-frame denoise (NPSS) as the fallback
  ↓  temporal-consistency blend (moving regions fall back to the single frame)
one clean photo
```

## Measured

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
