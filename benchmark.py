#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Burst fusion benchmark, including the ceiling that synthetic data imposes.

    python3 benchmark.py
    python3 benchmark.py --frames 8 16 --sigma 15

Two important columns:

  * `ceiling` — average the *noise-free* shifted frames using the known shifts. Bilinear
    resampling blurs every synthetic frame, so this is the best any fusion can reach on
    this data. It sits around 31–32 dB regardless of frame count.
  * fusion rows — the real algorithm on the noisy frames.

If you want to reproduce high-PSNR fusion results (40 dB+), you need real footage: captured
frames are not resampled, so they carry no such blur.
"""
import argparse
import zlib

import numpy as np
from PIL import Image

import npss_burst as B
import npss_image as M


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def make_scene(size=768):
    rng = np.random.default_rng(11)
    f = np.fft.fftfreq(size)
    F = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    F[0, 0] = 1e-6
    spec = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    base = np.real(np.fft.ifft2(spec / F ** 0.9))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    yy, xx = np.mgrid[0:size, 0:size]
    base[(yy > size * 0.3) & (yy < size * 0.7) & (xx > size * 0.2) & (xx < size * 0.8)] = 0.85
    return np.stack([base, base * 0.9 + 0.05, base * 0.8 + 0.1], -1).astype(np.float32)


def crop_at(img, dy, dx, size=384):
    """整数裁剪：不经过任何重采样，因此不会引入模糊（这是关键）。"""
    y0 = int(round(img.shape[0] / 2 - size / 2 + dy))
    x0 = int(round(img.shape[1] / 2 - size / 2 + dx))
    y0 = max(0, min(y0, img.shape[0] - size))
    x0 = max(0, min(x0, img.shape[1] - size))
    return img[y0:y0 + size, x0:x0 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--sigma", type=float, nargs="+", default=[15])
    ap.add_argument("--max-shift", type=float, default=1.5)
    ap.add_argument("--scale", type=int, nargs="+", default=[1, 4], help="flow pyramid scales")
    args = ap.parse_args()

    clean = make_scene()
    print(f"scene {clean.shape[1]}x{clean.shape[0]}, per-frame shift <= ±{args.max_shift} px")
    for s0 in args.sigma:
        sigma = s0 / 255.0
        for n in args.frames:
            rng = np.random.default_rng(zlib.crc32(f"{n}-{s0}".encode()) % 100000)
            amp = int(max(1, round(args.max_shift)))
            shifts = [rng.integers(-amp, amp + 1, 2) for _ in range(n)]
            clean_frames = np.stack([crop_at(clean, dy, dx) for dy, dx in shifts])
            clean = clean_frames[0] * 0 + crop_at(clean, 0, 0) if False else clean
            target = clean_frames[n // 2]                        # 参考帧的内容本身就是真值
            ceiling = psnr(clean_frames.mean(0), target)          # 完美对齐 + 零噪声
            frames = np.stack([np.clip(clean_frames[i] + rng.normal(0, sigma, target.shape), 0, 1)
                               for i in range(n)])
            ref = frames[n // 2]
            single, _ = M.denoise(ref, sigma=sigma)

            print(f"\nσ={s0:.0f}  {n:2d} frames   (synthetic ceiling {ceiling:.2f} dB)")
            print(f"  single frame (noisy)     {psnr(ref, target):6.2f}")
            print(f"  single frame denoised    {psnr(single, target):6.2f}")
            for sc in args.scale:
                raw, _ = B.fuse(frames, sigma=sigma, polish=False, flow_scale=sc)
                pol, info = B.fuse(frames, sigma=sigma, flow_scale=sc)
                print(f"  flow scale={sc}: accumulate {psnr(raw, target):6.2f}   + polish {psnr(pol, target):6.2f}")


if __name__ == "__main__":
    main()
