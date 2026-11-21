#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Burst fusion benchmark: how much does fusing N frames gain over one frame?

    python3 benchmark.py                      # synthetic burst from a built-in image
    python3 benchmark.py --frames 8 16 --sigma 20

Reports, per setting:
    noisy single frame / single-frame denoise / naive average / aligned fusion / oracle
`oracle` aligns with the *known* synthetic shifts, so the gap between it and the
aligned result is exactly what registration error costs.
"""
import argparse
import zlib

import numpy as np
from PIL import Image

import npss_burst as B
import npss_image as M


def psnr(a, b):
    return 10 * np.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-12))


def make_scene(size=384):
    """A deterministic textured scene with sharp and smooth regions."""
    rng = np.random.default_rng(11)
    f = np.fft.fftfreq(size)
    F = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    F[0, 0] = 1e-6
    spec = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    base = np.real(np.fft.ifft2(spec / F ** 0.9))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    yy, xx = np.mgrid[0:size, 0:size]
    base[(yy > size * 0.3) & (yy < size * 0.7) & (xx > size * 0.2) & (xx < size * 0.8)] = 0.85
    return base.astype(np.float32)


def shift_img(img, dy, dx):
    h, w = img.shape
    out = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).transform(
        (w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR), np.float32) / 255.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--sigma", type=float, nargs="+", default=[20])
    ap.add_argument("--max-shift", type=float, default=2.0)
    args = ap.parse_args()

    clean = make_scene()
    print(f"scene {clean.shape[1]}x{clean.shape[0]}")
    for s0 in args.sigma:
        sigma = s0 / 255.0
        for n in args.frames:
            rng = np.random.default_rng(zlib.crc32(f"{n}-{s0}".encode()) % 100000)
            frames, shifts = [], []
            for _ in range(n):
                dy, dx = rng.uniform(-args.max_shift, args.max_shift, 2)
                frames.append(np.clip(shift_img(clean, dy, dx) + rng.normal(0, sigma, clean.shape), 0, 1))
                shifts.append((dy, dx))
            frames = np.stack(frames)
            ref = frames[n // 2]

            single, _ = M.denoise(ref, sigma=sigma)
            naive = frames.mean(0)
            fused, info = B.fuse(frames, sigma=sigma)
            # oracle: 用已知位移反着配准回去
            oracle = np.stack([shift_img(frames[i], -shifts[i][0], -shifts[i][1]) for i in range(n)]).mean(0)

            print(f"\nσ={s0:.0f}  {n:2d} frames")
            print(f"  single frame (noisy)   {psnr(ref, clean):6.2f}")
            print(f"  single frame denoised  {psnr(single, clean):6.2f}")
            print(f"  naive average          {psnr(naive, clean):6.2f}")
            print(f"  aligned fusion         {psnr(fused, clean):6.2f}   (+{psnr(fused, clean) - psnr(ref, clean):.2f} vs noisy,"
                  f" +{psnr(fused, clean) - psnr(single, clean):.2f} vs denoised)")
            print(f"  oracle alignment       {psnr(oracle, clean):6.2f}   (registration costs {psnr(oracle, clean) - psnr(fused, clean):.2f} dB)")


if __name__ == "__main__":
    main()
