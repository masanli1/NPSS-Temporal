#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse a burst of frames into one clean photo.

    python3 demo.py --video clip.mp4 --time 3.0
    python3 demo.py --frames dir/          # PNG/JPG frames, sorted by name
    python3 demo.py --video clip.mp4 --frames-n 12 --out clean.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import npss_burst as B


def load_dir(d):
    files = sorted([p for p in Path(d).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")])
    if len(files) < 2:
        sys.exit(f"{d} 里至少要有 2 张图")
    arrs = [np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0 for f in files]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        sys.exit("所有帧尺寸必须一致")
    return np.stack(arrs), [f.name for f in files]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file")
    src.add_argument("--frames", help="directory of frames")
    ap.add_argument("--time", type=float, default=1.0, help="centre time in the video (s)")
    ap.add_argument("--frames-n", type=int, default=None, help="how many frames (default: auto)")
    ap.add_argument("--sigma", type=float, default=None, help="noise sigma on 0-255 scale")
    ap.add_argument("--out", default="burst_clean.png")
    args = ap.parse_args()

    if args.video:
        out, info = B.from_video(args.video, args.time, args.frames_n,
                                 sigma=None if args.sigma is None else args.sigma / 255.0)
        print(f"fused {info['frames']} frames @ {info.get('fps', 0):.1f} fps, "
              f"sigma={info['sigma'] * 255:.1f}")
    else:
        frames, names = load_dir(args.frames)
        info = {}
        out, info = B.fuse(frames, sigma=None if args.sigma is None else args.sigma / 255.0)
        print(f"fused {info['frames']} frames from {len(names)} files, sigma={info['sigma'] * 255:.1f}")

    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
