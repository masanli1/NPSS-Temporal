"""NPSS-Burst — fuse a burst of consecutive frames into one clean photo.

Consecutive frames of a static-enough scene are *complementary*: the content is the same,
the noise is independent. Aligning the frames and fusing them averages the noise down
while keeping the detail, which a single-frame denoiser cannot do.

Pipeline
    1. per-frame affine registration (block phase correlation, sub-pixel)
    2. robust weighted fusion (median reference -> Gaussian weights -> weighted mean)
    3. single-frame denoise as the fallback
    4. temporal-consistency blend: low temporal variance uses the fusion, high variance
       (moving regions) falls back to the single frame

Only numpy for the core. ffmpeg is needed only to read frames out of a video.
"""
from __future__ import annotations

import subprocess

import numpy as np

try:                                    # single-frame fallback is optional
    import npss_image as _M
except Exception:                       # pragma: no cover
    _M = None


# --------------------------------------------------------------------- helpers
def luma(a):
    if a.ndim == 2:
        return a.astype(np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def gaussian_blur(a, sigma=1.0):
    if sigma <= 0:
        return a.astype(np.float32)
    r = int(3 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        if axis == 0:
            out = sum(k[i] * p[i:i + out.shape[0]] for i in range(2 * r + 1))
        else:
            out = sum(k[i] * p[:, i:i + out.shape[1]] for i in range(2 * r + 1))
    return out


def estimate_sigma(a):
    y = luma(a)
    return float(np.median(np.abs(y - gaussian_blur(y, 1.2))) / 0.6745)


def _pc_sub(a, b):
    """Phase correlation; sub-pixel shift that aligns b onto a (parabolic fit)."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfft2(R, s=a.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = iy + _para(c, iy, ix, 0), ix + _para(c, iy, ix, 1)
    H, W = a.shape
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return float(dy), float(dx)


def _para(c, iy, ix, axis):
    if axis == 0:
        y0, y1, y2 = c[(iy - 1) % c.shape[0], ix], c[iy, ix], c[(iy + 1) % c.shape[0], ix]
    else:
        y0, y1, y2 = c[iy, (ix - 1) % c.shape[1]], c[iy, ix], c[iy, (ix + 1) % c.shape[1]]
    return float((y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12))


def fit_affine(ref, mov, grid=4, win=None):
    """Fit a 3-parameter affine (translation + linear drift) mapping mov onto ref.

    Returns `(cx, cy)`, each a length-3 coefficient vector, same convention as `warp`.
    """
    H, W = ref.shape
    if win is None:
        win = (max(64, H // grid), max(64, W // grid))
    win = (min(win[0], H), min(win[1], W))
    pts, sh = [], []
    for iy in range(grid):
        for ix in range(grid):
            cy = int((iy + 0.5) * H / grid)
            cx = int((ix + 0.5) * W / grid)
            y0 = min(max(0, cy - win[0] // 2), H - win[0])
            x0 = min(max(0, cx - win[1] // 2), W - win[1])
            dy, dx = _pc_sub(ref[y0:y0 + win[0], x0:x0 + win[1]],
                             mov[y0:y0 + win[0], x0:x0 + win[1]])
            pts.append((cx, cy))
            sh.append((dx, dy))
    P = np.array(pts, float)
    S = np.array(sh, float)
    A = np.stack([np.ones(len(P)), (P[:, 0] - W / 2) / W, (P[:, 1] - H / 2) / H], 1)
    cx = np.linalg.lstsq(A, S[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, S[:, 1], rcond=None)[0]
    return cx, cy


def refine_shift(ref, img, shift, radius=0.6, step=0.2):
    """Grid-search a small extra translation (SSD) to correct sub-pixel bias."""
    H, W = ref.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = (yy > 8) & (yy < H - 8) & (xx > 8) & (xx < W - 8)      # 避开边界
    best, best_s = None, None
    for dy in np.arange(-radius, radius + 1e-9, step):
        for dx in np.arange(-radius, radius + 1e-9, step):
            cx = np.array([shift[0] + dx, 0.0, 0.0])
            cy = np.array([shift[1] + dy, 0.0, 0.0])
            w = warp(img, cx, cy)
            if w.ndim > 2:
                d = ((w - ref)[..., 0])[m]
            else:
                d = (w - ref)[m]
            sc = float(np.mean(d * d))
            if best_s is None or sc < best_s:
                best_s, best = sc, (shift[0] + dx, shift[1] + dy)
    return best


def warp(img, cx, cy):
    """Bilinear warp by the affine coefficients from `fit_affine`."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = -(cx[0] + cx[1] * (xx - W / 2) / W + cx[2] * (yy - H / 2) / H)
    sy = -(cy[0] + cy[1] * (xx - W / 2) / W + cy[2] * (yy - H / 2) / H)
    gx = np.clip(xx + sx, 0, W - 1.001)
    gy = np.clip(yy + sy, 0, H - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    extra = (None,) * (img.ndim - 2)          # 兼容灰度(H,W)与彩色(H,W,C)
    fx = (gx - x0).reshape((gx - x0).shape + extra)
    fy = (gy - y0).reshape((gy - y0).shape + extra)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def align_stack(frames, ref_index=None, grid=4, refine=True):
    """Register every frame to one reference frame. Returns (aligned stack, shifts)."""
    frames = np.asarray(frames, np.float32)
    n = len(frames)
    if n == 0:
        raise ValueError("no frames")
    ri = n // 2 if ref_index is None else int(ref_index)
    ref_l = luma(frames[ri])
    aligned, shifts = [], []
    for i in range(n):
        cx, cy = fit_affine(ref_l, luma(frames[i]), grid=grid)
        if refine and i != ri:
            dx, dy = refine_shift(ref_l, luma(frames[i]), (float(cx[0]), float(cy[0])))
            cx = np.array([dx, 0.0, 0.0])
            cy = np.array([dy, 0.0, 0.0])
        aligned.append(warp(frames[i], cx, cy))
        shifts.append((float(cx[0]), float(cy[0])))
    return np.stack(aligned), np.array(shifts), ri


# ---------------------------------------------------------------------- fusion
def fuse(frames, sigma=None, single_denoise=True, consistency=True, grid=4,
         polish=False):
    """Fuse a burst into one image. Returns `(out, info)`.

    frames : (N, H, W) or (N, H, W, C) float in 0..1
    sigma  : noise level; estimated from the reference frame when None
    single_denoise : also denoise the reference frame and use it as the fallback
    consistency    : blend by temporal variance (moving regions keep the single frame)
    """
    frames = np.asarray(frames, np.float32)
    n = len(frames)
    if n < 2:
        raise ValueError("need at least 2 frames")
    if sigma is None:
        sigma = estimate_sigma(frames[n // 2])
    sigma = max(float(sigma), 1e-4)

    aligned, shifts, ri = align_stack(frames, grid=grid)
    med = np.median(aligned, 0)
    res = np.abs(aligned - med)
    w = np.exp(-(res / (2 * sigma)) ** 2)
    merged = (aligned * w).sum(0) / (w.sum(0) + 1e-6)

    ref = frames[ri]
    if single_denoise and _M is not None:
        single, _ = _M.denoise(ref, sigma=sigma)
        single = np.clip(single, 0, 1)
    else:
        single = ref

    info = {"frames": int(n), "sigma": float(sigma), "shifts": shifts}
    if not consistency:
        info["merged_fraction"] = 1.0
        return np.clip(merged, 0, 1).astype(np.float32), info

    var = ((aligned - merged) ** 2).mean(0)
    if var.ndim == 3:
        var = var.mean(-1)
    # 绝对阈值：对齐后静止区域的时域方差约为 σ²(1-1/N)，运动区域远大于此。
    # 用分位数会强行让固定比例的像素回退单帧，静止场景反而变差。
    thr = 2.0 * sigma ** 2
    mask = (var < thr).astype(np.float32)
    mask = gaussian_blur(mask, 2.0)
    if merged.ndim > 2:                        # 彩色时把掩码扩到通道维
        mask = mask[..., None]
    out = mask * merged + (1 - mask) * single
    info["merged_fraction"] = float(mask.mean())
    out = np.clip(out, 0, 1)
    if polish and _M is not None:
        # 融合已经把噪声压下去 √N，再走一遍单帧降噪继续压低
        out, _ = _M.denoise(out, sigma=float(sigma) / max(np.sqrt(info["frames"]), 1.0))
        out = np.clip(out, 0, 1)
    return out.astype(np.float32), info


# ------------------------------------------------------------------ video I/O
def find_ffmpeg():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_video_frames(path, t_sec=1.0, count=8, ffmpeg=None):
    """Read `count` consecutive frames around `t_sec` from a video (float 0..1)."""
    ff = ffmpeg or find_ffmpeg()
    probe = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True)
    png = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t_sec,
                          "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("cannot decode video frame")
    H = int.from_bytes(png[20:24], "big")
    W = int.from_bytes(png[16:20], "big")
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", probe.stderr)
    fps = float(m.group(1)) if m else 25.0
    start = max(0.0, t_sec - (count // 2) / fps)
    raw = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % start,
                          "-i", path, "-frames:v", str(count), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(raw) // (H * W * 3)
    if n < 2:
        raise RuntimeError("not enough frames")
    return np.frombuffer(raw[:n * H * W * 3], np.uint8).reshape(n, H, W, 3).astype(np.float32) / 255.0, fps


def auto_count(path, t_sec=1.0, ffmpeg=None):
    """Pick a frame count from inter-frame motion: still scene -> more frames."""
    frames, _ = read_video_frames(path, t_sec, 9, ffmpeg)
    if len(frames) < 3:
        return 3
    m = float(np.mean([np.abs(luma(frames[i]) - luma(frames[i - 1])).mean()
                       for i in range(1, len(frames))])) * 255
    for thr, n in ((1.0, 16), (2.0, 12), (4.0, 8), (8.0, 4)):
        if m < thr:
            return n
    return 3


def from_video(path, t_sec=1.0, count=None, **kw):
    """Convenience: read frames from a video and fuse them."""
    if count is None:
        count = auto_count(path, t_sec)
    frames, fps = read_video_frames(path, t_sec, count)
    out, info = fuse(frames, **kw)
    info["fps"] = fps
    return out, info
