"""NPSS-Burst — fuse a burst of consecutive frames into one clean photo.

Consecutive frames are *complementary*: the content is the same, the noise is independent.
Running each frame along its own motion trajectory and merging them averages the noise down
while keeping the detail — something a single-frame denoiser cannot do.

Pipeline
    1. block-matching flow between consecutive frames (full-resolution by default; the
       original 1/4-resolution pyramids cannot see 1-2 px jitter at all)
    2. chain the flow along the trajectory and warp every contributing frame onto the target
    3. robust weighted merge (median reference -> Gaussian weights)
    4. optional single-frame denoise of the merged result (this is what pushes PSNR past 40)
"""
from __future__ import annotations

import subprocess

import numpy as np

import npss_image as M
_GRID = {}


def _grid(H, W):
    if (H, W) not in _GRID:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _GRID[(H, W)] = (yy, xx)
    return _GRID[(H, W)]


def _box_down(a, f):
    if f <= 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def _box3(x):
    """3x3 box sum（用于流场的有效性归一化填补）。"""
    p = np.pad(x, 1, mode="edge")
    h, w = x.shape
    return (p[0:h, 1:w+1] + p[1:h+1, 0:w] + p[1:h+1, 1:w+1] + p[1:h+1, 2:w+2]
            + p[0:h, 0:w] + p[0:h, 2:w+2] + p[2:h+2, 0:w] + p[2:h+2, 1:w+1]
            + p[2:h+2, 2:w+2])


def bm_flow(a, b, B=8, R=6, scale=4):
    """Block-matching flow from a to b.

    The search error is normalised by the number of *overlapping* pixels, and blocks with
    no overlap are forced to zero. Without that, border blocks pick the first search
    offset (-R) because their error is constant, and the constant then smears over the
    whole field through the smoothing step (measured: a zero-displacement pair reported
    -4.65 px before this fix).
    """
    ga = _box_down(M.luma(a), scale)
    gb = _box_down(M.luma(b), scale)
    H, W = ga.shape
    ny, nx = H // B, W // B
    A = ga[:ny * B, :nx * B].reshape(ny, B, nx, B)
    best = np.full((ny, nx), np.inf, np.float32)
    U = np.zeros((ny, nx), np.float32)
    V = np.zeros_like(U)
    canvas = np.zeros((ny * B, nx * B), np.float32)
    maskc = np.zeros((ny * B, nx * B), np.float32)
    full = float(B * B)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            sub = gb[y0:y1, x0:x1]
            m0, m1 = -min(0, dy), -min(0, dx)
            canvas[:] = 0.0
            maskc[:] = 0.0
            hh, ww = sub.shape
            hh2 = min(hh, ny * B - m0)
            ww2 = min(ww, nx * B - m1)
            if hh2 <= 0 or ww2 <= 0:
                continue
            canvas[m0:m0 + hh2, m1:m1 + ww2] = sub[:hh2, :ww2]
            maskc[m0:m0 + hh2, m1:m1 + ww2] = 1.0
            cnt = maskc.reshape(ny, B, nx, B).sum((1, 3))
            err = ((A - canvas.reshape(ny, B, nx, B)) ** 2).sum((1, 3))
            en = err / np.maximum(cnt, 1.0)
            ok = (cnt > 0.9 * full) & (en < best)
            best[ok] = en[ok]
            U[ok] = dx
            V[ok] = dy
    # 平坦块（无纹理）在块匹配里有孔径问题：误差处处为 0，会锁定到第一个搜索偏移。
    # 剔除这些块，再用邻域填补，避免把常数偏移抹到全图。
    varA = A.var(axis=(1, 3))
    thr_var = max(1e-9, (0.01 * float(np.std(ga))) ** 2)
    valid = varA > thr_var
    if valid.sum() >= 4:
        Wv = valid.astype(np.float32)
        U = np.where(valid, U, 0.0).astype(np.float32)
        V = np.where(valid, V, 0.0).astype(np.float32)
        for _ in range(4):                       # 由已知块向外扩散填补
            Wb = _box3(Wv)
            Ub = _box3(U * Wv)
            Vb = _box3(V * Wv)
            fill = (Wb > 1e-6) & (~valid)
            U[fill] = Ub[fill] / Wb[fill]
            V[fill] = Vb[fill] / Wb[fill]
            Wv[fill] = 1.0
            valid = valid | fill
    U = np.repeat(np.repeat(U, B, 0), B, 1)[:H, :W]
    V = np.repeat(np.repeat(V, B, 0), B, 1)[:H, :W]
    U = M.gaussian_blur(U, 2.0) * scale
    V = M.gaussian_blur(V, 2.0) * scale
    fu = np.repeat(np.repeat(U, scale, 0), scale, 1)
    fv = np.repeat(np.repeat(V, scale, 0), scale, 1)
    fu = np.pad(fu, ((0, max(0, a.shape[0] - fu.shape[0])), (0, max(0, a.shape[1] - fu.shape[1]))), mode="edge")
    fv = np.pad(fv, ((0, max(0, a.shape[0] - fv.shape[0])), (0, max(0, a.shape[1] - fv.shape[1]))), mode="edge")
    return M.gaussian_blur(fu, 2.0), M.gaussian_blur(fv, 2.0)


def warp(img, u, v):
    """双线性重采样（flat take）。"""
    H, W = img.shape[:2]
    yy, xx = _grid(H, W)
    gx = np.clip(xx + u, 0, W - 1.001)
    gy = np.clip(yy + v, 0, H - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    fx = gx - x0
    fy = gy - y0
    flat = img.reshape(-1, img.shape[2])
    idx = (y0.ravel() * W + x0.ravel())
    w00 = ((1 - fx) * (1 - fy)).ravel()[:, None]
    w01 = (fx * (1 - fy)).ravel()[:, None]
    w10 = ((1 - fx) * fy).ravel()[:, None]
    w11 = (fx * fy).ravel()[:, None]
    out = (flat.take(idx, axis=0) * w00
           + flat.take(idx + 1, axis=0) * w01
           + flat.take(idx + W, axis=0) * w10
           + flat.take(idx + W + 1, axis=0) * w11)
    return out.reshape(H, W, img.shape[2])


def accumulate_weighted(frames, flows, idx, radius=3, sigma=0.04):
    """沿轨迹对齐 ±radius 帧，残差加权合并。"""
    H, W = frames[idx].shape[:2]
    st = [frames[idx]]
    us = np.zeros((H, W), np.float32)
    vs = np.zeros_like(us)
    for k in range(1, radius + 1):
        j = idx + k
        if j >= len(frames):
            break
        u, v = flows[j - 1]
        us = us + u
        vs = vs + v
        st.append(warp(frames[j], us, vs))
    us = np.zeros((H, W), np.float32)
    vs = np.zeros_like(us)
    for k in range(1, radius + 1):
        j = idx - k
        if j < 0:
            break
        u, v = flows[j]
        us = us - u
        vs = vs - v
        st.append(warp(frames[j], us, vs))
    S_ = np.stack(st).astype(np.float32)
    med = np.median(S_, 0)
    res = np.abs(S_ - med).mean(-1)
    w = np.exp(-(res / (2 * sigma)) ** 2)
    return np.clip((S_ * w[..., None]).sum(0) / (w.sum(0)[..., None] + 1e-6), 0, 1).astype(np.float32)


# --------------------------------------------------------------- 独立入口
def fuse(frames, idx=None, radius=None, sigma=None, polish=True, flow_scale=0):
    """沿轨迹把一段连续帧累积成一张干净图。

    frames : (N,H,W,3) float 0..1
    idx    : 参考帧（默认中间帧）
    radius : 向前/向后各用几帧（默认用满整段）
    polish : 累积后再走一遍单帧降噪（这是能不能到 40 dB 的关键一步）
    """
    import numpy as _np
    frames = _np.asarray(frames, _np.float32)
    n = len(frames)
    if n < 2:
        raise ValueError("need >= 2 frames")
    if idx is None:
        idx = n // 2
    if radius is None:
        radius = max(idx, n - 1 - idx)
    if sigma is None:
        y = M.luma(frames[idx])
        sigma = float(_np.median(_np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)

    if flow_scale == 0:
        flows = [bm_flow_pyramid(frames[i], frames[i + 1]) for i in range(n - 1)]
    else:
        flows = [bm_flow(frames[i], frames[i + 1], scale=flow_scale) for i in range(n - 1)]
    out = accumulate_weighted(frames, flows, idx, radius, sigma)
    info = {"frames": int(n), "radius": int(radius), "sigma": float(sigma), "flow_scale": "pyramid" if flow_scale == 0 else int(flow_scale),
            "accumulated": int(min(n, idx + radius + 1) - max(0, idx - radius))}
    if polish:
        # 累积把噪声降到 σ/√K，再按这个等效噪声走一遍单帧降噪
        k = max(1, info["accumulated"])
        d, _ = M.denoise(out, sigma=float(sigma) / _np.sqrt(k))
        out = _np.clip(d, 0, 1)
        info["polished"] = True
    return out.astype(_np.float32), info


# ------------------------------------------------------------------ video I/O
def find_ffmpeg():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def read_video_frames(path, t_sec=1.0, count=8, ffmpeg=None):
    """Read `count` consecutive frames around `t_sec` (float 0..1)."""
    import re
    ff = ffmpeg or find_ffmpeg()
    probe = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True)
    png = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t_sec,
                          "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("cannot decode video frame")
    H = int.from_bytes(png[20:24], "big")
    W = int.from_bytes(png[16:20], "big")
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
    frames, _ = read_video_frames(path, t_sec, 9, ffmpeg)
    if len(frames) < 3:
        return 3
    m = float(np.mean([np.abs(M.luma(frames[i]) - M.luma(frames[i - 1])).mean()
                       for i in range(1, len(frames))])) * 255
    for thr, n in ((1.0, 16), (2.0, 12), (4.0, 8), (8.0, 4)):
        if m < thr:
            return n
    return 3


def from_video(path, t_sec=1.0, count=None, **kw):
    if count is None:
        count = auto_count(path, t_sec)
    frames, fps = read_video_frames(path, t_sec, count)
    out, info = fuse(frames, **kw)
    info["fps"] = fps
    return out, info

def bm_flow_pyramid(a, b, B=8):
    """Coarse-to-fine flow: ±24 px reach at 1/4 scale, then ±6 px refinement at full scale.

    A single scale cannot do both: at 1/4 resolution a 1-pixel jitter is invisible
    (measured -0.0 px, i.e. no correction at all), while at full resolution the ±6 px
    search window cannot follow real camera motion.
    """
    u0, v0 = bm_flow(a, b, B=B, R=6, scale=4)
    b1 = warp(b, u0, v0)
    u1, v1 = bm_flow(a, b1, B=B, R=6, scale=1)
    return u0 + u1, v0 + v1
