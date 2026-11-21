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
from PIL import Image

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
    """双线性重采样（flat take）。支持灰度 (H,W) 与彩色 (H,W,C)。"""
    if img.ndim == 2:
        return warp(img[..., None], u, v)[..., 0]
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
    # 权重尺度不能小于实际残差：干净视频（σ 估计到 1/255）时 exp(-(res/2σ)²) 会全部塌成 0，
    # 归一化就变成 0/0，出现黑点。用残差中位数兜底。
    sig_w = max(float(sigma), float(np.median(res)), 1e-3)
    w = np.exp(-(res / (2 * sig_w)) ** 2)
    sw = w.sum(0)
    merged = (S_ * w[..., None]).sum(0) / np.maximum(sw, 1e-6)[..., None]
    bad = sw < 1e-6                       # 权重全灭的像素退回普通平均
    if np.any(bad):
        merged[bad] = S_.mean(0)[bad]
    return np.clip(merged, 0, 1).astype(np.float32)


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

# ------------------------------------------------- continuous temporal denoise
def denoise_sequence(frames, radius=3, sigma=None, flows=None, polish=False):
    """Denoise a whole sequence **continuously**: every output frame uses its temporal
    neighbours (radius frames before and after), accumulating along their motion
    trajectories. This is the time-series complement, not a burst-to-one-still fusion:
    the output has the same number of frames as the input.

    edges shrink the window automatically (`radius` is clamped per frame).
    Returns `(out_frames, info)`.
    """
    frames = np.asarray(frames, np.float32)
    n = len(frames)
    if n == 0:
        raise ValueError("empty sequence")
    if sigma is None:
        mid = frames[n // 2]
        y = M.luma(mid)
        sigma = float(np.median(np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)
    if flows is None:
        flows = [bm_flow_pyramid(frames[i], frames[i + 1]) for i in range(n - 1)]
    out = np.empty_like(frames)
    for i in range(n):
        out[i] = accumulate_weighted(frames, flows, i, min(radius, max(i, n - 1 - i)), sigma)
        if polish and M is not None:
            d, _ = M.denoise(out[i], sigma=float(sigma) / np.sqrt(2 * min(radius, max(i, n - 1 - i)) + 1))
            out[i] = np.clip(d, 0, 1)
    return out, {"frames": n, "radius": int(radius), "sigma": float(sigma)}


# ------------------------------------------------------------------ video out
def write_video(path, frames, fps=30, crf=18):
    """Write float frames (N,H,W,3) to an H.264 mp4 via ffmpeg."""
    ff = find_ffmpeg()
    n, h, w = frames.shape[:3]
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        p.stdin.close()
    finally:
        p.wait()
    return path

# ------------------------------------------------------------ noise injection
def add_noise(frames, sigma, seed=None):
    """Add white Gaussian noise to a sequence (for testing against a clean reference).

    sigma is on the 0-255 scale. Returns float32 frames in 0..1.
    """
    frames = np.asarray(frames, np.float32)
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, float(sigma) / 255.0, frames.shape).astype(np.float32)
    return np.clip(frames + n, 0, 1)

# ------------------------------------------------- 3D sparse reconstruction
def _dct_mat(n):
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * m + 1) * k / (2.0 * n))
    d[0] /= np.sqrt(2.0)
    return d


def _sparse_window(win, Ks, Kt, k0, sigma, B):
    """One K-frame window -> sparse 3D reconstruction of its centre frame.

    win: (K, H, W) float. Transform: 2D DCT (space) x 1D DCT (time), threshold, invert.
    """
    K, H, W = win.shape
    h8, w8 = H // B * B, W // B * B
    ny, nx = h8 // B, w8 // B
    blk = (win[:, :h8, :w8].reshape(K, ny, B, nx, B).transpose(1, 3, 2, 4, 0)
           .reshape(ny, nx, B, B, K).astype(np.float64))
    C = np.einsum("ua,xyabt->xyubt", Ks, blk, optimize=True)
    C = np.einsum("vb,xyubt->xyuvt", Ks, C, optimize=True)
    C = np.einsum("kt,xyuvt->xyuvk", Kt, C, optimize=True)
    th = k0 * sigma          # 正交 DCT：每个系数的噪声标准差都是 σ，不随维度放大
    g = np.abs(C) > th
    g[0, 0, 0, 0, :] = True                    # 空间 DC 的时间分量全留
    g[..., 0] = True                           # 时间 DC 全留（静态内容在这里）
    C = C * g
    R = np.einsum("kt,xyuvk->xyuvt", Kt, C, optimize=True)
    R = np.einsum("vb,xyuvt->xyubt", Ks, R, optimize=True)
    R = np.einsum("ua,xyubt->xyabt", Ks, R, optimize=True)
    rec = R.transpose(0, 1, 2, 3, 4).reshape(ny, B, nx, B, K)
    mid = K // 2
    rec = R.reshape(ny, nx, B, B, K)[..., mid].transpose(0, 2, 1, 3).reshape(h8, w8)
    out = np.zeros((H, W), np.float32)
    out[:h8, :w8] = rec
    return out


def sparse_3d(seq, K=3, k0=3.0, sigma=None, B=8):
    """Sparse reconstruction over a persistent time series (3D transform domain).

    For every frame a K-frame tensor is formed, transformed by 2D DCT (space) x 1D DCT
    (time), sparsely thresholded, and inverted back to the centre frame. Static content piles
    up in the temporal DC coefficient; noise spreads into the temporal high frequencies and is
    thresholded away. Motion produces *large* temporal coefficients that survive the
    threshold, so moving regions keep their detail instead of being averaged into a blur.

    seq: (N, H, W) or (N, H, W, C) float 0..1. K=3 is the sweet spot.
    """
    seq = np.asarray(seq, np.float32)
    n = len(seq)
    if n < K:
        K = n
    single = seq.ndim == 3
    if single:
        seq = seq[..., None]
    H, W, C = seq.shape[1:]
    if sigma is None:
        y = M.luma(seq[n // 2])
        sigma = float(np.median(np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)
    Ks, Kt = _dct_mat(B), _dct_mat(K)
    half = K // 2
    out = np.zeros_like(seq)
    for c in range(C):
        pad = np.pad(seq[..., c], ((half, K - 1 - half), (0, 0), (0, 0)), mode="edge")
        for i in range(n):
            out[i, ..., c] = _sparse_window(pad[i:i + K], Ks, Kt, k0, sigma, B)
    out = np.clip(out, 0, 1).astype(np.float32)
    return (out[..., 0] if single else out), {"frames": n, "K": int(K), "sigma": float(sigma)}

# --------------------------------------------- low-rank / sparse decomposition
def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _svt(x, t):
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    sv = np.maximum(sv - t, 0.0)
    return (u * sv) @ vt


def rpca(seq, scale=4, iters=60, lam=None):
    """Robust PCA: V = L + S, L low-rank (background), S sparse (moving subject).

    The low-rank part aggregates the *whole* sequence, so background noise is averaged over
    every frame rather than a short window — that is where the 40 dB+ static-region numbers
    come from. The sparse part keeps the subject intact instead of averaging it into a blur.

    Computed on a 1/scale grid for speed, then the low-rank part is upsampled and the
    residual (subject + noise) is recovered at full resolution.
    """
    seq = np.asarray(seq, np.float32)
    single = seq.ndim == 3
    if single:
        seq = seq[..., None]
    n, H, W, C = seq.shape
    sc = max(1, int(scale))
    h, w = H // sc, W // sc
    small = seq[:, :h * sc, :w * sc].reshape(n, h, sc, w, sc, C).mean((2, 4))   # (n,h,w,C)
    M = small.reshape(n, -1).T.astype(np.float64)         # (D,n)
    D = M.shape[0]
    if lam is None:
        lam = 1.0 / np.sqrt(max(D, n))
    u, sv, vt = np.linalg.svd(M, full_matrices=False)
    norm2 = float(sv[0]) if sv.size else 1.0
    Y = M / max(norm2, 1e-9)
    mu = 1.25 / max(norm2, 1e-9)
    S = np.zeros_like(M)
    L = np.zeros_like(M)
    for _ in range(iters):
        L = _svt(M - S + Y / mu, 1.0 / mu)
        S = _soft(M - L + Y / mu, lam / mu)
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * 1.05, mu * 1.05)
        if float(np.abs(Z).max()) < 1e-6:
            break
    Ls = L.T.reshape(n, h, w, C)
    # 上采样低秩部分
    Lb = np.stack([np.stack([np.asarray(Image.fromarray(np.float32(Ls[i, ..., c])).resize((W, H), Image.BICUBIC))
                             for c in range(C)], -1) for i in range(n)])
    S_full = seq - np.clip(Lb, 0, 1)
    info = {"frames": n, "scale": sc, "lam": float(lam), "rank_energy": float(sv[:5].sum() / max(sv.sum(), 1e-9))}
    return np.clip(Lb, 0, 1).astype(np.float32), S_full.astype(np.float32), info


def denoise_rpca(seq, sigma=None, subject_denoise=True, scale=4, iters=60):
    """Background reconstruction + subject compositing.

    L (low-rank background, noise averaged over the whole clip) + denoised S (sparse
    subject). This is the "背景重建 + 人物合成" route recorded in NPSS-Enhancer.
    """
    Lb, S_full, info = rpca(seq, scale=scale, iters=iters)
    n = len(Lb)
    if subject_denoise:
        sd = np.stack([np.clip(M.denoise(S_full[i], sigma=sigma)[0], -1, 1) for i in range(n)])
    else:
        sd = S_full
    out = np.clip(Lb + sd, 0, 1).astype(np.float32)
    info["subject_denoise"] = bool(subject_denoise)
    return out, info

# ------------------------------------------- flow-based subject/background split
def camera_model(u, v, iters=3):
    """Fit the global camera motion (affine) to a flow field, robustly."""
    H, W = u.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = ((xx - W / 2) / W).ravel()
    yn = ((yy - H / 2) / H).ravel()
    A = np.stack([np.ones_like(xn), xn, yn], 1)
    uu, vv = u.ravel().astype(np.float64), v.ravel().astype(np.float64)
    w = np.ones_like(uu)
    cu = cv = None
    for _ in range(iters):
        Aw = A * w[:, None]
        cu = np.linalg.lstsq(Aw, uu * w, rcond=None)[0]
        cv = np.linalg.lstsq(Aw, vv * w, rcond=None)[0]
        r = np.hypot(uu - A @ cu, vv - A @ cv)
        sc = max(float(np.median(r)), 0.25)   # 残差极小时不重加权，否则权重塌成 0 后 lshtsq 溢出
        w = 1.0 / (1.0 + (r / (3 * sc)) ** 2)      # 软降权，抑制主体对拟合的拉扯
    return (A @ cu).reshape(H, W).astype(np.float32), (A @ cv).reshape(H, W).astype(np.float32)


def motion_mask(u, v, thr_px=1.0, dilate=3):
    """Subject mask: flow that deviates from the *global camera motion*.

    Using the median flow as the camera model does not work — handheld drift makes the field
    spatially uneven, so nearly everything looks like a subject (measured 87% of pixels).
    An affine fit removes the camera motion properly and leaves only genuine movement.
    """
    cu, cv = camera_model(u, v)
    dev = np.hypot(u - cu, v - cv)
    m = (dev > thr_px).astype(np.float32)
    if dilate > 0:
        m = (M.gaussian_blur(m, dilate) > 0.05).astype(np.float32)
    return m


def flow_split(seq, sigma=None, thr_sigma=2.0, dilate=3, subject_denoise=True, two_pass=True):
    """Optical-flow alignment + background subtraction + subject compositing.

    1. flow of every frame against the reference; align
    2. background = per-pixel mean over the frames where the pixel is *static*
       (noise drops by √N over the whole clip — this is the 40 dB static region)
    3. subject mask = residual against that background (standard background subtraction;
       the raw flow field is too rough here to serve as a mask directly)
    4. output = background + the denoised subject composited back in
    """
    seq = np.asarray(seq, np.float32)
    single = seq.ndim == 3
    if single:
        seq = seq[..., None]
    n, H, W, C = seq.shape
    if sigma is None:
        y = M.luma(seq[n // 2])
        sigma = float(np.median(np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)

    ri = n // 2
    ref_l = M.luma(seq[ri])
    flows, aligned = [], []
    for i in range(n):
        if i == ri:
            z = np.zeros((H, W), np.float32); u, v = z, z
        else:
            u, v = bm_flow_pyramid(ref_l, M.luma(seq[i]))
        flows.append((u, v))
        aligned.append(warp(seq[i], u, v))
    aligned = np.stack(aligned)

    # 第一遍：全帧均值当背景；第二遍：剔除主体后重新平均
    bg = aligned.mean(0)
    for _ in range(2 if two_pass else 1):
        res = np.abs(aligned - bg).mean(-1)
        masks = (res > thr_sigma * sigma).astype(np.float32)
        if dilate > 0:
            masks = np.stack([(M.gaussian_blur(masks[i], dilate) > 0.05).astype(np.float32)
                              for i in range(n)])
        w = (1.0 - masks)[..., None]
        wsum = w.sum(0)
        bg = (aligned * w).sum(0) / np.maximum(wsum, 1.0)
        hole = (wsum < 0.5)[..., 0]
        if np.any(hole):
            bg[hole] = aligned.mean(0)[hole]
    background = np.clip(bg, 0, 1).astype(np.float32)

    out = np.empty_like(seq)
    for i in range(n):
        u, v = flows[i]
        bg_i = warp(background, -u, -v)
        mk = masks[i][..., None]
        fg = seq[i]
        if subject_denoise:
            fg = np.clip(M.denoise(fg, sigma=sigma)[0], 0, 1)
        out[i] = np.clip(mk * fg + (1 - mk) * bg_i, 0, 1)
    info = {"frames": n, "sigma": float(sigma), "subject_frac": float(masks.mean())}
    return (out[..., 0] if single else out), info

# ------------------------------------------------- temporal completion in DCT
def complete_dct(aligned, K=3, B=8, stat="median"):
    """Complete every frame from its temporal neighbours **in the sparse (DCT) domain**.

    Not a subject/background split and not pixel averaging: for each 8x8 block and each DCT
    coefficient, the K frames are independent observations of the same coefficient. Taking a
    robust estimate across them cancels noise (which is inconsistent) while keeping structure
    (which is consistent). Reconstructing from those coefficients avoids the blur that pixel
    averaging leaves behind.

    aligned: (N,H,W) or (N,H,W,C), already registered to a common frame.
    """
    aligned = np.asarray(aligned, np.float32)
    single = aligned.ndim == 3
    if single:
        aligned = aligned[..., None]
    n, H, W, C = aligned.shape
    K = max(1, min(int(K), n))
    h8, w8 = H // B * B, W // B * B
    ny, nx = h8 // B, w8 // B
    Ks = _dct_mat(B)
    half = K // 2
    out = np.empty_like(aligned)
    for c in range(C):
        pad = np.pad(aligned[..., c], ((half, K - 1 - half), (0, 0), (0, 0)), mode="edge")
        win = np.stack([pad[i:i + K] for i in range(n)], 0)          # (n,K,H,W)
        win = win[:, :, :h8, :w8]
        blk = (win.reshape(n, K, ny, B, nx, B).transpose(0, 2, 4, 1, 3, 5)
               .reshape(n, ny, nx, K, B, B))                        # (n,ny,nx,K,B,B)
        # 正向：D = Ks · X · Ksᵀ
        D = np.einsum("ab,nxykbc->nxykac", Ks, blk, optimize=True)
        D = np.einsum("cd,nxykad->nxykac", Ks, D, optimize=True)
        Dc = (np.median(D, axis=3) if stat == "median" else np.mean(D, axis=3))[..., None]
        # 逆向：X = Ksᵀ · D · Ks
        R = np.einsum("ba,nxykbc->nxykac", Ks, Dc, optimize=True)
        R = np.einsum("dc,nxykad->nxykac", Ks, R, optimize=True)
        rec = R[:, :, :, 0].transpose(0, 1, 3, 2, 4).reshape(n, h8, w8)
        full = np.zeros((n, H, W), np.float32)
        full[:, :h8, :w8] = rec
        out[..., c] = full
    out = np.clip(out, 0, 1).astype(np.float32)
    return (out[..., 0] if single else out), {"frames": n, "K": K, "stat": stat}

# ------------------------------------- three-layer temporal completion (IDM)
def idm_temporal(aligned, sigma=None, K=3, B=8, tau=2.0, min_consistent=0.6):
    """Three-layer temporal completion, block-adaptive (蒸馏 / 升华 / 背景 / 重建).

    * 蒸馏 (distill): keep only blocks whose content is a *critical point* over time —
      stable enough that the temporal neighbours really do observe the same thing.
    * 升华 (sublime): inside such blocks, take the temporal median (the robust complement).
    * 背景 (background): everything else keeps the frame's own content, so moving detail is
      never averaged away.
    * 重建: no transform to invert — the decision is made per 8x8 block.

    The point is that this is completion, not fusion: a block is completed only when the
    neighbours agree about it; otherwise nothing is borrowed at all.
    """
    aligned = np.asarray(aligned, np.float32)
    single = aligned.ndim == 3
    if single:
        aligned = aligned[..., None]
    n, H, W, C = aligned.shape
    K = max(1, min(int(K), n))
    if sigma is None:
        y = M.luma(aligned[n // 2])
        sigma = float(np.median(np.abs(y - M.gaussian_blur(y, 1.2))) / 0.6745)
    sigma = max(float(sigma), 1e-4)
    half = K // 2
    h8, w8 = H // B * B, W // B * B
    ny, nx = h8 // B, w8 // B
    pad = np.pad(aligned, ((half, K - 1 - half), (0, 0), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(aligned)
    for i in range(n):
        win = pad[i:i + K, :h8, :w8]                       # (K,h8,w8,C)
        blocks = win.reshape(K, ny, B, nx, B, C).transpose(1, 3, 0, 2, 4, 5)  # (ny,nx,K,B,B,C)
        med = np.median(blocks, axis=2)                    # 升华：时间中值 (ny,nx,B,B,C)
        dev = np.abs(blocks - med[:, :, None]).mean(axis=(-1, -2, -3)) / sigma  # (ny,nx,K)
        agree = (dev < tau).mean(axis=2)                   # (ny,nx) 一致帧比例
        consistent = agree >= min_consistent
        own = blocks[:, :, half]                           # 本帧原内容
        merged = np.where(consistent[:, :, None, None, None], med, own)
        rec = merged.transpose(0, 2, 1, 3, 4).reshape(h8, w8, C)   # (ny,B,nx,B,C)
        out[i, :h8, :w8] = rec
        out[i, h8:] = aligned[i, h8:]
        out[i, :, w8:] = aligned[i, :, w8:]
    out = np.clip(out, 0, 1).astype(np.float32)
    return (out[..., 0] if single else out), {"frames": n, "K": K, "tau": tau,
                                              "consistent_frac": float(np.mean(consistent))
                                              if 'consistent' in dir() else 0.0}
