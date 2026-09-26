# Imagination corpus: a procedural dream-journal of forms.
# Generates N seeded SDF "forms" from a grammar (primitives fused with random-k
# smin, domain deformations, symmetry folds, sinusoid displacement), samples
# each one (uniform + near-surface via Newton projection + far shell), and
# stores float16 [N, P, 4] in an .npz for train_imagination.py. Also raymarches
# a few forms to previews/corpus/*.png with a tiny numpy marcher — Gate 1 is a
# human (or agent) eyeballing those.
# Run: .venv/Scripts/python scripts/imagination_corpus.py [--n 2000] [--previews 12]
import argparse
import json
import struct
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BOUND = 1.35          # sampling ball radius; forms are designed to sit within ~1.2
P_PER_SHAPE = 8192    # samples per form
DATA_DIR = Path("data")
PREVIEW_DIR = Path("previews/corpus")


# ---------------------------------------------------------------- PNG writer
def write_png(path: Path, rgb: np.ndarray) -> None:
    """rgb: uint8 [H, W, 3]. Pure-stdlib PNG (no PIL in this venv)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    def chunk(tag: bytes, payload: bytes) -> bytes:
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


# ------------------------------------------------------------- SDF primitives
# All take p: (M, 3) array (already translated/rotated into local frame).
def sd_sphere(p, r):
    return np.linalg.norm(p, axis=1) - r

def sd_ellipsoid(p, radii):
    k0 = np.linalg.norm(p / radii, axis=1)
    k1 = np.linalg.norm(p / (radii * radii), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def sd_capsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r

def sd_rbox(p, he, r):
    q = np.abs(p) - he
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0) - r

def sd_torus(p, R, r):
    q = np.stack([np.linalg.norm(p[:, [0, 2]], axis=1) - R, p[:, 1]], axis=1)
    return np.linalg.norm(q, axis=1) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1.0 - h)

def smax(a, b, k):
    return -smin(-a, -b, k)


def random_rotation(rng):
    m = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def make_primitive(rng, scale_lo=0.15, scale_hi=0.55, near_origin=0.45):
    """Returns (sdf_fn, approx_radius). Frame: random offset + rotation."""
    kind = rng.choice(["sphere", "ellipsoid", "capsule", "box", "torus"], p=[0.14, 0.34, 0.3, 0.1, 0.12])
    rot = random_rotation(rng)
    off = rng.uniform(-near_origin, near_origin, size=3)
    if kind == "sphere":
        r = rng.uniform(scale_lo, scale_hi)
        base = lambda p: sd_sphere(p, r)
        rad = r
    elif kind == "ellipsoid":
        radii = rng.uniform(scale_lo * 0.6, scale_hi, size=3)
        base = lambda p: sd_ellipsoid(p, radii)
        rad = float(np.max(radii))
    elif kind == "capsule":
        r = rng.uniform(scale_lo * 0.4, scale_hi * 0.5)
        half = rng.uniform(0.1, scale_hi, size=3) * rng.choice([-1.0, 1.0], size=3)
        base = lambda p: sd_capsule(p, -half, half, r)
        rad = float(np.linalg.norm(half) + r)
    elif kind == "box":
        he = rng.uniform(scale_lo * 0.5, scale_hi * 0.8, size=3)
        r = rng.uniform(0.02, 0.12)
        base = lambda p: sd_rbox(p, he, r)
        rad = float(np.linalg.norm(he) + r)
    else:
        R = rng.uniform(scale_lo, scale_hi)
        r = rng.uniform(0.05, R * 0.6)
        base = lambda p: sd_torus(p, R, r)
        rad = R + r
    def sdf(p):
        q = (p - off) @ rot
        return base(q)
    return sdf, rad + float(np.linalg.norm(off))


def gen_shape(seed: int):
    """Seeded grammar -> (sdf_fn, approx_radius). Deterministic per seed."""
    rng = np.random.default_rng(seed)
    parts = []

    # domain deformation applied before everything else
    deform = rng.random() < 0.5
    twist_k = rng.uniform(-1.2, 1.2) if deform and rng.random() < 0.6 else 0.0
    taper_a = rng.uniform(-0.5, 0.5) if deform and rng.random() < 0.5 else 0.0
    bend_k = rng.uniform(-0.6, 0.6) if deform and rng.random() < 0.4 else 0.0

    def domain(p):
        q = p.copy()
        if twist_k:
            ang = twist_k * q[:, 1]
            c, s = np.cos(ang), np.sin(ang)
            x = q[:, 0] * c - q[:, 2] * s
            z = q[:, 0] * s + q[:, 2] * c
            q[:, 0], q[:, 2] = x, z
        if taper_a:
            q[:, 0] /= 1.0 + taper_a * q[:, 1]
            q[:, 2] /= 1.0 + taper_a * q[:, 1]
        if bend_k:
            ang = bend_k * q[:, 0]
            c, s = np.cos(ang), np.sin(ang)
            y = q[:, 1] * c - q[:, 2] * s
            z = q[:, 1] * s + q[:, 2] * c
            q[:, 1], q[:, 2] = y, z
        return q

    # symmetry fold for the main body
    u = rng.random()
    if u < 0.40:
        sym = ("none", 0)
    elif u < 0.80:
        sym = ("bilateral", 0)
    else:
        sym = ("radial", int(rng.integers(2, 6)))

    def fold(p):
        q = p.copy()
        if sym[0] == "bilateral":
            q[:, 0] = np.abs(q[:, 0])
        elif sym[0] == "radial":
            n = sym[1]
            theta = np.arctan2(q[:, 2], q[:, 0])
            sector = 2 * np.pi / n
            theta = (theta % sector) - sector / 2
            rlen = np.linalg.norm(q[:, [0, 2]], axis=1)
            q[:, 0] = rlen * np.cos(theta)
            q[:, 2] = rlen * np.sin(theta)
        return q

    # main body: 1-3 primitives fused in folded space
    n_core = int(rng.integers(1, 4))
    core_fns, core_rad = [], 0.0
    for _ in range(n_core):
        fn, rad = make_primitive(rng, 0.2, 0.6, 0.35)
        core_fns.append(fn)
        core_rad = max(core_rad, rad)
    core_k = rng.uniform(0.1, 0.45)

    def core(p):
        q = fold(domain(p))
        d = core_fns[0](q)
        for fn in core_fns[1:]:
            d = smin(d, fn(q), core_k)
        # deformations shrink distances; compensate roughly
        return d * (0.7 if deform else 1.0)

    parts.append(core)

    # appendages: 0-3 limb-like capsules anchored inside the core and reaching
    # outward (never detached debris), fused tight so they read as fins/limbs
    extras: list[tuple] = []
    for _ in range(int(rng.integers(0, 4))):
        a = rng.uniform(-0.3, 0.3, size=3)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        b = a + direction * rng.uniform(0.3, 0.75)
        r = rng.uniform(0.06, 0.18)
        fn = (lambda a=a, b=b, r=r: lambda p: sd_capsule(p, a, b, r))()
        extras.append((fn, rng.uniform(0.08, 0.22)))

    # occasional carve: a small cavity, always inside the body mass
    subtract = rng.random() < 0.2
    sub_fn = None
    if subtract:
        sub_fn, _ = make_primitive(rng, 0.08, 0.22, 0.25)
        sub_k = rng.uniform(0.04, 0.15)

    # displacement: sum of sinusoid products — gentle: fine detail is what the
    # decoder underfits, and the imagination should dream big shapes first
    n_disp = int(rng.integers(1, 4))
    disp_amp = rng.uniform(0.01, 0.035, size=n_disp)
    disp_freq = rng.uniform(2.0, 5.0, size=(n_disp, 3))
    disp_phase = rng.uniform(0, 2 * np.pi, size=(n_disp, 3))

    def sdf(p):
        d = parts[0](p)
        for fn, k in extras:
            d = smin(d, fn(domain(p)) * (0.7 if deform else 1.0), k)
        if subtract:
            d = smax(d, -sub_fn(domain(p)), sub_k)
        for i in range(n_disp):
            d = d + disp_amp[i] * np.prod(np.sin(p * disp_freq[i] + disp_phase[i]), axis=1)
        return d

    return sdf, core_rad + 0.6


def numeric_grad(fn, p, eps=2e-3):
    g = np.empty_like(p)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        g[:, k] = (fn(p + d) - fn(p - d)) / (2 * eps)
    return g


def normalized_shape(seed: int):
    """Grammar output rescaled so the form spans radius ~0.9 inside BOUND.
    Returns None for degenerate forms (no appreciable interior)."""
    sdf_raw, _ = gen_shape(seed)
    rng = np.random.default_rng(seed ^ 0x9E3779B9)
    probe = rng.uniform(-1, 1, size=(8192, 3)) * BOUND
    inside = probe[sdf_raw(probe) < 0]
    if len(inside) < 20:
        return None
    r_est = float(np.percentile(np.linalg.norm(inside, axis=1), 99))
    if r_est < 0.15:
        return None
    s = 0.9 / r_est
    return lambda p: sdf_raw(p / s) * s


def sample_shape(seed: int, p_per_shape: int = P_PER_SHAPE):
    """-> (pts float32 [P,3], sdf float32 [P]) or None if rejected."""
    rng = np.random.default_rng(seed ^ 0x5DEECE66)
    sdf = None
    for _ in range(6):
        sdf = normalized_shape(seed)
        if sdf is None:
            seed = seed * 1664525 + 1013904223  # reroll
            continue
        probe = rng.uniform(-1, 1, size=(6000, 3)) * BOUND
        frac = float(np.mean(sdf(probe) < 0))
        if 0.012 < frac < 0.45:
            break
        seed = seed * 1664525 + 1013904223  # reroll
        sdf = None
    if sdf is None:
        return None

    n_uniform = int(p_per_shape * 0.35)
    n_surf = p_per_shape - n_uniform - int(p_per_shape * 0.10)
    n_shell = p_per_shape - n_uniform - n_surf

    # uniform in ball
    u = rng.standard_normal((n_uniform, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u *= (rng.uniform(0, 1, size=(n_uniform, 1)) ** (1 / 3)) * BOUND

    # near-surface: project uniform points onto the surface, then jitter
    s0 = rng.standard_normal((n_surf, 3))
    s0 /= np.linalg.norm(s0, axis=1, keepdims=True)
    s0 *= (rng.uniform(0, 1, size=(n_surf, 1)) ** (1 / 3)) * BOUND
    p = s0
    for _ in range(5):
        d = sdf(p)[:, None]
        g = numeric_grad(sdf, p)
        p = p - d * g / np.maximum(np.sum(g * g, axis=1, keepdims=True), 1e-8)
    p += rng.standard_normal(p.shape) * rng.uniform(0.006, 0.05, size=(n_surf, 1))

    # far shell
    sh = rng.standard_normal((n_shell, 3))
    sh /= np.linalg.norm(sh, axis=1, keepdims=True)
    sh *= rng.uniform(BOUND, BOUND * 1.3, size=(n_shell, 1))

    pts = np.concatenate([u, p, sh]).astype(np.float32)
    vals = sdf(pts).clip(-1.5, 1.5).astype(np.float32)
    return pts, vals


def gen_one(idx: int):
    out = sample_shape(idx)
    tries = 0
    while out is None and tries < 4:  # persistently rejected: reroll harder
        out = sample_shape((idx + 1) * 7919 + tries * 104729)
        tries += 1
    if out is None:  # give up gracefully: plain sphere, always valid
        rng = np.random.default_rng(idx)
        pts = rng.standard_normal((P_PER_SHAPE, 3)).astype(np.float32)
        vals = (np.linalg.norm(pts, axis=1) - 0.7).astype(np.float32)
        out = (pts, vals)
    return idx, out[0], out[1]


# ------------------------------------------------------- numpy preview marcher
def render_sdf(sdf, size=288, cam_dist=3.0, fov_deg=40.0, seed_tilt=0.5):
    """Sphere-trace a callable SDF -> uint8 RGB. Teal-on-void, house palette."""
    rng = np.random.default_rng(4)
    yaw, pitch = rng.uniform(0, 6.28), seed_tilt
    eye = cam_dist * np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = np.tan(np.radians(fov_deg) / 2)

    xs = (np.arange(size) + 0.5) / size * 2 - 1
    gx, gy = np.meshgrid(xs, xs)
    ro = np.tile(eye, (size * size, 1))
    rd = fwd + gx.ravel()[:, None] * f * right + (-gy.ravel())[:, None] * f * up
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)

    t = np.zeros(size * size)
    alive = np.ones(size * size, dtype=bool)
    for _ in range(96):
        if not alive.any():
            break
        p = ro[alive] + rd[alive] * t[alive, None]
        d = sdf(p)
        hit = d < 0.0018
        esc = t[alive] > cam_dist + BOUND * 1.5
        done = hit | esc
        t[np.where(alive)[0][~done]] += d[~done] * 0.95
        alive[np.where(alive)[0][done]] = False

    img = np.full((size * size, 3), np.array([0.016, 0.012, 0.024]))
    # recompute hit mask with a final pass
    p = ro + rd * t[:, None]
    d = sdf(p)
    hit = (d < 0.004) & (t < cam_dist + BOUND * 1.5)
    if hit.any():
        ph = p[hit]
        g = numeric_grad(sdf, ph, eps=1e-3)
        n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        l1 = np.array([0.6, 0.8, 0.5]); l1 /= np.linalg.norm(l1)
        l2 = np.array([-0.7, -0.2, -0.4]); l2 /= np.linalg.norm(l2)
        dif = np.clip(n @ l1, 0, 1) * 0.9 + np.clip(n @ l2, 0, 1) * 0.3
        fres = (1 - np.clip(np.sum(n * -rd[hit], axis=1, keepdims=True), 0, 1)) ** 3
        base = np.array([0.35, 0.55, 0.62])
        col = base * (0.08 + dif[:, None]) + fres * (0.5 * base + 0.3 * np.array([0.9, 0.5, 0.8]))
        img[hit] = 1 - np.exp(-col * 1.8)
    img = (np.clip(img, 0, 1) ** (1 / 1.6) * 255).astype(np.uint8).reshape(size, size, 3)
    return img


def preview_one(idx: int):
    sdf = normalized_shape(idx)
    if sdf is None:
        sdf = lambda p: np.linalg.norm(p, axis=1) - 0.8
    return idx, render_sdf(sdf, cam_dist=3.4, fov_deg=38.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--previews", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1207)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pts_all = np.empty((args.n, P_PER_SHAPE, 3), dtype=np.float16)
    sdf_all = np.empty((args.n, P_PER_SHAPE), dtype=np.float16)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, (idx, pts, vals) in enumerate(ex.map(gen_one, range(args.n), chunksize=8)):
            pts_all[idx] = pts
            sdf_all[idx] = vals
            if (done + 1) % 200 == 0:
                print(f"  {done + 1}/{args.n} forms ({time.time() - t0:.0f}s)", flush=True)

    out = DATA_DIR / "imagination_corpus.npz"
    np.savez_compressed(out, xyz=pts_all, sdf=sdf_all)
    (DATA_DIR / "corpus_meta.json").write_text(json.dumps({
        "n": args.n, "p": P_PER_SHAPE, "seed": args.seed, "bound": BOUND,
    }, indent=2))
    print(f"corpus -> {out} ({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, img in ex.map(preview_one, range(min(args.previews, args.n))):
            path = PREVIEW_DIR / f"form_{idx:03d}.png"
            write_png(path, img)
            print(f"  preview -> {path}", flush=True)


if __name__ == "__main__":
    main()
