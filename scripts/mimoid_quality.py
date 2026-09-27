# Library quality pass: score every form as the browser will see it (the grid
# AE's reconstruction of its latent), so junk can be dropped before export.
# Per form:
#   vol       occupied fraction of the 64³ grid
#   main      share of the occupied voxels in the largest connected part
#             (low = scattered fragments / debris)
#   flat      smallest / largest occupied bbox extent (low = plates, map tiles)
#   shell     share of inside voxels within ~1 voxel of the surface (not
#             used to filter: open scans are thin shells and still read well)
#   base      share of the volume in the bottom tenth of the form's height
#             (high = a diorama: small things standing on a big flat plate)
#   iou       reconstruction IoU against the true grid (low = the AE broke it)
#   scene     commas + " and " in the caption (multi-object scenes)
# Writes data/mimoid_quality.npz; `--sheet` renders the worst of each metric.
# Run: .venv/Scripts/python scripts/mimoid_quality.py [--sheet]
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimoid_dream import Dreamer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mimoid_quality.npz"
TRUTH = ROOT / "data" / "mimoid" / "library_all.npy"
SHELL = 127 / 7 * 1.0  # int8 units of one voxel (TRUNC 0.2 ≈ 7 voxels)

# a form is kept when all of these hold (see keep())
LIMITS = {"main": 0.85, "flat": 0.12, "base": 0.4, "iou": 0.6, "scene": 4, "vol": 0.002}
HIGH_IS_BAD = ("base", "scene")


def keep(q) -> np.ndarray:
    ok = np.ones(len(q["vol"]), bool)
    for m, lim in LIMITS.items():
        ok &= q[m] <= lim if m in HIGH_IS_BAD else q[m] >= lim
    return ok


def base_share(g: np.ndarray) -> float:
    """g [X, Y, Z] int8, +y up -> share of the occupied voxels in the lowest 10% of the height"""
    rows = (g < 0).sum((0, 2))  # per y
    ys = np.nonzero(rows)[0]
    if len(ys) == 0:
        return 0.0
    cut = ys[0] + max(1, int(round(0.1 * (ys[-1] - ys[0] + 1))))
    return rows[:cut].sum() / rows.sum()


def metrics(g: np.ndarray, t: np.ndarray):
    occ = g < 0
    n = occ.sum()
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0, 0.0
    lab, k = ndimage.label(occ)
    main = np.bincount(lab.ravel())[1:].max() / n if k else 0.0
    idx = np.nonzero(occ)
    ext = np.array([i.max() - i.min() + 1 for i in idx], np.float32)
    shell = (g[occ] > -SHELL).mean()
    tocc = t < 0
    iou = (occ & tocc).sum() / max((occ | tocc).sum(), 1)
    return n / occ.size, main, ext.min() / ext.max(), shell, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()
    if not OUT.exists():
        d = Dreamer("cuda" if torch.cuda.is_available() else "cpu")
        truth = np.load(TRUTH, mmap_mode="r")
        N = len(d.lat)
        cols = {k: np.zeros(N, np.float32) for k in ("vol", "main", "flat", "shell", "iou")}
        for i in range(0, N, 64):
            g = d.grids(d.lat[i:i + 64])
            for j, gj in enumerate(g):
                for k, v in zip(cols, metrics(gj, truth[i + j])):
                    cols[k][i + j] = v
            if i % 1280 == 0:
                print(f"  {i}/{N}", flush=True)
        caps = [str(c) for c in d.captions]
        cols["scene"] = np.array([c.count(",") + c.count(" and ") for c in caps], np.float32)
        np.savez(OUT, **cols)
    q = dict(np.load(OUT))
    if "base" not in q:  # added after the first pass: decode-only, no truth needed
        d = Dreamer("cuda" if torch.cuda.is_available() else "cpu")
        q["base"] = np.array([base_share(g) for i in range(0, len(d.lat), 64) for g in d.grids(d.lat[i:i + 64])], np.float32)
        np.savez(OUT, **q)
    k = keep(q)
    print(f"{len(k)} forms, keep {k.sum()} ({k.mean():.1%})")
    for m, lim in LIMITS.items():
        bad = q[m] > lim if m in HIGH_IS_BAD else q[m] < lim
        print(f"  {m:6s} limit {lim}: fails {bad.sum():5d}  (p5 {np.percentile(q[m], 5):.3f}  p50 {np.median(q[m]):.3f}  p95 {np.percentile(q[m], 95):.3f})")

    if args.sheet:
        from imagination_corpus import write_png
        from mimoid_data import render_grids
        from mimoid_train_ae import sheet
        d = Dreamer("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(0)
        rows = []
        # per metric: 8 forms just past the limit (what the threshold removes), then 8 kept at random
        for m, lim in LIMITS.items():
            order = np.argsort(q[m]) if m not in HIGH_IS_BAD else np.argsort(-q[m])
            bad = [i for i in order if (q[m][i] < lim if m not in HIGH_IS_BAD else q[m][i] > lim)]
            if not bad:
                continue
            pick = (bad[-8:] * 8)[:8]
            rows.append(d.grids(d.lat[pick]))
            print(f"{m} (just failing): " + " | ".join(str(d.captions[i])[:30] for i in pick))
        kept = rng.choice(np.nonzero(k)[0], 16, replace=False)
        rows.append(d.grids(d.lat[kept]))
        write_png(ROOT / "previews" / "mimoid_quality.png", sheet(render_grids(np.concatenate(rows)), 8))
        print("-> previews/mimoid_quality.png: one row per metric (forms just past its limit), last 2 rows = kept")


if __name__ == "__main__":
    main()
