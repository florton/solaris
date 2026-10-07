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
#   mood      1 if the caption is a toy / plushie / cartoon (mimoid_data.MOOD)
#   dup       1 if the form repeats the shape of a better form that passes the
#             other limits (latent cos > DUP_COS; Objaverse has some models
#             uploaded many times, e.g. one sports car ×8). Same orientation only.
# Writes data/mimoid_quality.npz; when the library has grown (mimoid_encode.py)
# only the new rows are scored. `--sheet` renders the worst of each metric.
# Run: .venv/Scripts/python scripts/mimoid_quality.py [--sheet]
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimoid_data import FAMILY_JUNK, MOOD  # noqa: E402
from mimoid_dream import CAPS, Dreamer  # noqa: E402
from mimoid_train_ae import cache_rot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mimoid_quality.npz"
TRUTH = ROOT / "data" / "mimoid" / "library_all.npy"
SHELL = 127 / 7 * 1.0  # int8 units of one voxel (TRUNC 0.2 ≈ 7 voxels)

# a form is kept when all of these hold (see keep())
LIMITS = {"main": 0.85, "flat": 0.12, "base": 0.4, "iou": 0.6, "scene": 4, "vol": 0.002, "mood": 0, "dup": 0}
HIGH_IS_BAD = ("base", "scene", "mood", "dup")
DUP_COS = 0.97


def keep(q, skip=()) -> np.ndarray:
    ok = np.ones(len(q["vol"]), bool)
    for m, lim in LIMITS.items():
        if m not in skip:
            ok &= q[m] <= lim if m in HIGH_IS_BAD else q[m] >= lim
    return ok


def duplicates(q) -> np.ndarray:
    """1 for each form that passes the other limits but repeats a better one's
    shape: clusters of latent cos > DUP_COS (single linkage), each keeps its
    best-reconstructed form."""
    from mimoid_dream import AE
    lat = torch.load(AE, weights_only=False)["latents"].float()
    lat = (lat - lat.mean((0, 2, 3, 4), keepdim=True)) / lat.std((0, 2, 3, 4), keepdim=True).clamp(min=1e-3)
    cand = np.nonzero(keep(q, skip=("dup",)))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = lat[torch.from_numpy(cand)].reshape(len(cand), -1).to(dev)
    z = z / z.norm(dim=1, keepdim=True)
    par = np.arange(len(cand))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for s in range(0, len(z), 2048):
        c = z[s:s + 2048] @ z.T
        ii, jj = torch.nonzero(c > DUP_COS, as_tuple=True)
        for a, b in zip((ii + s).tolist(), jj.tolist()):
            if a < b:
                par[find(a)] = find(b)
    best = {}
    for i in range(len(cand)):
        r = find(i)
        if r not in best or q["iou"][cand[i]] > q["iou"][cand[best[r]]]:
            best[r] = i
    dup = np.zeros(len(q["vol"]), np.float32)
    dup[cand] = 1
    dup[cand[list(best.values())]] = 0
    return dup


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
    q = dict(np.load(OUT)) if OUT.exists() else {}
    n_old = len(q.get("vol", []))
    caps = [str(c) for c in np.load(CAPS)["captions"]]
    rot = cache_rot("library")  # pose of each grid (mimoid_orient.py); base share depends on it
    turned = np.nonzero(rot[:n_old] != q.get("rot", np.zeros(n_old, int))[:n_old])[0] if n_old else np.zeros(0, int)
    if n_old < len(caps) or len(turned):  # first pass, the library grew, or forms were turned: score those rows only
        d = Dreamer("cuda" if torch.cuda.is_available() else "cpu")
        truth = np.load(TRUTH, mmap_mode="r")
        N = len(d.lat)
        if len(truth) != N or len(caps) != N or len(rot) != N:
            raise SystemExit(f"out of step: {len(truth)} grids, {N} latents, {len(caps)} captions, {len(rot)} poses")
        for k in ("vol", "main", "flat", "shell", "iou", "base"):
            q[k] = np.concatenate([q[k], np.zeros(N - n_old, np.float32)]) if n_old else np.zeros(N, np.float32)
        rows = np.concatenate([turned, np.arange(n_old, N)])
        print(f"  scoring {len(rows)} rows ({len(turned)} turned, {N - n_old} new)", flush=True)
        for s in range(0, len(rows), 64):
            idx = rows[s:s + 64]
            g = d.grids(d.lat[idx])
            for j, gj in enumerate(g):
                for k, v in zip(("vol", "main", "flat", "shell", "iou", "base"), (*metrics(gj, truth[idx[j]]), base_share(gj))):
                    q[k][idx[j]] = v
            if s % 1280 == 0:
                print(f"  {s}/{len(rows)}", flush=True)
    q["rot"] = rot
    # caption metrics are cheap: always recomputed, so a new filter reaches old rows too
    q["scene"] = np.array([c.count(",") + c.count(" and ") for c in caps], np.float32)
    fam = {}
    for line in open(ROOT / "data" / "mimoid" / "library.jsonl", encoding="utf-8"):
        r = json.loads(line)
        fam[r["uid"]] = r["family"]
    fam_of = np.array([fam[str(u)] for u in np.load(CAPS)["uids"]])
    q["mood"] = np.array([bool(MOOD.search(c)) or bool(f in FAMILY_JUNK and FAMILY_JUNK[f].search(c))
                          for c, f in zip(caps, fam_of)], np.float32)
    q["dup"] = duplicates(q)  # after the others: a cluster's representative must pass them
    np.savez(OUT, **q)
    k = keep(q)
    print(f"{len(k)} forms, keep {k.sum()} ({k.mean():.1%})")
    for m, lim in LIMITS.items():
        bad = q[m] > lim if m in HIGH_IS_BAD else q[m] < lim
        print(f"  {m:6s} limit {lim}: fails {bad.sum():5d}  (p5 {np.percentile(q[m], 5):.3f}  p50 {np.median(q[m]):.3f}  p95 {np.percentile(q[m], 95):.3f})")
    print("  kept per family: " + " · ".join(f"{f} {k[fam_of == f].sum()}/{(fam_of == f).sum()}" for f in dict.fromkeys(fam_of)))

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
