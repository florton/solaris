# Up-orientation pass: every form should stand +y up in the browser (facing is
# left as authored: the user only cares about up). glTF says +Y up and the build
# keeps each GLB's own axes, so most forms are already right; the rest are Z-up
# exports lying on their back or side.
#
# Self-supervised: a small 3D CNN sees a form turned by a random yaw (about y,
# so facing never matters) and then one of 6 tilts, and learns which tilt it
# was. The labels are noisy (a few % of the library is itself lying down), the
# majority convention wins: bases down, heads above bodies, wheels down. Then
# each form is scored with each of the 6 tilts undone (× 4 yaws), and only forms
# the net is confident are lying down get stood up.
# (Tried first: CLIP zero-shot, 45% right up axis; a 24-way net that also had to
# learn facing, which has no convention in the data: 35% after 12 min.)
#
# Non-destructive: shards keep the raw grids. data/mimoid_orient.npz holds, per
# uid, the rotation raw -> canonical (an index into ROTS); load_all() applies it
# when it builds the grid cache, mimoid_encode.py re-encodes rows whose
# rotation changed, mimoid_quality.py rescores them. Running again trains on
# the already-stood-up library (cleaner labels) and composes its corrections.
#   -> data/mimoid_orient_net.pt, data/mimoid_orient.npz,
#      previews/mimoid_orient.png (stood-up forms: as is | turned)
# Run: .venv/Scripts/python scripts/mimoid_orient.py [--minutes 10] [--margin 0.5] [--dry-run]
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ORIENT = ROOT / "data" / "mimoid_orient.npz"
NET = ROOT / "data" / "mimoid_orient_net.pt"
PREVIEW = ROOT / "previews" / "mimoid_orient.png"
RES = 16  # the net sees the 64³ grid average-pooled to 16³ (enough for up; 32³ ran at 1.2 steps/s)

# the 24 proper rotations as signed permutation matrices (new coords = M @ old); ROTS[0] = identity
ROTS = []
for _perm in itertools.permutations(range(3)):
    for _signs in itertools.product((1, -1), repeat=3):
        _M = np.zeros((3, 3), int)
        for _i, (_j, _s) in enumerate(zip(_perm, _signs)):
            _M[_i, _j] = _s
        if round(np.linalg.det(_M)) == 1:
            ROTS.append(_M)
assert len(ROTS) == 24 and (ROTS[0] == np.eye(3)).all()


def ridx(M) -> int:
    return next(k for k, R in enumerate(ROTS) if (R == M).all())


COMPOSE = np.array([[ridx(ROTS[a] @ ROTS[b]) for b in range(24)] for a in range(24)])  # ROTS[a] @ ROTS[b]
INV = np.array([ridx(R.T) for R in ROTS])
_PERM = [[int(np.nonzero(M[i])[0][0]) for i in range(3)] for M in ROTS]
_FLIP = [[i for i in range(3) if M[i, _PERM[k][i]] < 0] for k, M in enumerate(ROTS)]
# yaws about +y (facing only) and the 6 tilts: TILT[k] brings the form's old
# direction DOWN_OF[k] to -y, i.e. TILT[0] = none, then the 90° / 180° tilts
YAW = [ridx(np.array(m)) for m in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                                   [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
TILT = [ridx(np.array(m)) for m in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]],      # as is
                                    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],     # 90° about x
                                    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],     # -90° about x
                                    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],     # 90° about z
                                    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],     # -90° about z
                                    [[1, 0, 0], [0, -1, 0], [0, 0, -1]])]   # 180° about x
assert len({tuple(ROTS[t].T @ [0, 1, 0]) for t in TILT}) == 6  # each tilt raises a different old axis to +y


def rotate(g: np.ndarray, r: int) -> np.ndarray:
    """g [X, Y, Z] (centered grid) turned by ROTS[r]; exact (index permutation + flips)."""
    out = g.transpose(_PERM[r])
    return np.ascontiguousarray(np.flip(out, _FLIP[r]) if _FLIP[r] else out)


def rotate_t(x: torch.Tensor, r: int) -> torch.Tensor:
    """x [B, 1, X, Y, Z] turned by ROTS[r]"""
    out = x.permute(0, 1, *[p + 2 for p in _PERM[r]])
    return out.flip([i + 2 for i in _FLIP[r]]) if _FLIP[r] else out


class UpNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        ch = [1, w, 2 * w, 4 * w, 8 * w][: int(np.log2(RES))]  # RES -> ... -> 2
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                       nn.Conv3d(b, b, 4, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch[-1] * 8, 256), nn.SiLU(), nn.Linear(256, 6))

    def forward(self, x):
        return self.head(self.body(x))


def pooled(grids, dev) -> torch.Tensor:
    """int8 64³ grids -> int8 [n, 1, RES, RES, RES] on the GPU"""
    out = torch.empty(len(grids), 1, RES, RES, RES, dtype=torch.int8, device=dev)
    for s in range(0, len(grids), 256):
        x = torch.from_numpy(np.asarray(grids[s:s + 256])).to(dev).float()
        out[s:s + 256] = F.avg_pool3d(x[:, None], 64 // RES).round().to(torch.int8)
    return out


def tilted(x: torch.Tensor, gen: torch.Generator):
    """each sample: random yaw, then a random tilt k -> (x', k)"""
    n = len(x)
    yaw = torch.randint(0, 4, (n,), generator=gen, device=x.device)
    k = torch.randint(0, 6, (n,), generator=gen, device=x.device)
    out = torch.empty_like(x)
    for y in range(4):
        for t in range(6):
            m = (yaw == y) & (k == t)
            if m.any():
                out[m] = rotate_t(x[m], COMPOSE[TILT[t], YAW[y]])
    return out, k


def as_input(x):
    return x.float() / 127


def train(x: torch.Tensor, minutes: float):
    dev = x.device
    gen = torch.Generator(device=dev).manual_seed(0)
    perm = torch.randperm(len(x), generator=gen, device=dev)
    val, tr = perm[: min(1000, len(x) // 20)], perm[min(1000, len(x) // 20):]
    xv, kv = tilted(x[val], gen)
    net = UpNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    t0, step, B = time.time(), 0, 256
    while (el := time.time() - t0) < minutes * 60:
        for g in opt.param_groups:
            g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * el / (minutes * 60)))
        xb, kb = tilted(x[tr[torch.randint(0, len(tr), (B,), generator=gen, device=dev)]], gen)
        loss = F.cross_entropy(net(as_input(xb)), kb, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 250 == 0:
            net.eval()
            with torch.no_grad():
                acc = torch.cat([net(as_input(xv[s:s + 256])).argmax(1) for s in range(0, len(xv), 256)]).eq(kv).float().mean()
            net.train()
            print(f"  step {step:5d} · {el / 60:4.1f} min · loss {loss.item():.3f} · held-out up axis {acc:.3f}", flush=True)
    return net.eval()


def score(net, x: torch.Tensor) -> np.ndarray:
    """P[n, 6]: how upright each form looks with tilt j undone (class 0 = 'as is'),
    averaged over 4 yaws"""
    P = torch.zeros(len(x), 6, device=x.device)
    undo = [INV[t] for t in TILT]
    with torch.no_grad():
        for s in range(0, len(x), 256):
            xb = x[s:s + 256]
            for j in range(6):
                for y in YAW:
                    P[s:s + 256, j] += F.softmax(net(as_input(rotate_t(xb, COMPOSE[y, undo[j]]))), 1)[:, 0] / 4
    return P.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10, help="training time")
    ap.add_argument("--margin", type=float, default=0.5, help="stand a form up only if P(best) - P(as is) exceeds this")
    ap.add_argument("--reuse", action="store_true", help="skip training, use data/mimoid_orient_net.pt")
    ap.add_argument("--dry-run", action="store_true", help="score + review sheet, don't write mimoid_orient.npz")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mimoid_train_ae import load_all
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    grids, uids = load_all("library")  # already turned by any earlier pass
    x = pooled(grids, dev)
    print(f"{len(uids)} forms pooled to {RES}³ in {time.time() - t0:.0f} s", flush=True)
    if args.reuse:
        net = UpNet().to(dev)
        net.load_state_dict(torch.load(NET, weights_only=True))
        net.eval()
    else:
        net = train(x, args.minutes)
        torch.save(net.state_dict(), NET)
    t0 = time.time()
    P = score(net, x)
    best = P.argmax(1)
    gain = P[np.arange(len(P)), best] - P[:, 0]
    turn = (best != 0) & (gain > args.margin)
    print(f"scored in {time.time() - t0:.0f} s · P(upright as is) p10 {np.percentile(P[:, 0], 10):.2f} "
          f"p50 {np.median(P[:, 0]):.2f} · stand up {turn.sum()} of {len(uids)} ({turn.mean():.1%}, margin {args.margin}); "
          f"just under the margin: {((best != 0) & (gain > args.margin - 0.2) & ~turn).sum()}", flush=True)
    print("  would stand up at margin " + " · ".join(f"{m}: {((best != 0) & (gain > m)).sum()}" for m in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

    fix = np.array([INV[TILT[j]] for j in range(6)])[best]  # undo the tilt
    if not args.dry_run:
        old = dict(np.load(ORIENT)) if ORIENT.exists() else {}
        prev = dict(zip([str(u) for u in old["uids"]], old["rot"].tolist())) if old else {}
        cur = np.array([prev.get(u, 0) for u in uids])
        new = np.where(turn, COMPOSE[fix, cur], cur)
        np.savez(ORIENT, uids=np.array(uids), rot=new.astype(np.int8), p_asis=P[:, 0].astype(np.float32),
                 p_best=P[np.arange(len(P)), best].astype(np.float32))
        print(f"-> {ORIENT.name}: {(new != 0).sum()} forms differ from their raw pose")

    # review: the most confident turns, those just over the margin, and some just under it (as is | turned)
    from imagination_corpus import write_png
    from mimoid_data import render_grids
    from mimoid_train_ae import sheet
    order = np.nonzero(turn)[0][np.argsort(-gain[turn])]
    near = np.nonzero((best != 0) & ~turn)[0]
    near = near[np.argsort(-gain[near])][:16]
    pick = np.concatenate([order[:24], order[-16:] if len(order) > 40 else order[24:], near])
    tiles = [render_grids(np.stack([np.asarray(grids[i]), rotate(np.asarray(grids[i]), fix[i])]), size=128) for i in pick]
    if tiles:
        write_png(PREVIEW, sheet(np.concatenate(tiles), 8))
        print(f"-> {PREVIEW.relative_to(ROOT)}: pairs (as is | stood up), 4 per row: 6 rows most confident, "
              f"4 rows just over the margin, 4 rows just under it (not turned)")


if __name__ == "__main__":
    main()
