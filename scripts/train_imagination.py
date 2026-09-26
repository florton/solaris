# Train the imagination: a DeepSDF-class auto-decoder over the procedural
# corpus. Each corpus form gets a learned latent code (L2 prior); the decoder
# maps (fourier features of xyz, z) -> signed distance. CPU-sized on purpose:
# latent 24, MLP 4x128. Wall-clock capped; checkpoints + probe renders along
# the way. Probe renders go through a 3D *grid* (the same representation the
# WebGL raymarcher will sample), so Gate 2 validates the shipping path, not a
# prettier one.
# Run: .venv/Scripts/python scripts/train_imagination.py [--minutes 60] [--resume]
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from imagination_corpus import write_png

LATENT_DIM = 32
FOURIER_BANDS = 6
WIDTH = 160
DEPTH = 5
BOUND = 1.35
CKPT = Path("data/imagination_ckpt.pt")
FINAL = Path("data/imagination.pt")
STATS = Path("data/latent_stats.json")
PREVIEW_DIR = Path("previews/train")


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, bands=FOURIER_BANDS, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.bands = bands
        in_dim = 3 + 3 * 2 * bands + latent_dim
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyz, z):
        feats = [xyz]
        for k in range(self.bands):
            w = (2.0 ** k) * np.pi
            feats += [torch.sin(w * xyz), torch.cos(w * xyz)]
        return self.net(torch.cat(feats + [z], dim=1)).squeeze(1)


def fourier_eval(decoder, z, pts, chunk=65536):
    """Batched no-grad SDF eval of the decoder at pts (M,3) with latent z."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            q = torch.from_numpy(pts[i : i + chunk].astype(np.float32))
            zz = z.expand(len(q), -1)
            outs.append(decoder(q, zz).numpy())
    return np.concatenate(outs)


def grid_from_decoder(decoder, z, g=64, bound=BOUND):
    xs = np.linspace(-bound, bound, g, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return fourier_eval(decoder, z, pts).reshape(g, g, g)


def render_grid(grid, bound=BOUND, size=224, cam_dist=3.4, fov_deg=38.0, yaw=0.9, pitch=0.35):
    """Sphere-trace a trilinearly-sampled SDF grid — the runtime's exact data path."""
    g = grid.shape[0]

    def sdf(p):
        uvw = (p / (2 * bound) + 0.5) * (g - 1)
        i0 = np.clip(np.floor(uvw).astype(int), 0, g - 2)
        fr = np.clip(uvw - i0, 0, 1)
        c = np.stack([grid[i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + cc] for a in (0, 1) for b in (0, 1) for cc in (0, 1)], axis=1)
        w = np.stack([(1 - fr[:, 0] if a == 0 else fr[:, 0]) * (1 - fr[:, 1] if b == 0 else fr[:, 1]) * (1 - fr[:, 2] if cc == 0 else fr[:, 2])
                      for a in (0, 1) for b in (0, 1) for cc in (0, 1)], axis=1)
        return np.sum(c * w, axis=1)

    eye = cam_dist * np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = np.tan(np.radians(fov_deg) / 2)
    xs = (np.arange(size) + 0.5) / size * 2 - 1
    gx, gy = np.meshgrid(xs, xs)
    ro = np.tile(eye, (size * size, 1)).astype(np.float32)
    rd = fwd + gx.ravel()[:, None] * f * right + (-gy.ravel())[:, None] * f * up
    rd = (rd / np.linalg.norm(rd, axis=1, keepdims=True)).astype(np.float32)

    t = np.zeros(size * size, dtype=np.float32)
    alive = np.ones(size * size, dtype=bool)
    for _ in range(80):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        d = sdf(ro[idx] + rd[idx] * t[idx, None])
        hit = d < 0.004
        esc = t[idx] > cam_dist + bound * 1.6
        done = hit | esc
        t[idx[~done]] += d[~done] * 0.9
        alive[idx[done]] = False

    img = np.full((size * size, 3), np.array([0.016, 0.012, 0.024]))
    p = ro + rd * t[:, None]
    d = sdf(p)
    hit = (d < 0.01) & (t < cam_dist + bound * 1.6)
    if hit.any():
        eps = 1e-3
        ph = p[hit]
        grad = np.stack([sdf(ph + eps * np.eye(3)[k]) - sdf(ph - eps * np.eye(3)[k]) for k in range(3)], axis=1) / (2 * eps)
        n = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
        l1 = np.array([0.6, 0.8, 0.5]); l1 /= np.linalg.norm(l1)
        l2 = np.array([-0.7, -0.2, -0.4]); l2 /= np.linalg.norm(l2)
        dif = np.clip(n @ l1, 0, 1) * 0.9 + np.clip(n @ l2, 0, 1) * 0.3
        fres = (1 - np.clip(np.sum(n * -rd[hit], axis=1, keepdims=True), 0, 1)) ** 3
        base = np.array([0.35, 0.55, 0.62])
        col = base * (0.08 + dif[:, None]) + fres * (0.5 * base + 0.3 * np.array([0.9, 0.5, 0.8]))
        img[hit] = 1 - np.exp(-col * 1.8)
    return (np.clip(img, 0, 1) ** (1 / 1.6) * 255).astype(np.uint8).reshape(size, size, 3)


def strip(decoder, latents, path, g=48):
    """Render a horizontal strip of latents side by side into one PNG."""
    tiles = [render_grid(grid_from_decoder(decoder, z, g=g)) for z in latents]
    write_png(path, np.concatenate(tiles, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=75)
    ap.add_argument("--batch-shapes", type=int, default=48)
    ap.add_argument("--batch-points", type=int, default=448)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1207)
    torch.set_num_threads(12)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load("data/imagination_corpus.npz")
    xyz = data["xyz"].astype(np.float32)  # [N, P, 3]
    sdf = data["sdf"].astype(np.float32)  # [N, P]
    n_shapes, n_pts = sdf.shape
    print(f"corpus: {n_shapes} forms x {n_pts} pts")

    decoder = Decoder()
    latents = nn.Embedding(n_shapes, LATENT_DIM)
    nn.init.normal_(latents.weight, std=0.01)
    opt = torch.optim.Adam([
        {"params": decoder.parameters(), "lr": 1e-3},
        {"params": latents.parameters(), "lr": 2e-3},
    ])

    start_step = 0
    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, weights_only=False)
        decoder.load_state_dict(ck["decoder"])
        latents.load_state_dict(ck["latents"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {CKPT} @ step {start_step}")

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    rng = np.random.default_rng(7)
    perm = rng.permutation(n_shapes)
    step = start_step
    last_ckpt = t0
    last_probe = t0
    probe_shapes = [0, 1, 4, 6, 7, 10]

    # cosine schedule toward a horizon re-estimated from the measured step rate
    lr0 = [g["lr"] for g in opt.param_groups]
    def set_lr(f):
        for g, base in zip(opt.param_groups, lr0):
            g["lr"] = base * (0.05 + 0.95 * (1 + np.cos(np.pi * f)) / 2)

    loss_ema = None
    while time.time() < deadline:
        # estimate horizon from measured rate, refreshed periodically
        if step == start_step:
            set_lr(0.0)
        elif step % 200 == 0:
            rate = (step - start_step) / (time.time() - t0)
            remaining = max(0.0, deadline - time.time())
            horizon = step + rate * remaining
            set_lr(step / max(1.0, horizon))

        bi = np.array([perm[(step * args.batch_shapes + k) % n_shapes] for k in range(args.batch_shapes)])
        pi = rng.integers(0, n_pts, size=(args.batch_shapes, args.batch_points))
        bx = torch.from_numpy(xyz[bi[:, None], pi].reshape(-1, 3))
        by = torch.from_numpy(sdf[bi[:, None], pi].reshape(-1))
        bz = latents(torch.from_numpy(np.repeat(bi, args.batch_points)))

        pred = decoder(bx, bz)
        # unclamped Huber: linear gradient far out keeps far-field magnitudes
        # honest (a clamped loss lets them drift and the marcher overshoots),
        # quadratic near zero for a precise level set; near-surface upweighted
        w = 1.0 + 2.0 * (by.abs() < 0.05).float()
        loss = (w * torch.nn.functional.smooth_l1_loss(pred, by, beta=0.05, reduction="none")).mean()
        prior = 1e-4 * (bz ** 2).sum(dim=1).mean()
        (loss + prior).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        lv = loss.item()
        loss_ema = lv if loss_ema is None else loss_ema * 0.98 + lv * 0.02
        if step % 50 == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / elapsed
            eta = (deadline - time.time()) / 60
            print(f"step {step} · loss {loss_ema:.5f} · {rate:.1f} steps/s · eta {eta:.0f} min", flush=True)

        now = time.time()
        if now - last_ckpt > 300:
            torch.save({"decoder": decoder.state_dict(), "latents": latents.state_dict(),
                        "opt": opt.state_dict(), "step": step}, CKPT)
            last_ckpt = now
            # field sanity: negative fraction + far magnitude of a probe form
            with torch.no_grad():
                gprobe = grid_from_decoder(decoder, latents.weight[0].detach(), g=24)
            print(f"  ckpt @ {step} · field min {gprobe.min():.3f} max {gprobe.max():.3f} neg-frac {np.mean(gprobe < 0):.3f}", flush=True)
        if now - last_probe > 600:
            zs = latents.weight[torch.tensor(probe_shapes)].detach()
            strip(decoder, zs, PREVIEW_DIR / f"probe_step{step}.png")
            # interpolation strip: corpus form 0 -> corpus form 6
            z0, z1 = zs[0], zs[3]
            strip(decoder, [z0 * (1 - a) + z1 * a for a in np.linspace(0, 1, 6)],
                  PREVIEW_DIR / f"interp_step{step}.png")
            last_probe = now
            print(f"  probes -> {PREVIEW_DIR}", flush=True)

    torch.save({"decoder": decoder.state_dict(), "latents": latents.state_dict(),
                "opt": opt.state_dict(), "step": step}, CKPT)
    torch.save({"decoder": decoder.state_dict(), "latents": latents.state_dict(),
                "latent_dim": LATENT_DIM, "bands": FOURIER_BANDS, "width": WIDTH, "depth": DEPTH,
                "bound": BOUND}, FINAL)

    # latent cloud stats -> the bridge's on-manifold calibration
    zs = latents.weight.detach().numpy()
    norms = np.linalg.norm(zs, axis=1)
    stats = {
        "latent_dim": LATENT_DIM,
        "dim_std": zs.std(axis=0).tolist(),
        "norm_median": float(np.median(norms)),
        "norm_p95": float(np.percentile(norms, 95)),
    }
    STATS.write_text(json.dumps(stats, indent=2))

    # held-out-point reconstruction error
    dec = decoder.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(n_shapes, 128)):
            q = torch.from_numpy(xyz[i, :512])
            z = latents.weight[i].expand(512, -1)
            tot += float((dec(q, z) - torch.from_numpy(sdf[i, :512])).abs().mean())
            cnt += 1
    print(f"final: step {step} · recon L1 {tot / cnt:.5f} · latent norm med {stats['norm_median']:.3f} p95 {stats['norm_p95']:.3f}")
    zs_p = latents.weight[torch.tensor(probe_shapes)].detach()
    strip(dec, zs_p, PREVIEW_DIR / "final_probe.png", g=64)
    z0, z1 = zs_p[0], zs_p[3]
    strip(dec, [z0 * (1 - a) + z1 * a for a in np.linspace(0, 1, 6)], PREVIEW_DIR / "final_interp.png", g=64)
    print("done ->", FINAL, STATS)


if __name__ == "__main__":
    main()
