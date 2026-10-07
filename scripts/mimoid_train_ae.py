# Stage 1 of the mimoid imagination: a 3D convolutional VAE over the library's
# 64³ truncated SDF grids. The encoder only exists for training; the decoder
# ships to the browser (global latent -> whole 64³ grid in one pass, the exact
# volume the raymarcher samples). A small KL term keeps the latent space
# smooth and roughly Gaussian so stage 2 (the text-conditioned dreaming prior)
# can diffuse in it and so any two forms blend continuously.
#
# Two latent layouts (--arch):
#   global  one 256-vector per form. Keeps silhouettes, loses detail (faces come
#           back as mannequin heads) — 256 numbers can't hold a 64³ form.
#   grid    an 8³ grid of 16-channel codes (8192 numbers); each cell only has
#           to remember its own region. Stage 2 then diffuses on the grid.
# Run: .venv/Scripts/python scripts/mimoid_train_ae.py [--arch grid] [--hours 8] [--stop-hours 1] [--resume]
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagination_corpus import write_png  # noqa: E402
from mimoid_data import G, OUT_DIR, TRUNC, render_grids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Z_DIM = 256
CHANNELS = [128, 128, 64, 32]  # global decoder widths at 4³, 8³, 16³, 32³ (64³ via depth-to-space)
LAT_C = 16  # grid latent: [LAT_C, 8, 8, 8]
GRID_CHANNELS = [128, 96, 48]  # grid decoder widths at 8³, 16³, 32³


# Nothing runs a convolution at 64³: the encoder folds each 2×2×2 block into 8
# channels at 32³ (space-to-depth) and the decoder unfolds 8 channels back
# (depth-to-space). Full-resolution convs with few channels were 3× slower on
# the laptop GPU for no visible gain at this grid size.
def s2d(x):
    """[B, 64, 64, 64] -> [B, 8, 32, 32, 32]"""
    b, X, Y, Z = x.shape
    return x.reshape(b, X // 2, 2, Y // 2, 2, Z // 2, 2).permute(0, 2, 4, 6, 1, 3, 5).reshape(b, 8, X // 2, Y // 2, Z // 2)


def d2s(x):
    """[B, 8, 32, 32, 32] -> [B, 64, 64, 64]; exact inverse of s2d."""
    b, _, X, Y, Z = x.shape
    return x.reshape(b, 2, 2, 2, X, Y, Z).permute(0, 4, 1, 5, 2, 6, 3).reshape(b, 2 * X, 2 * Y, 2 * Z)


def up2(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Decoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, ch=CHANNELS):
        super().__init__()
        self.ch0 = ch[0]
        self.fc = nn.Linear(z_dim, ch[0] * 4 ** 3)
        self.stages = nn.ModuleList()
        for cin, cout in zip(ch[:-1], ch[1:]):  # 4 -> 8 -> 16 -> 32
            self.stages.append(nn.ModuleList([nn.Conv3d(cin, cout, 3, padding=1), nn.Conv3d(cout, cout, 3, padding=1)]))
        self.out = nn.Conv3d(ch[-1], 8, 3, padding=1)

    def forward(self, z):
        x = F.silu(self.fc(z)).reshape(-1, self.ch0, 4, 4, 4)
        for a, b in self.stages:
            x = F.silu(a(up2(x)))
            x = x + F.silu(b(x))
        return d2s(self.out(x))  # [B, 64, 64, 64], TSDF units (1 == TRUNC)


class GridDecoder(nn.Module):
    def __init__(self, c=LAT_C, ch=GRID_CHANNELS):
        super().__init__()
        self.inp = nn.Conv3d(c, ch[0], 3, padding=1)
        self.res0 = nn.ModuleList([nn.Conv3d(ch[0], ch[0], 3, padding=1) for _ in range(2)])
        self.stages = nn.ModuleList()
        for cin, cout in zip(ch[:-1], ch[1:]):  # 8 -> 16 -> 32
            self.stages.append(nn.ModuleList([nn.Conv3d(cin, cout, 3, padding=1), nn.Conv3d(cout, cout, 3, padding=1)]))
        self.out = nn.Conv3d(ch[-1], 8, 3, padding=1)

    def forward(self, z):
        x = F.silu(self.inp(z))
        for r in self.res0:
            x = x + F.silu(r(x))
        for a, b in self.stages:
            x = F.silu(a(up2(x)))
            x = x + F.silu(b(x))
        return d2s(self.out(x))


class GridEncoder(nn.Module):
    def __init__(self, c=LAT_C):
        super().__init__()
        ch = [32, 64, 128]
        layers = [nn.Conv3d(8, ch[0], 3, padding=1), nn.GroupNorm(8, ch[0]), nn.SiLU()]
        for cin, cout in zip(ch[:-1], ch[1:]):  # 32 -> 16 -> 8
            layers += [nn.Conv3d(cin, cout, 4, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                       nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU()]
        layers += [nn.Conv3d(ch[-1], ch[-1], 3, padding=1), nn.GroupNorm(8, ch[-1]), nn.SiLU(),
                   nn.Conv3d(ch[-1], 2 * c, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mu, logvar = self.net(s2d(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 6)


class Encoder(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        ch = [32, 64, 128, 128]
        layers = [nn.Conv3d(8, ch[0], 3, padding=1), nn.GroupNorm(8, ch[0]), nn.SiLU()]
        for cin, cout in zip(ch[:-1], ch[1:]):  # 32 -> 16 -> 8 -> 4
            layers += [nn.Conv3d(cin, cout, 4, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                       nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(ch[-1] * 4 ** 3, 2 * z_dim)

    def forward(self, x):
        h = self.net(s2d(x)).flatten(1)
        mu, logvar = self.fc(h).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 6)


def augment(x):
    """Y stays up; random quarter-turn about Y and a mirror — exact on the grid."""
    k = int(torch.randint(0, 4, ()))
    x = torch.rot90(x, k, dims=(1, 3))  # axes: [B, X, Y, Z]; rotate in the X-Z plane
    if torch.rand(()) < 0.5:
        x = x.flip(1)
    return x


def load_all(tag):
    """All shards as one int8 array, cached as a .npy for fast restarts, each
    form turned to its canonical pose (data/mimoid_orient.npz, mimoid_orient.py)."""
    cache = OUT_DIR / f"{tag}_all.npy"
    uid_cache = OUT_DIR / f"{tag}_all_uids.json"
    n_shards = len(list(OUT_DIR.glob(f"{tag}_[0-9]*.npz")))
    if cache.exists() and uid_cache.exists():
        meta = json.loads(uid_cache.read_text())
        if meta["shards"] == n_shards:
            orient_cache(cache, uid_cache, meta)
            return np.load(cache, mmap_mode="r"), meta["uids"]
    # stream shard by shard into the memmap: concatenating in RAM needs 2× the library (13 GB at 25k forms)
    shards = sorted(OUT_DIR.glob(f"{tag}_[0-9]*.npz"))
    counts = [len(np.load(p)["uids"]) for p in shards]
    tmp = cache.with_name(cache.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int8, shape=(sum(counts), G, G, G))
    uids, i = [], 0
    for p, n in zip(shards, counts):
        z = np.load(p)
        out[i : i + n] = z["grids"]
        uids += [str(u) for u in z["uids"]]
        i += n
    out.flush()
    del out
    os.replace(tmp, cache)
    meta = {"shards": n_shards, "uids": uids, "rot": [0] * len(uids)}
    uid_cache.write_text(json.dumps(meta))
    orient_cache(cache, uid_cache, meta)
    return np.load(cache, mmap_mode="r"), uids


def orient_cache(cache, uid_cache, meta):
    """Turn the cached grids (in place) to the poses mimoid_orient.npz asks for;
    meta["rot"] records what each row currently has (raw = 0)."""
    from mimoid_orient import COMPOSE, INV, ORIENT, rotate
    n = len(meta["uids"])
    have = meta.setdefault("rot", [0] * n)
    want = [0] * n
    if ORIENT.exists():
        o = np.load(ORIENT)
        by_uid = dict(zip([str(u) for u in o["uids"]], o["rot"].tolist()))
        want = [by_uid.get(u, 0) for u in meta["uids"]]
    todo = [i for i in range(n) if want[i] != have[i]]
    if not todo:
        return
    print(f"  turning {len(todo)} cached grids to their canonical pose", flush=True)
    arr = np.load(cache, mmap_mode="r+")
    for k, i in enumerate(todo):
        arr[i] = rotate(np.asarray(arr[i]), COMPOSE[want[i], INV[have[i]]])
        have[i] = want[i]
        if (k + 1) % 2000 == 0 or k + 1 == len(todo):  # progress survives an interruption
            arr.flush()
            uid_cache.write_text(json.dumps(meta))
    del arr


def cache_rot(tag) -> np.ndarray:
    """the pose index each cached row currently has (after load_all)"""
    meta = json.loads((OUT_DIR / f"{tag}_all_uids.json").read_text())
    return np.array(meta.get("rot", [0] * len(meta["uids"])))


def sheet(tiles, cols):
    pad = (-len(tiles)) % cols
    tiles = np.concatenate([tiles, np.zeros((pad,) + tiles.shape[1:], np.uint8)])
    return np.ascontiguousarray(tiles.reshape(-1, cols, *tiles.shape[1:]).transpose(0, 2, 1, 3, 4).reshape(-1, cols * tiles.shape[2], 3))


def to_int8(t):
    return (t.float().clamp(-1, 1) * 127).round().to(torch.int8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="library")
    ap.add_argument("--arch", choices=["global", "grid"], default="global")
    ap.add_argument("--name", default=None, help="output name (default mimoid_ae / mimoid_grid)")
    ap.add_argument("--hours", type=float, default=8.0, help="length of the LR schedule")
    ap.add_argument("--stop-hours", type=float, default=None, help="stop early, e.g. a 1 h A/B test on the full schedule")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--beta", type=float, default=None, help="KL weight on the per-sample KL sum (default 2e-5 global, 5e-6 grid)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=float, default=60.0)
    # fp16 3D convs run ~2x SLOWER than fp32 on the GTX 1650 (no tensor cores): off by default
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (only on GPUs with tensor cores)")
    args = ap.parse_args()
    grid = args.arch == "grid"
    if args.beta is None:
        args.beta = 5e-6 if grid else 2e-5
    name = args.name or ("mimoid_grid" if grid else "mimoid_ae")
    ckpt_path = ROOT / "data" / f"{name}_ckpt.pt"
    final_path = ROOT / "data" / f"{name}.pt"
    preview_dir = ROOT / "previews" / ("mimoid_train" if name == "mimoid_ae" else name)

    torch.manual_seed(1207)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and dev == "cuda"
    print(f"device: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'} · {'fp16 autocast' if amp else 'fp32'}", flush=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    grids, uids = load_all(args.tag)
    n = len(uids)
    rng = np.random.default_rng(1207)
    perm = rng.permutation(n)
    n_val = max(8, n // 50)
    val_idx, train_idx = np.sort(perm[:n_val]), perm[n_val:]
    print(f"library: {n} forms · {len(train_idx)} train · {n_val} held out", flush=True)

    enc, dec = (GridEncoder().to(dev), GridDecoder().to(dev)) if grid else (Encoder().to(dev), Decoder().to(dev))
    n_dec = sum(p.numel() for p in dec.parameters())
    print(f"decoder params: {n_dec / 1e6:.2f}M · encoder {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=amp)
    step, epoch, elapsed0 = 0, 0, 0.0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); opt.load_state_dict(ck["opt"])
        step, epoch, elapsed0 = ck["step"], ck["epoch"], ck.get("elapsed", 0.0)
        print(f"resumed @ step {step} epoch {epoch} ({elapsed0 / 3600:.2f}h into the schedule)", flush=True)

    val = torch.from_numpy(np.asarray(grids[val_idx[:12]]).astype(np.float32) / 127).to(dev)
    write_png(preview_dir / "val_truth.png", sheet(render_grids(np.asarray(grids[val_idx[:12]])), 6))

    start = time.time()
    t0 = start - elapsed0  # the LR schedule's clock survives --resume
    deadline = t0 + args.hours * 3600
    stop_at = min(deadline, t0 + args.stop_hours * 3600) if args.stop_hours else deadline
    last_log = last_probe = last_ckpt = start
    step0 = step
    ema = {}

    def probe(tag):
        enc.eval(); dec.eval()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp):
            mu, _ = enc(val)
            rec = dec(mu)
            # interpolation between two held-out forms: the blends must stay forms
            a, b = mu[0], mu[3]
            zs = torch.stack([a * (1 - t) + b * t for t in np.linspace(0, 1, 6)])
            interp = dec(zs)
        err = float((rec.float().clamp(-1, 1) - val).abs().mean())
        inside, rin = val < 0, rec.float() < 0
        iou = float(((inside & rin).sum((1, 2, 3)).float() / (inside | rin).sum((1, 2, 3)).clamp(min=1).float()).mean())
        spread = float(mu.float().flatten(1).std(0).mean())
        write_png(preview_dir / f"val_recon_{tag}.png", sheet(render_grids(to_int8(rec)), 6))
        write_png(preview_dir / f"interp_{tag}.png", sheet(render_grids(to_int8(interp)), 6))
        enc.train(); dec.train()
        return err, iou, spread

    running = True
    while running:
        order = rng.permutation(train_idx)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if time.time() > stop_at:
                running = False
                break
            # cosine decay toward the wall-clock deadline
            frac = (time.time() - t0) / (deadline - t0)
            for g in opt.param_groups:
                g["lr"] = args.lr * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(1.0, frac))))
            idx = np.sort(order[i : i + args.batch])
            x = torch.from_numpy(np.asarray(grids[idx]).astype(np.float32) / 127).to(dev, non_blocking=True)
            x = augment(x)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                mu, logvar = enc(x)
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                y = dec(z).float()
            # L1 in TSDF units; the band near the surface is what the eye sees
            w = 1.0 + 4.0 * (x.abs() < 0.25).float()
            rec = (w * (y - x).abs()).mean()
            kl = 0.5 * (mu.float() ** 2 + logvar.float().exp() - 1 - logvar.float()).flatten(1).sum(1).mean()
            sig = float((0.5 * logvar.float()).exp().mean())
            loss = rec + args.beta * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            for k, v in (("rec", rec.item()), ("kl", kl.item()), ("sig", sig)):
                ema[k] = v if k not in ema else ema[k] * 0.98 + v * 0.02

            now = time.time()
            if now - last_log > args.log_every:
                print(f"step {step} · epoch {epoch} · rec {ema['rec']:.4f} · kl {ema['kl']:.1f} · sigma {ema['sig']:.3f} · "
                      f"{(step - step0) * args.batch / (now - start):.1f} forms/s · {(now - t0) / 3600:.2f}h / {args.hours}h", flush=True)
                last_log = now
            if now - last_probe > 1800:
                err, iou, spread = probe(f"step{step}")
                print(f"  probe · held-out L1 {err:.4f} (TSDF units; x{TRUNC} = world) · IoU {iou:.3f} · "
                      f"latent spread {spread:.3f} vs sigma {ema['sig']:.3f}", flush=True)
                last_probe = now
            if now - last_ckpt > 900:
                torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                            "step": step, "epoch": epoch, "elapsed": now - t0, "arch": args.arch}, ckpt_path)
                last_ckpt = now
        else:
            epoch += 1

    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                "step": step, "epoch": epoch, "elapsed": time.time() - t0, "arch": args.arch}, ckpt_path)
    err, iou, _ = probe("final")
    # latent codes (posterior means) for every form -> stage 2 trains on these
    enc.eval()
    mus = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp):
        for i in range(0, n, 32):
            x = torch.from_numpy(np.asarray(grids[i : i + 32]).astype(np.float32) / 127).to(dev)
            mus.append(enc(x)[0].float().cpu())
    mus = torch.cat(mus)
    arch = {"arch": "grid", "lat_c": LAT_C, "channels": GRID_CHANNELS} if grid else {"arch": "global", "z_dim": Z_DIM, "channels": CHANNELS}
    torch.save({"dec": dec.state_dict(), **arch, "trunc": TRUNC, "grid": G,
                "latents": mus.half() if grid else mus, "uids": uids, "val_idx": val_idx.tolist(), "params": n_dec}, final_path)
    print(f"final: step {step} · epoch {epoch} · held-out L1 {err:.4f} · IoU {iou:.3f} · latent std {mus.std(0).mean():.3f} -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
