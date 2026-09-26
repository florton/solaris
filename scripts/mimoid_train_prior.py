# Stage 2 of the mimoid imagination: the dreaming prior. A small 3D conv
# denoiser that diffuses in the grid AE's live latent channels ([3, 8, 8, 8] of 16),
# conditioned on a MiniLM pooled embedding — the same vector the app computes
# for a thought. Trained on (Cap3D caption -> library form latent) pairs.
#
# Conditioning is one layer's pooled embedding plus a learned layer tag
# (0 = token embeddings ... 6 = last layer), sampled at random per training
# pair, so the app can dream one form per layer of the thought. Caption
# dropout (null condition) enables classifier-free guidance.
#
# Diffusion: continuous cosine schedule, v-prediction; sampling is DDIM
# (deterministic from the initial noise, so seed = hash(text) -> same form).
#
# Needs data/mimoid_grid.pt (mimoid_train_ae.py --arch grid) and
# data/mimoid_captions.npz (mimoid_captions.py).
# Run: .venv/Scripts/python scripts/mimoid_train_prior.py [--hours 3] [--resume]
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagination_corpus import write_png  # noqa: E402
from mimoid_captions import embed  # noqa: E402
from mimoid_data import render_grids  # noqa: E402
from mimoid_train_ae import GridDecoder, sheet, to_int8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AE = ROOT / "data" / "mimoid_grid.pt"
CAPS = ROOT / "data" / "mimoid_captions.npz"
CKPT = ROOT / "data" / "mimoid_prior_ckpt.pt"
FINAL = ROOT / "data" / "mimoid_prior.pt"
PREVIEW_DIR = ROOT / "previews" / "mimoid_prior"
C0, C1, EMB = 96, 160, 256  # widths at 8³ and 4³, conditioning width
N_LAYERS = 7
PROMPTS = ["a stone pillar", "a tall rocky mountain", "a red sports car", "a man's head",
           "an old castle tower", "a horse", "a weathered stone face", "a small house"]
LAYER_PROMPT = "a statue of a woman's face"


def alpha_sigma(t):
    """cosine schedule on t ∈ [0, 1]; clipped so t=1 is (almost) pure noise"""
    t = t.clamp(0.0, 0.999)
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def timestep_embedding(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, cin, cout, drop=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv3d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1)
        self.film = nn.Linear(EMB, 2 * cout)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.film(e)[..., None, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b
        return self.skip(x) + self.c2(self.drop(F.silu(h)))


class Attn(nn.Module):
    """self-attention over the 4³ = 64 cells: lets distant parts agree on one form"""
    def __init__(self, c, heads=4):
        super().__init__()
        self.h = heads
        self.n, self.qkv, self.proj = nn.GroupNorm(8, c), nn.Conv3d(c, 3 * c, 1), nn.Conv3d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, d, hh, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.h, c // self.h, d * hh * w).unbind(1)
        att = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.h), -1)  # [b, h, N, N]
        o = (v @ att.transpose(-1, -2)).reshape(b, c, d, hh, w)
        return x + self.proj(o)


class Prior(nn.Module):
    def __init__(self, lat_c=3):
        super().__init__()
        self.lat_c = lat_c
        self.t_mlp = nn.Sequential(nn.Linear(128, EMB), nn.SiLU(), nn.Linear(EMB, EMB))
        self.c_in = nn.Linear(384, EMB)
        self.layer_tag = nn.Embedding(N_LAYERS, EMB)
        self.null = nn.Parameter(torch.zeros(EMB))  # "no thought" for classifier-free guidance
        self.c_mlp = nn.Sequential(nn.SiLU(), nn.Linear(EMB, EMB))
        self.inp = nn.Conv3d(lat_c, C0, 3, padding=1)
        self.d1 = nn.ModuleList([Res(C0, C0), Res(C0, C0)])
        self.down = nn.Conv3d(C0, C1, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([Res(C1, C1), Res(C1, C1), Res(C1, C1)])
        self.a2 = nn.ModuleList([Attn(C1), Attn(C1), Attn(C1)])
        self.up = nn.Conv3d(C1, C0, 3, padding=1)
        self.u1 = nn.ModuleList([Res(2 * C0, C0), Res(C0, C0)])
        self.out_n, self.out = nn.GroupNorm(8, C0), nn.Conv3d(C0, lat_c, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def cond(self, emb, layer, drop_mask=None):
        """emb [B, 384] unit-norm, layer [B] long; drop_mask [B] bool -> null condition"""
        c = self.c_in(emb) + self.layer_tag(layer)
        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.null.expand_as(c), c)
        return c

    def forward(self, x, t, c):
        e = self.t_mlp(timestep_embedding(t)) + self.c_mlp(c)
        h = self.inp(x)
        for r in self.d1:
            h = r(h, e)
        skip = h
        h = self.down(h)
        for r, a in zip(self.d2, self.a2):
            h = a(r(h, e))
        h = self.up(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = torch.cat([h, skip], 1)
        for r in self.u1:
            h = r(h, e)
        return self.out(F.silu(self.out_n(h)))


@torch.no_grad()
def sample(model, c, steps=40, guidance=3.0, noise=None):
    """DDIM (eta 0), v-prediction, classifier-free guidance. c: [B, EMB] condition."""
    b = c.shape[0]
    x = noise if noise is not None else torch.randn(b, model.lat_c, 8, 8, 8, device=c.device)
    null = model.null.expand_as(c)
    ts = torch.linspace(1, 0, steps + 1, device=c.device)
    for i in range(steps):
        t, tn = ts[i].expand(b), ts[i + 1].expand(b)
        v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, null]))
        vc, vu = v.chunk(2)
        v = vu + guidance * (vc - vu)
        a, s = alpha_sigma(t)
        a, s = a[:, None, None, None, None], s[:, None, None, None, None]
        x0 = a * x - s * v
        eps = s * x + a * v
        an, sn = alpha_sigma(tn)
        x = an[:, None, None, None, None] * x0 + sn[:, None, None, None, None] * eps
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--stop-hours", type=float, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p-null", type=float, default=0.1, help="caption dropout for classifier-free guidance")
    ap.add_argument("--emb-noise", type=float, default=0.1, help="embedding jitter; covers the app's q8 MiniLM (cos ≈ 0.99)")
    ap.add_argument("--live-frac", type=float, default=0.25, help="latent channels with std above this × the max are diffused")
    ap.add_argument("--probe-min", type=float, default=20.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1207)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}", flush=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    ae = torch.load(AE, weights_only=False)
    assert ae.get("arch") == "grid", "stage 2 needs the grid AE (mimoid_train_ae.py --arch grid)"
    caps = np.load(CAPS)
    assert list(caps["uids"]) == list(ae["uids"]), "caption embeddings and AE latents are in different orders"
    lat = ae["latents"].float()  # [N, 16, 8, 8, 8]
    # The grid AE only uses a few of its 16 channels (the rest collapsed to the KL
    # prior; decoding with them held at their mean gives IoU 0.999). Diffuse only the
    # live channels, each normalized to unit std — a global scale left them at std ~2.5,
    # which skews the schedule toward low noise, where shape and text barely matter.
    mean = lat.mean((0, 2, 3, 4))
    std = lat.std((0, 2, 3, 4))
    live = (std > args.live_frac * std.max()).nonzero()[:, 0]
    lat = ((lat[:, live] - mean[live, None, None, None]) / std[live, None, None, None]).to(dev)
    fill = mean[:, None, None, None].to(dev)  # dead channels are decoded at their mean

    def to_ae(z):
        """normalized live-channel latent -> the AE's full [B, 16, 8, 8, 8] latent"""
        full = fill.expand(len(z), -1, *z.shape[2:]).clone()
        full[:, live] = z * std[live, None, None, None].to(dev) + mean[live, None, None, None].to(dev)
        return full
    emb = torch.from_numpy(caps["emb"].astype(np.float32)).to(dev)  # [N, 7, 384]
    n = len(lat)
    val_idx = torch.tensor(ae["val_idx"], device=dev)
    is_val = torch.zeros(n, dtype=torch.bool, device=dev); is_val[val_idx] = True
    train_idx = (~is_val).nonzero()[:, 0]
    print(f"pairs: {n} · {len(train_idx)} train · {len(val_idx)} held out · live channels {live.tolist()} "
          f"(std {', '.join(f'{s:.3f}' for s in std[live].tolist())})", flush=True)

    dec = GridDecoder(ae["lat_c"], ae["channels"]).to(dev).eval()
    dec.load_state_dict(ae["dec"])
    lat_c = len(live)
    model = Prior(lat_c).to(dev)
    ema = Prior(lat_c).to(dev).eval()
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"prior params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    step, elapsed0 = 0, 0.0
    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, elapsed0 = ck["step"], ck["elapsed"]
        print(f"resumed @ step {step} ({elapsed0 / 3600:.2f}h into the schedule)", flush=True)

    # probe prompts, embedded the same way as the captions
    probe_emb = torch.from_numpy(embed(PROMPTS + [LAYER_PROMPT])).to(dev)  # [P+1, 7, 384]
    g = torch.Generator(dev).manual_seed(7)
    probe_noise = torch.randn(3 * len(PROMPTS) + N_LAYERS, lat_c, 8, 8, 8, device=dev, generator=g)
    vg = torch.Generator(dev).manual_seed(11)
    v_t = torch.rand(len(val_idx), device=dev, generator=vg)
    v_eps = torch.randn(len(val_idx), lat_c, 8, 8, 8, device=dev, generator=vg)

    def v_loss(net, idx, t, eps, layer, drop):
        x0 = lat[idx]
        a, s = alpha_sigma(t)
        a, s = a[:, None, None, None, None], s[:, None, None, None, None]
        e = emb[idx, layer]
        return F.mse_loss(net(a * x0 + s * eps, t, net.cond(e, layer, drop)), a * eps - s * x0)

    def probe(tag):
        with torch.no_grad():
            layer6 = torch.full((len(val_idx),), 6, device=dev)
            vl = float(v_loss(ema, val_idx, v_t, v_eps, layer6, None))
            # rows: each prompt × 3 seeds (last layer), then the layer walk for one prompt
            c = ema.cond(probe_emb[:len(PROMPTS), 6].repeat_interleave(3, 0), torch.full((3 * len(PROMPTS),), 6, device=dev))
            walk = ema.cond(probe_emb[-1], torch.arange(N_LAYERS, device=dev))  # layer k's own embedding
            z = sample(ema, torch.cat([c, walk]), noise=probe_noise)
            grids = dec(to_ae(z))
        tiles = render_grids(to_int8(grids))
        write_png(PREVIEW_DIR / f"prompts_{tag}.png", sheet(tiles[: 3 * len(PROMPTS)], 6))
        write_png(PREVIEW_DIR / f"layers_{tag}.png", sheet(tiles[3 * len(PROMPTS):], N_LAYERS))
        return vl

    def save(now):
        """checkpoint + the shippable EMA prior, every time — safe to Ctrl+C the run whenever"""
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "elapsed": now - t0}, CKPT)
        torch.save({"prior": ema.state_dict(), "channels": [C0, C1, EMB], "n_layers": N_LAYERS, "step": step,
                    "live": live.tolist(), "latent_mean": mean, "latent_std": std, "ae": str(AE.name)}, FINAL)

    start = time.time()
    t0 = start - elapsed0
    deadline = t0 + args.hours * 3600
    stop_at = min(deadline, t0 + args.stop_hours * 3600) if args.stop_hours else deadline
    last_log = last_ckpt = start
    last_probe = start - args.probe_min * 60 + 120  # first probe after 2 min
    ema_loss = None
    model.train()
    while time.time() < stop_at:
        frac = (time.time() - t0) / (deadline - t0)
        for gr in opt.param_groups:
            gr["lr"] = args.lr * min(1.0, (step + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        idx = train_idx[torch.randint(0, len(train_idx), (args.batch,), device=dev)]
        layer = torch.randint(0, N_LAYERS, (args.batch,), device=dev)
        t = torch.rand(args.batch, device=dev)
        eps = torch.randn(args.batch, lat_c, 8, 8, 8, device=dev)
        x0 = lat[idx]
        a, s = alpha_sigma(t)
        a, s = a[:, None, None, None, None], s[:, None, None, None, None]
        e = emb[idx, layer]
        e = F.normalize(e + args.emb_noise / math.sqrt(384) * torch.randn_like(e), dim=-1)
        drop = torch.rand(args.batch, device=dev) < args.p_null
        # fp32 on purpose: fp16 3D convs are ~3x slower on the GTX 1650 (no tensor cores)
        pred = model(a * x0 + s * eps, t, model.cond(e, layer, drop))
        loss = F.mse_loss(pred.float(), a * eps - s * x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()

        now = time.time()
        if now - last_log > 60:
            print(f"step {step} · loss {ema_loss:.4f} · {(now - t0) / 3600:.2f}h / {args.hours}h", flush=True)
            last_log = now
        if now - last_probe > args.probe_min * 60:
            print(f"  probe · held-out v-loss {probe(f'step{step}'):.4f}", flush=True)
            last_probe = now
        if now - last_ckpt > 900:
            save(now)
            last_ckpt = now

    save(time.time())
    vl = probe("final")
    print(f"final: step {step} · held-out v-loss {vl:.4f} -> {FINAL}", flush=True)


if __name__ == "__main__":
    main()
