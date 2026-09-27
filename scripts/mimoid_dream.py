# Stage 2b of the mimoid imagination: dreaming by retrieval + SDEdit.
# The prior alone blurs forms into an average (HANDOFF: prior v3), but it is
# good at re-denoising a real form. So a thought dreams in three steps, per layer:
#   1. retrieve: cosine between the thought's pooled embedding and every
#      library caption embedding of the same layer; pick one of the top k,
#      weighted by exp((sim - best) / temp)
#   2. noise the picked form's latent to t0 (live channels, normalized)
#   3. denoise it back with the prior, conditioned on the thought (CFG)
# Everything random comes from one mulberry32 stream seeded by FNV-1a(text),
# both portable to the browser, so a thought always dreams the same forms.
#
# Needs data/mimoid_grid.pt, data/mimoid_prior.pt, data/mimoid_captions.npz.
# 8 DDIM steps from t0 0.4 look the same as 16 (or 4); the browser pays per step.
# Run: .venv/Scripts/python scripts/mimoid_dream.py [--t0 0.4] [--k 8]
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagination_corpus import write_png  # noqa: E402
from mimoid_captions import embed  # noqa: E402
from mimoid_data import render_grids  # noqa: E402
from mimoid_train_ae import GridDecoder, sheet, to_int8  # noqa: E402
from mimoid_train_prior import PROMPTS, Prior, alpha_sigma  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AE = ROOT / "data" / "mimoid_grid.pt"
PRIOR = ROOT / "data" / "mimoid_prior.pt"
CAPS = ROOT / "data" / "mimoid_captions.npz"
PREVIEW_DIR = ROOT / "previews" / "mimoid_dream"
N_LAYERS = 7
THOUGHTS = ["a whale drifting through fog", "my grandmother's kitchen in november",
            "the hum of a server room at 3am", "an argument I keep rehearsing",
            "cathedral light on dust", "a city seen from a night train"]


def fnv1a(text: str) -> int:
    """32-bit FNV-1a over UTF-8 bytes (same as a TextEncoder + Math.imul loop in JS)."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def mulberry32(seed: int, n: int) -> np.ndarray:
    """Bit-compatible with mulberry32 in src/bridge.ts; uniforms in [0, 1)."""
    out = np.empty(n, np.float64)
    a = seed & 0xFFFFFFFF
    for i in range(n):
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = (t ^ (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        out[i] = (t ^ (t >> 14)) / 4294967296.0
    return out


def gaussians(u: np.ndarray) -> np.ndarray:
    """Box-Muller on consecutive uniform pairs -> len(u) normals."""
    r = np.sqrt(-2.0 * np.log(1.0 - u[0::2]))
    a = 2.0 * math.pi * u[1::2]
    return np.stack([r * np.cos(a), r * np.sin(a)], 1).reshape(-1)


class Dreamer:
    def __init__(self, dev="cuda"):
        self.dev = dev
        ae = torch.load(AE, weights_only=False)
        pr = torch.load(PRIOR, weights_only=False)
        caps = np.load(CAPS)
        self.captions = caps["captions"]
        self.dec = GridDecoder(ae["lat_c"], ae["channels"]).to(dev).eval()
        self.dec.load_state_dict(ae["dec"])
        self.live = torch.tensor(pr["live"])
        self.mean, self.std = pr["latent_mean"].to(dev), pr["latent_std"].to(dev)
        self.prior = Prior(len(self.live)).to(dev).eval()
        self.prior.load_state_dict(pr["prior"])
        lm, ls = self.mean[self.live, None, None, None], self.std[self.live, None, None, None]
        self.lat = (ae["latents"].float().to(dev)[:, self.live] - lm) / ls  # [N, 3, 8, 8, 8]
        self.cap_emb = torch.from_numpy(caps["emb"].astype(np.float32)).to(dev)  # [N, 7, 384]

    def retrieve(self, e, layer, u, k=8, temp=0.05):
        """e [384] unit-norm; u uniform in [0, 1) -> (library index, candidates, sims)"""
        s, idx = (self.cap_emb[:, layer] @ e).topk(k)
        w = torch.exp((s - s[0]) / temp)
        pick = int(torch.searchsorted(w.cumsum(0) / w.sum(), torch.tensor([u], device=self.dev, dtype=w.dtype)).clamp(max=k - 1))
        return int(idx[pick]), idx.tolist(), s.tolist()

    @torch.no_grad()
    def sdedit(self, x0, c, t0, noise, steps=8, guidance=3.0):
        b = x0.shape[0]
        a, s = alpha_sigma(torch.tensor(t0, device=self.dev))
        x = a * x0 + s * noise
        null = self.prior.null.expand_as(c)
        ts = torch.linspace(t0, 0, steps + 1, device=self.dev)
        for i in range(steps):
            t, tn = ts[i].expand(b), ts[i + 1].expand(b)
            vc, vu = self.prior(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, null])).chunk(2)
            v = vu + guidance * (vc - vu)
            a, s = (z[:, None, None, None, None] for z in alpha_sigma(t))
            x0p, eps = a * x - s * v, s * x + a * v
            an, sn = (z[:, None, None, None, None] for z in alpha_sigma(tn))
            x = an * x0p + sn * eps
        return x

    @torch.no_grad()
    def dream(self, emb, seed, layers=range(N_LAYERS), t0=0.4, k=8, temp=0.05, guidance=3.0):
        """emb [7, 384] pooled thought embedding -> (normalized live latents [L, 3, 8, 8, 8], picks).
        Stream layout: 7 pick uniforms, then one noise field shared by all layers."""
        n_noise = self.lat[0].numel()
        u = mulberry32(seed, N_LAYERS + n_noise)
        noise = torch.from_numpy(gaussians(u[N_LAYERS:]).astype(np.float32)).to(self.dev).view(1, *self.lat.shape[1:])
        e = torch.as_tensor(emb, dtype=torch.float32, device=self.dev)
        layers = list(layers)
        picks = [self.retrieve(e[l], l, u[l], k, temp) for l in layers]
        x0 = self.lat[[p[0] for p in picks]]
        lt = torch.tensor(layers, device=self.dev)
        c = self.prior.cond(e[lt], lt)
        z = x0 if t0 <= 0 else self.sdedit(x0, c, t0, noise.expand_as(x0), guidance=guidance)
        return z, picks

    @torch.no_grad()
    def grids(self, z):
        """normalized live latents -> int8 SDF grids [B, 64, 64, 64]"""
        full = self.mean[None, :, None, None, None].expand(len(z), -1, *z.shape[2:]).clone()
        full[:, self.live] = z * self.std[self.live, None, None, None] + self.mean[self.live, None, None, None]
        return np.concatenate([to_int8(self.dec(b)) for b in full.split(16)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--tag", default="", help="suffix for the preview file names")
    ap.add_argument("--variants", type=int, default=5, help="seed variants per prompt in the prompt sheet")
    args = ap.parse_args()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    d = Dreamer("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(t0=args.t0, k=args.k, temp=args.temp, guidance=args.guidance)

    # 1. prompts (last layer): the text's own seed first, then seed variants
    prompts = PROMPTS + ["a lighthouse on a cliff", "an old castle tower on a hill"]
    embs = embed(prompts)
    tiles = []
    for p, e in zip(prompts, embs):
        z = torch.cat([d.dream(e, fnv1a(p) + v, layers=[6], **kw)[0] for v in range(args.variants)])
        tiles.append(render_grids(d.grids(z)))
    write_png(PREVIEW_DIR / f"prompts{args.tag}.png", sheet(np.concatenate(tiles), args.variants))
    print(f"prompts.png: rows {prompts}, cols = seed variants (L6)")

    # 2. thoughts: one strip per thought, layers 0..6 as the app would show them
    thoughts = THOUGHTS + ["a horse", "a man's head"]
    tiles = []
    for t, e in zip(thoughts, embed(thoughts)):
        z, picks = d.dream(e, fnv1a(t), **kw)
        tiles.append(render_grids(d.grids(z)))
        print(f"\n'{t}' (seed {fnv1a(t)})")
        for l, (i, cand, sims) in enumerate(picks):
            print(f"  L{l}  {sims[cand.index(i)]:.3f}  {d.captions[i]}")
    write_png(PREVIEW_DIR / f"thoughts{args.tag}.png", sheet(np.concatenate(tiles), N_LAYERS))
    print(f"\nthoughts.png: rows {thoughts}, cols = layers 0..6")


if __name__ == "__main__":
    main()
