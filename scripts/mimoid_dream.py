# Stage 2b of the mimoid imagination: dreaming = retrieval + a 4D walk.
# The reference the browser (src/dream.ts, src/dream-worker.ts) must match:
#   1. split the thought into phrases at connecting words ("tea | grandmas
#      kitchen | a rainstorm"); a phrase whose best match is under minSim is
#      dropped. One phrase (or none left) -> the whole thought is the phrase.
#      Without the split the dominant noun owns every layer: the whole
#      sentence's top 24 were all teapots.
#   2. retrieve per phrase: cosine between its sentence embedding (pooled
#      layer 6: what all-MiniLM-L6-v2 was trained to output) and every kept
#      library caption (scripts/mimoid_quality.py drops fragments, AE-broken
#      forms, multi-object scenes and dioramas); the top `pool` are its
#      neighbourhood, same-caption repeats (cos > dup) counted once
#   3. walk the layers deepest first: the first phrase (the subject) gets the
#      deep layers, each later phrase the next ones up (`allot`), so scrubbing
#      w reads as a scene. Each layer draws a different form from its phrase's
#      neighbourhood, weighted by exp((sim - best) / temp); temp widens toward
#      the shallow layers (deep = closest match, shallow = looser association)
#   4. decode, extend each truncated SDF to a full distance field (EDT), and
#      between neighbouring layers melt: lerp the full fields along w and
#      inflate the middle so dissimilar forms pass through one fused mass
#      instead of tearing (plain truncated-SDF lerps vanish half-way)
# One mulberry32 stream seeded by FNV-1a(text): same thought, same body.
#
# Retired (kept for diagnostics): SDEdit with the prior after retrieval. It
# shreds thin forms (helicopter rotors, sword blades) and cost the browser
# 33 MB + ~10 s per thought; `Dreamer.sdedit`/`dream` still run it.
#
# Needs data/mimoid_grid.pt, data/mimoid_captions.npz, data/mimoid_quality.npz
# (and data/mimoid_prior.pt for the SDEdit path).
# Run: .venv/Scripts/python scripts/mimoid_dream.py [--tag _x]
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagination_corpus import write_png  # noqa: E402
from mimoid_captions import embed  # noqa: E402
from mimoid_data import G, TRUNC, render_grids  # noqa: E402
from mimoid_train_ae import GridDecoder, sheet, to_int8  # noqa: E402
from mimoid_train_prior import PROMPTS, Prior, alpha_sigma  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AE = ROOT / "data" / "mimoid_grid.pt"
PRIOR = ROOT / "data" / "mimoid_prior.pt"
CAPS = ROOT / "data" / "mimoid_captions.npz"
PREVIEW_DIR = ROOT / "previews" / "mimoid_dream"
N_LAYERS = 7
# the browser's dreaming parameters (exported into dream_model.json)
WALK = {"pool": 24, "tempDeep": 0.02, "tempShallow": 0.08, "meltMin": 0.06, "meltMax": 0.25, "meltKeep": 1.1,
        "minSim": 0.35, "dup": 0.99, "maxPhrases": 4,
        # a phrase ends at these words (dropped); a phrase of only `stop` words is dropped
        "connect": ["in", "inside", "into", "during", "through", "throughout", "on", "onto", "at", "with", "and",
                    "under", "beneath", "below", "over", "above", "across", "beside", "near", "by", "from", "after",
                    "before", "while", "among", "amid", "between", "behind", "around", "toward", "towards",
                    "against", "along"],
        "stop": ["a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "this", "that", "some", "of",
                 "to"]}
EMB_LAYER = 6
THOUGHTS = ["a whale drifting through fog", "my grandmother's kitchen in november",
            "the hum of a server room at 3am", "an argument I keep rehearsing",
            "cathedral light on dust", "a city seen from a night train"]
SCENES = ["tea in grandmas kitchen during a rainstorm", "a knight and a dragon on a mountain",
          "a fox sleeping under the snow", "the bust of a woman in marble", "a lighthouse in a storm",
          "an elephant walking through a dream"]


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


def phrases(text: str) -> list[str]:
    """split at connecting words; phrases of only stop words are dropped. Mirrors phrases in dream.ts."""
    connect, stop = set(WALK["connect"]), set(WALK["stop"])
    out, cur = [], []
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        if w in connect:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        out.append(cur)
    return [" ".join(p) for p in out if any(w not in stop for w in p)]


def allot(n_phrases: int, n_layers: int = N_LAYERS) -> list[int]:
    """layers per phrase, first (deepest) phrase first: 7 -> [7], [5, 2], [3, 2, 2], [2, 2, 2, 1]"""
    if n_phrases <= 1:
        return [n_layers]
    first = max(2, n_layers - 2 * (n_phrases - 1))
    rest, m = n_layers - first, n_phrases - 1
    return [first] + [rest // m + (1 if i < rest % m else 0) for i in range(m)]


def neighbourhood(sims: np.ndarray, emb: np.ndarray, pool=WALK["pool"], dup=WALK["dup"]):
    """sims [N] (excluded at -inf) -> (indices, sims) of the top `pool`, skipping same-caption repeats"""
    idx = []
    for i in np.argsort(-sims, kind="stable"):
        if len(idx) == pool or not np.isfinite(sims[i]):
            break
        if idx and (emb[idx] @ emb[i]).max() > dup:
            continue
        idx.append(int(i))
    return np.array(idx), sims[idx]


def pick_forms(sims_per_phrase: list[np.ndarray], emb: np.ndarray, seed: int, n_layers: int = N_LAYERS,
               temp_deep=WALK["tempDeep"], temp_shallow=WALK["tempShallow"]):
    """one sims [N] per phrase (excluded forms at -inf), emb [N, 384] unit-norm -> [(index, sim, phrase)]
    per layer, all distinct. Mirrors pickForms."""
    u = mulberry32(seed, n_layers)
    picks, taken, l = [None] * n_layers, set(), n_layers - 1
    for p, (sims, count) in enumerate(zip(sims_per_phrase, allot(len(sims_per_phrase), n_layers))):
        sims = sims.copy()
        sims[list(taken)] = -np.inf
        top, ts = neighbourhood(sims, emb)
        left = list(range(len(top)))
        for _ in range(count):
            temp = temp_shallow + (temp_deep - temp_shallow) * (l / (n_layers - 1))
            w = np.exp((ts[left] - ts[0]) / temp)
            k = min(int(np.searchsorted(np.cumsum(w / w.sum()), u[n_layers - 1 - l])), len(left) - 1)
            j = left.pop(k) if len(left) > 1 else left[0]
            picks[l] = (int(top[j]), float(ts[j]), p)
            taken.add(int(top[j]))
            l -= 1
    return picks


def full_sdf(g: np.ndarray, h: float, trunc: float = TRUNC) -> np.ndarray:
    """truncated SDF in world units -> full signed distance (band kept, EDT beyond). Mirrors fullSdf."""
    from scipy.ndimage import distance_transform_edt
    band = trunc * 0.98
    inb = np.abs(g) < band
    dist = np.minimum(distance_transform_edt(~inb), 2 * g.shape[0])
    return np.where(inb, g, np.sign(np.where(g == 0, 1, g)) * (band + (dist - 1) * h)).astype(np.float32)


def melt_amount(a: np.ndarray, b: np.ndarray) -> float:
    """inflation at the a -> b midpoint keeping meltKeep of the smaller volume. Mirrors meltAmount."""
    target = int(WALK["meltKeep"] * min((a < 0).sum(), (b < 0).sum()))
    if target <= 0:
        return WALK["meltMin"]
    mid = np.sort((0.5 * (a + b)).ravel())
    return float(np.clip(mid[target], WALK["meltMin"], WALK["meltMax"]))


def body_at(fields, melts, t: float) -> np.ndarray:
    """the 4D body's slice at layer coordinate t in [0, L-1] (the shader's formField, untilted)"""
    l0 = min(int(np.floor(t)), len(fields) - 2)
    fr = t - l0
    s = np.clip((fr - 0.12) / 0.76, 0, 1)
    s = s * s * (3 - 2 * s)
    return (1 - s) * fields[l0] + s * fields[l0 + 1] - melts[l0] * np.sin(np.pi * s)


class Walker:
    """The browser's dreamer: kept library, sentence-embedding retrieval, full SDF fields."""

    def __init__(self, dev="cuda"):
        from mimoid_quality import OUT as QUALITY, keep
        self.d = Dreamer(dev)
        self.keep = keep(dict(np.load(QUALITY)))
        self.h = 2.0 / (G - 1)
        self.emb = self.d.cap_emb[:, EMB_LAYER].cpu().numpy()

    def sims(self, texts: list[str]) -> np.ndarray:
        """[len(texts), N] cosine to every library caption, excluded forms at -inf"""
        q = torch.as_tensor(embed(texts)[:, EMB_LAYER], device=self.d.dev)
        s = (q @ self.d.cap_emb[:, EMB_LAYER].T).cpu().numpy()
        s[:, ~self.keep] = -np.inf
        return s

    def segments(self, text: str, split=True) -> tuple[list[str], np.ndarray]:
        """the phrases a thought is dreamt as, and their sims"""
        ph = phrases(text) if split else []
        if len(ph) > 1:
            s = self.sims(ph)
            ok = s.max(1) >= WALK["minSim"]
            ph, s = [p for p, o in zip(ph, ok) if o][: WALK["maxPhrases"]], s[ok][: WALK["maxPhrases"]]
        if len(ph) <= 1:
            ph, s = [text], self.sims([text])
        return ph, s

    def walk(self, text: str, split=True):
        ph, s = self.segments(text, split)
        picks = [(i, sim, ph[p]) for i, sim, p in pick_forms(list(s), self.emb, fnv1a(text))]
        g = self.d.grids(self.d.lat[[i for i, _, _ in picks]]).astype(np.float32) / 127 * TRUNC
        fields = [full_sdf(x, self.h) for x in g]
        melts = [melt_amount(fields[l], fields[l + 1]) for l in range(len(fields) - 1)]
        return picks, fields, melts


def to_grid(f: np.ndarray) -> np.ndarray:
    return (np.clip(f / TRUNC, -1, 1) * 127).round().astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for the preview file name")
    ap.add_argument("--samples", type=int, default=13, help="slices along w per thought")
    ap.add_argument("--flat", action="store_true", help="no phrase split (the v0.02 walk), for comparison")
    args = ap.parse_args()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    w = Walker("cuda" if torch.cuda.is_available() else "cpu")
    thoughts = SCENES + THOUGHTS + ["a horse", "a knife and an arrow", "ants on a log", "a helicopter", "a man's head"]
    tiles = []
    for t in thoughts:
        picks, fields, melts = w.walk(t, split=not args.flat)
        print(f"\n'{t}'  melts " + " ".join(f"{m:.2f}" for m in melts))
        for l, (i, sim, p) in enumerate(picks):
            print(f"  L{l}  {sim:.3f}  [{p[:24]}]  {w.d.captions[i]}")
        ts = np.linspace(0, N_LAYERS - 1, args.samples)
        tiles.append(render_grids(np.stack([to_grid(body_at(fields, melts, x)) for x in ts])))
    write_png(PREVIEW_DIR / f"walk{args.tag}.png", sheet(np.concatenate(tiles), args.samples))
    print(f"\nwalk{args.tag}.png: one row per thought, {args.samples} slices along w (layer 0 -> 6)")


if __name__ == "__main__":
    main()
