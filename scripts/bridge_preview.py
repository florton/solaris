# Bridge preview (Gate 3): real thoughts -> real latents -> rendered forms,
# before anything ships to the browser. Runs MiniLM on sample thoughts, pools
# per-layer embeddings exactly as src/model.ts does (specials stripped, plain
# mean, L2-normalized), applies the exact bridge math of src/bridge.ts
# (int32-compatible mulberry32 mirror), decodes grids and renders strips.
# Also calibrates `spread` so bridged latents sit on the trained latent
# manifold, and writes data/bridge_calib.json for export_imagination.py.
# Run: .venv/Scripts/python scripts/bridge_preview.py
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from imagination_corpus import write_png
from train_imagination import Decoder, strip

BRIDGE_SEED = 1207
MID_DIM = 64
GAIN1 = float(np.sqrt(3.0))
GAIN2 = float(np.sqrt(3.0 / MID_DIM))
LAYER_ALPHA = 0.65
BOUND = 1.35

THOUGHTS = [
    "a whale drifting through fog",
    "a whale drifting through thick fog",  # near-duplicate: should morph, not jump
    "my grandmother's kitchen in november",
    "the hum of a server room at 3am",
    "an argument I keep rehearsing",
    "cathedral light on dust",
    "the taste of copper and adrenaline",
    "a city seen from a night train",
]


def mulberry32(seed: int, n: int) -> np.ndarray:
    """Bit-compatible mirror of the JS mulberry32 (uint32 wrap arithmetic)."""
    out = np.empty(n, dtype=np.float64)
    a = np.uint32(seed)
    for i in range(n):
        a = np.uint32(a + np.uint32(0x6D2B79F5))
        t = a
        t = np.uint32((t ^ (t >> np.uint32(15))) * np.uint32(t | np.uint32(1)))
        t = np.uint32(t ^ (t + np.uint32((t ^ (t >> np.uint32(7))) * np.uint32(t | np.uint32(61)))))
        t = np.uint32(t ^ (t >> np.uint32(14)))
        out[i] = int(t) / 4294967296.0
    return out


class NumpyBridge:
    def __init__(self, latent_dim: int, hidden_dim: int, dim_std: np.ndarray, spread: float):
        # one stream, same fill order as src/bridge.ts (w1 then w2, row-major)
        stream = mulberry32(BRIDGE_SEED, hidden_dim * MID_DIM + MID_DIM * latent_dim)
        self.w1 = (stream[: hidden_dim * MID_DIM] * 2 - 1).astype(np.float32).reshape(hidden_dim, MID_DIM)
        self.w2 = (stream[hidden_dim * MID_DIM :] * 2 - 1).astype(np.float32).reshape(MID_DIM, latent_dim)
        self.dim_std = dim_std
        self.spread = spread

    def map(self, e: np.ndarray) -> np.ndarray:
        h = np.tanh((e @ self.w1) * GAIN1)
        raw = (h @ self.w2) * GAIN2
        return np.tanh(raw) * self.dim_std * self.spread

    def map_layers(self, pooled: np.ndarray) -> np.ndarray:
        base = self.map(pooled.mean(axis=0))
        return base + LAYER_ALPHA * (np.stack([self.map(e) for e in pooled]) - base)


def pooled_layers(model, tokenizer, text: str) -> np.ndarray:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    layers = []
    for h in out.hidden_states:
        rows = h[0, 1:-1].numpy()  # strip [CLS] / [SEP], matching the app's keep-list
        e = rows.mean(axis=0)
        layers.append(e / (np.linalg.norm(e) + 1e-12))
    return np.stack(layers)


def main():
    ck = torch.load("data/imagination.pt", weights_only=False)
    decoder = Decoder(ck["latent_dim"], ck["bands"], ck["width"], ck["depth"])
    decoder.load_state_dict(ck["decoder"])
    decoder.eval()
    stats = json.loads(Path("data/latent_stats.json").read_text())
    dim_std = np.array(stats["dim_std"], dtype=np.float32)
    print(f"latent cloud: norm med {stats['norm_median']:.3f} p95 {stats['norm_p95']:.3f}")

    print("running MiniLM on sample thoughts…")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_hidden_states=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    pooled = {t: pooled_layers(model, tokenizer, t) for t in THOUGHTS}

    # calibrate spread: match median bridged norm to the latent cloud median
    probe = NumpyBridge(ck["latent_dim"], 384, dim_std, spread=1.0)
    norms = [np.linalg.norm(probe.map_layers(p)) for p in pooled.values()]
    spread = stats["norm_median"] / float(np.median(norms))
    print(f"calibration: bridged norm med {np.median(norms):.3f} -> spread {spread:.3f}")
    bridge = NumpyBridge(ck["latent_dim"], 384, dim_std, spread=spread)

    Path("data/bridge_calib.json").write_text(json.dumps({
        "bridgeSeed": BRIDGE_SEED, "gain1": GAIN1, "gain2": GAIN2,
        "spread": float(spread), "layerAlpha": LAYER_ALPHA,
    }, indent=2))

    out_dir = Path("previews/bridge")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, thought in enumerate(THOUGHTS):
        zs = bridge.map_layers(pooled[thought])
        latents = [torch.from_numpy(zs[l]).float() for l in (0, 3, 6)]
        strip(decoder, latents, out_dir / f"thought_{i}.png", g=48)
        print(f"  [{i}] {thought!r} -> {out_dir / f'thought_{i}.png'}")
    # near-duplicate morph check
    za = bridge.map_layers(pooled[THOUGHTS[0]])[6]
    zb = bridge.map_layers(pooled[THOUGHTS[1]])[6]
    strip(decoder,
          [torch.from_numpy(za * (1 - a) + zb * a).float() for a in np.linspace(0, 1, 6)],
          out_dir / "morph_pair.png", g=48)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
