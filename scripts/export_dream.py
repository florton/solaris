# Export the mimoid dreamer (scripts/mimoid_dream.py, the reference) for the browser:
#   decoder.onnx     live latent [1, 3, 8, 8, 8] (normalized) -> sdf [64, 64, 64]
#                    in world units, x-fastest (texImage3D order). Dead channels
#                    and the de-normalization are a fixed 1×1×1 conv in front.
#   library_lat.bin  int8 [N, 3·8·8·8] normalized live latents (× latScale),
#                    one fixed-size row per form so the browser can Range-fetch
#                    single rows before the whole file has arrived
#   library_emb.bin  int8 [N, 384] caption sentence embeddings (pooled layer 6,
#                    × embScale[d]) — the retrieval index
#   captions.json    what each library form is (shown in the stats line)
#   credits.json     [uid, name, author, license] per form (Sketchfab via Objaverse)
#   dream_model.json the rest: shapes, scales, the walk/melt parameters
# Only forms that pass scripts/mimoid_quality.py are shipped. The graph avoids
# what ORT WebGPU struggles with: no Resize (nearest 2× is expand + reshape)
# and no tensor above 6-D. Verified against torch, staged in public/models/dream/.
# The SDEdit prior is no longer shipped (see mimoid_dream.py).
# Run: .venv/Scripts/python scripts/export_dream.py
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimoid_data import G, TRUNC  # noqa: E402
from mimoid_dream import AE, CAPS, EMB_LAYER, N_LAYERS, PRIOR, WALK  # noqa: E402
from mimoid_quality import OUT as QUALITY, keep  # noqa: E402
from mimoid_train_ae import GridDecoder, d2s  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "models" / "dream"


def up2(x):
    """nearest 2× on [B, C, D, H, W] without Resize: one axis at a time, ≤ 6-D"""
    b, c, d, h, w = x.shape
    x = x.reshape(b, c, d, h, w, 1).expand(b, c, d, h, w, 2).reshape(b, c, d, h, 2 * w)
    x = x.reshape(b, c, d, h, 1, 2 * w).expand(b, c, d, h, 2, 2 * w).reshape(b, c, d, 2 * h, 2 * w)
    x = x.reshape(b, c, d, 1, 4 * h * w).expand(b, c, d, 2, 4 * h * w).reshape(b, c, 2 * d, 2 * h, 2 * w)
    return x


class DecoderExport(nn.Module):
    def __init__(self, dec: GridDecoder, live, mean, std):
        super().__init__()
        self.dec = dec
        c = mean.numel()
        self.lift = nn.Conv3d(len(live), c, 1)  # z_full = mean + scatter(std · z_live)
        with torch.no_grad():
            self.lift.weight.zero_()
            for j, ch in enumerate(live):
                self.lift.weight[ch, j] = std[ch]
            self.lift.bias.copy_(mean)

    def forward(self, z):
        d = self.dec
        x = F.silu(d.inp(self.lift(z)))
        for r in d.res0:
            x = x + F.silu(r(x))
        for a, b in d.stages:
            x = F.silu(a(up2(x)))
            x = x + F.silu(b(x))
        x = d.out(x)[0]  # [8, 32, 32, 32]; batch 1 keeps depth-to-space at 6-D
        _, X, Y, Z = x.shape
        # d2s gives [x, y, z] with z fastest; the texture wants x fastest -> [z, y, x]
        x = x.reshape(2, 2, 2, X, Y, Z).permute(5, 2, 4, 1, 3, 0).reshape(2 * Z, 2 * Y, 2 * X)
        return x * TRUNC


def check(name, sess, feeds, ref):
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})[0]
    err = float(np.abs(got - ref).max())
    print(f"  {name}: max |onnx - torch| = {err:.2e}")
    if err > 2e-3:
        raise SystemExit(f"{name}: export drift too large")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("prior.onnx",):
        (OUT / stale).unlink(missing_ok=True)
    ae = torch.load(AE, weights_only=False)
    pr = torch.load(PRIOR, weights_only=False)  # only for the live channels + latent normalization
    caps = np.load(CAPS)
    live = [int(c) for c in pr["live"]]
    mean, std = pr["latent_mean"].float().cpu(), pr["latent_std"].float().cpu()

    dec = GridDecoder(ae["lat_c"], ae["channels"]).eval()
    dec.load_state_dict(ae["dec"])
    dx = DecoderExport(dec, live, mean, std).eval()

    kept = np.nonzero(keep(dict(np.load(QUALITY))))[0]
    lat = (ae["latents"].float()[:, live] - mean[live, None, None, None]) / std[live, None, None, None]
    lat = lat[torch.from_numpy(kept)]
    z = lat[:1].clone()
    with torch.no_grad():
        full = mean[None, :, None, None, None].expand(1, -1, 8, 8, 8).clone()
        full[:, live] = z * std[live, None, None, None] + mean[live, None, None, None]
        ref = dec(full)[0].permute(2, 1, 0) * TRUNC  # the original decoder, transposed to x-fastest
        assert torch.allclose(dx(z), ref, atol=1e-4), "DecoderExport differs from GridDecoder"
    torch.onnx.export(dx, (z,), str(OUT / "decoder.onnx"), input_names=["z"], output_names=["sdf"],
                      opset_version=17, dynamo=False)
    check("decoder", ort.InferenceSession(str(OUT / "decoder.onnx")), {"z": z}, ref.numpy())

    # library: int8 latents with one global scale; embeddings per dim
    lat_np = lat.reshape(len(lat), -1).numpy()
    lat_scale = float(np.abs(lat_np).max() / 127)
    (OUT / "library_lat.bin").write_bytes(np.round(lat_np / lat_scale).astype(np.int8).tobytes())
    emb = caps["emb"][kept, EMB_LAYER].astype(np.float32)  # [N, 384]
    emb_scale = np.maximum(np.abs(emb).max(0), 1e-6) / 127  # [384]; some dims are always 0
    q = np.round(emb / emb_scale).astype(np.int8)
    (OUT / "library_emb.bin").write_bytes(np.ascontiguousarray(q).tobytes())
    deq = q.astype(np.float32) * emb_scale
    cos = (deq * emb).sum(-1) / np.linalg.norm(deq, axis=-1)
    print(f"  library: {len(kept)} of {len(ae['latents'])} forms kept; latent int8 step {lat_scale:.4f} (unit std); "
          f"embedding cos after int8 >= {cos.min():.4f}")
    (OUT / "captions.json").write_text(json.dumps([str(c).strip() for c in caps["captions"][kept]]), encoding="utf-8")
    # attribution for every shipped form (CC-BY / CC-BY-SA need it): [uid, name, author, license], same order
    manifest = {}
    for line in (ROOT / "data" / "mimoid" / "library.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        manifest[r["uid"]] = r
    credits = [[str(u), manifest[str(u)]["name"], manifest[str(u)]["author"], manifest[str(u)]["license"]] for u in caps["uids"][kept]]
    (OUT / "credits.json").write_text(json.dumps(credits, ensure_ascii=False), encoding="utf-8")

    meta = {
        "n": len(lat), "layers": N_LAYERS, "hiddenDim": 384, "embLayer": EMB_LAYER, "latShape": [len(live), 8, 8, 8],
        "latScale": lat_scale, "embScale": np.round(emb_scale, 8).tolist(),
        "grid": G, "bound": G / (G - 1),  # grid samples span [-1, 1] at texel centers
        "trunc": TRUNC,
        "dream": WALK,
        "params": int(sum(p.numel() for p in dec.parameters())),
        "dtype": "fp32",
        "credit": str(caps["credit"]) + "; forms: Objaverse (CC-BY / CC-BY-SA / CC0)",
    }
    (OUT / "dream_model.json").write_text(json.dumps(meta))
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
