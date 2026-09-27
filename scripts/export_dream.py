# Export the mimoid dreamer (scripts/mimoid_dream.py) for the browser:
#   decoder.onnx   live latent [1, 3, 8, 8, 8] (normalized) -> sdf [64, 64, 64]
#                  in world units, x-fastest (texImage3D order). Dead channels
#                  and the de-normalization are a fixed 1×1×1 conv in front.
#   prior.onnx     one guided denoiser call: x [B, 3, 8, 8, 8], t [B],
#                  emb [B, 384], layer [B] (int64), guidance [1] -> v [B, ...];
#                  runs cond + null (CFG) as one batch of 2B inside the graph.
#   library_lat.bin  int8 [N, 3·8·8·8] normalized live latents (× latScale)
#   library_emb.bin  int8 [7, N, 384] caption embeddings (× embScale[l][d])
#   captions.json    what each library form is (shown in the stats line)
#   dream_model.json the rest: shapes, scales, dreaming parameters
# Both graphs avoid what ORT WebGPU struggles with: no Resize (nearest 2× is
# expand + reshape) and no tensor above 6-D (depth-to-space drops the batch).
# Verifies both against torch, then stages public/models/dream/.
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
from mimoid_dream import AE, CAPS, N_LAYERS, PRIOR  # noqa: E402
from mimoid_train_ae import GridDecoder, d2s  # noqa: E402
from mimoid_train_prior import Prior, timestep_embedding  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "models" / "dream"
DREAM = {"t0": 0.4, "steps": 8, "k": 8, "temp": 0.05, "guidance": 3.0}


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


class PriorExport(nn.Module):
    def __init__(self, p: Prior):
        super().__init__()
        self.p = p

    def forward(self, x, t, emb, layer, guidance):
        p = self.p
        c = p.cond(emb, layer)
        c = torch.cat([c, p.null.expand_as(c)])
        x2, t2 = torch.cat([x, x]), torch.cat([t, t])
        e = p.t_mlp(timestep_embedding(t2)) + p.c_mlp(c)
        h = p.inp(x2)
        for r in p.d1:
            h = r(h, e)
        skip = h
        h = p.down(h)
        for r, a in zip(p.d2, p.a2):
            h = a(r(h, e))
        h = p.up(up2(h))
        h = torch.cat([h, skip], 1)
        for r in p.u1:
            h = r(h, e)
        vc, vu = p.out(F.silu(p.out_n(h))).chunk(2)
        return vu + guidance * (vc - vu)


def check(name, sess, feeds, ref):
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})[0]
    err = float(np.abs(got - ref).max())
    print(f"  {name}: max |onnx - torch| = {err:.2e}")
    if err > 2e-3:
        raise SystemExit(f"{name}: export drift too large")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ae = torch.load(AE, weights_only=False)
    pr = torch.load(PRIOR, weights_only=False)
    caps = np.load(CAPS)
    live = [int(c) for c in pr["live"]]
    mean, std = pr["latent_mean"].float().cpu(), pr["latent_std"].float().cpu()

    dec = GridDecoder(ae["lat_c"], ae["channels"]).eval()
    dec.load_state_dict(ae["dec"])
    prior = Prior(len(live)).eval()
    prior.load_state_dict(pr["prior"])
    dx, px = DecoderExport(dec, live, mean, std).eval(), PriorExport(prior).eval()

    lat = (ae["latents"].float()[:, live] - mean[live, None, None, None]) / std[live, None, None, None]
    z = lat[:1].clone()
    with torch.no_grad():
        full = mean[None, :, None, None, None].expand(1, -1, 8, 8, 8).clone()
        full[:, live] = z * std[live, None, None, None] + mean[live, None, None, None]
        ref = dec(full)[0].permute(2, 1, 0) * TRUNC  # the original decoder, transposed to x-fastest
        assert torch.allclose(dx(z), ref, atol=1e-4), "DecoderExport differs from GridDecoder"
    torch.onnx.export(dx, (z,), str(OUT / "decoder.onnx"), input_names=["z"], output_names=["sdf"],
                      opset_version=17, dynamo=False)

    B = 3
    feeds = {"x": torch.randn(B, len(live), 8, 8, 8), "t": torch.rand(B), "emb": F.normalize(torch.randn(B, 384), dim=1),
             "layer": torch.tensor([0, 3, 6]), "guidance": torch.tensor([3.0])}
    torch.onnx.export(px, tuple(feeds.values()), str(OUT / "prior.onnx"), input_names=list(feeds), output_names=["v"],
                      dynamic_axes={"x": {0: "B"}, "t": {0: "B"}, "emb": {0: "B"}, "layer": {0: "B"}, "v": {0: "B"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        # reference: the training-time sampler's guided v
        c = prior.cond(feeds["emb"], feeds["layer"])
        vc = prior(feeds["x"], feeds["t"], c)
        vu = prior(feeds["x"], feeds["t"], prior.null.expand_as(c))
        vref = (vu + 3.0 * (vc - vu)).numpy()
    check("decoder", ort.InferenceSession(str(OUT / "decoder.onnx")), {"z": z}, ref.numpy())
    check("prior", ort.InferenceSession(str(OUT / "prior.onnx")), feeds, vref)

    # library: int8 latents with one global scale; embeddings per (layer, dim)
    lat_np = lat.reshape(len(lat), -1).numpy()
    lat_scale = float(np.abs(lat_np).max() / 127)
    (OUT / "library_lat.bin").write_bytes(np.round(lat_np / lat_scale).astype(np.int8).tobytes())
    emb = caps["emb"].astype(np.float32).transpose(1, 0, 2)  # [7, N, 384]
    emb_scale = np.maximum(np.abs(emb).max(1), 1e-6) / 127  # [7, 384]; some dims are always 0
    q = np.round(emb / emb_scale[:, None]).astype(np.int8)
    (OUT / "library_emb.bin").write_bytes(np.ascontiguousarray(q).tobytes())
    deq = q.astype(np.float32) * emb_scale[:, None]
    cos = (deq * emb).sum(-1) / np.linalg.norm(deq, axis=-1)
    print(f"  library: latent int8 step {lat_scale:.4f} (unit std); embedding cos after int8 ≥ {cos.min():.4f}")
    (OUT / "captions.json").write_text(json.dumps([str(c).strip() for c in caps["captions"]]), encoding="utf-8")

    meta = {
        "n": len(lat), "layers": N_LAYERS, "hiddenDim": 384, "latShape": [len(live), 8, 8, 8],
        "latScale": lat_scale, "embScale": np.round(emb_scale, 8).tolist(),
        "grid": G, "bound": G / (G - 1),  # grid samples span [-1, 1] at texel centers
        "dream": DREAM,
        "params": int(sum(p.numel() for p in dec.parameters()) + sum(p.numel() for p in prior.parameters())),
        "dtype": "fp32",
        "credit": str(caps["credit"]) + "; forms: Objaverse (CC-BY / CC-BY-SA / CC0)",
    }
    (OUT / "dream_model.json").write_text(json.dumps(meta))
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
