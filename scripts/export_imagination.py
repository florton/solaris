# Export the trained imagination decoder to ONNX (fp32 — q8 risks visible
# raymarch artifacts, and the decoder is only ~200 KB anyway). The graph takes
# coords [M,3] + z [1,latent] (expanded internally), so the worker sends one
# latent per layer regardless of grid size. Verifies numerically against torch
# at two different M, then stages public/models/imagination/.
# Run: .venv/Scripts/python scripts/export_imagination.py
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from train_imagination import Decoder, LATENT_DIM, WIDTH, DEPTH, FOURIER_BANDS, BOUND

OUT = Path("public/models/imagination")
OUT.mkdir(parents=True, exist_ok=True)


class Exportable(torch.nn.Module):
    """coords [M,3], z [1,latent] -> sdf [M]; z expanded inside the graph."""

    def __init__(self, dec: Decoder):
        super().__init__()
        self.dec = dec

    def forward(self, coords, z):
        return self.dec(coords, z.expand(coords.shape[0], -1))


def main():
    ck = torch.load("data/imagination.pt", weights_only=False)
    dec = Decoder(ck["latent_dim"], ck["bands"], ck["width"], ck["depth"])
    dec.load_state_dict(ck["decoder"])
    dec.eval()

    stats = json.loads(Path("data/latent_stats.json").read_text())
    calib = json.loads(Path("data/bridge_calib.json").read_text())

    ex = Exportable(dec)
    coords = torch.randn(64, 3) * 0.7
    z = torch.randn(1, ck["latent_dim"]) * 0.3
    onnx_path = OUT / "decoder.onnx"
    torch.onnx.export(
        ex,
        (coords, z),
        str(onnx_path),
        input_names=["coords", "z"],
        output_names=["sdf"],
        dynamic_axes={"coords": {0: "M"}, "sdf": {0: "M"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"onnx: {onnx_path} ({onnx_path.stat().st_size / 1e3:.0f} KB)")

    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(3)
    worst = 0.0
    for m in (17, 65536):
        q = (rng.uniform(-BOUND, BOUND, size=(m, 3))).astype(np.float32)
        zz = rng.normal(0, 0.3, size=(1, ck["latent_dim"])).astype(np.float32)
        ref = ex(torch.from_numpy(q), torch.from_numpy(zz)).detach().numpy()
        got = sess.run(None, {"coords": q, "z": zz})[0]
        err = float(np.abs(ref - got).max())
        worst = max(worst, err)
        print(f"  M={m}: max |onnx - torch| = {err:.2e}")
    if worst > 1e-4:
        raise SystemExit(f"export drift too large: {worst}")

    meta = {
        "latentDim": ck["latent_dim"],
        "hiddenDim": 384,
        "bands": ck["bands"],
        "width": ck["width"],
        "depth": ck["depth"],
        "params": int(sum(p.numel() for p in dec.parameters())),
        "dtype": "fp32",
        "bound": ck["bound"],
        "dimStd": stats["dim_std"],
        "normMedian": stats["norm_median"],
        "grid": {"webgpu": 64, "wasm": 40},
        **calib,
    }
    (OUT / "imagination_model.json").write_text(json.dumps(meta, indent=2))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
