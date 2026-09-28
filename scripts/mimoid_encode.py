# Grow the library without retraining: encode newly built grids with the grid
# AE's trained encoder (data/mimoid_grid_ckpt.pt, the same weights that made
# the stored latents) and append them to data/mimoid_grid.pt. Existing rows
# keep their order, so data/mimoid_quality.npz rows stay aligned (it appends
# the new ones itself). Checks first that re-encoding old forms reproduces
# their stored latents. The first run keeps the original as mimoid_grid_base.pt.
# Then: mimoid_captions.py -> mimoid_quality.py -> export_dream.py
# Run: .venv/Scripts/python scripts/mimoid_encode.py
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimoid_train_ae import GridEncoder, load_all  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AE = ROOT / "data" / "mimoid_grid.pt"
CKPT = ROOT / "data" / "mimoid_grid_ckpt.pt"


def encode(enc, grids, idx, dev):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            x = torch.from_numpy(np.asarray(grids[idx[i : i + 32]]).astype(np.float32) / 127).to(dev)
            out.append(enc(x)[0].float().cpu())
    return torch.cat(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grids, uids = load_all("library")
    ae = torch.load(AE, weights_only=False)
    ck = torch.load(CKPT, weights_only=False)
    n_old = len(ae["uids"])
    if list(ae["uids"]) != uids[:n_old]:
        raise SystemExit("library order changed: the stored latents are no longer a prefix of library_all")
    if any(not torch.equal(ck["dec"][k], ae["dec"][k]) for k in ae["dec"]):
        raise SystemExit(f"{CKPT.name} is not the checkpoint that made {AE.name}")
    print(f"{len(uids)} forms · {n_old} encoded · {len(uids) - n_old} new", flush=True)
    if len(uids) == n_old:
        return
    enc = GridEncoder(ae["lat_c"]).to(dev).eval()
    enc.load_state_dict(ck["enc"])

    probe = np.arange(0, n_old, max(1, n_old // 64))[:64]
    err = float((encode(enc, grids, probe, dev) - ae["latents"][probe].float()).abs().max())
    print(f"  re-encoding {len(probe)} stored forms: max |diff| {err:.4f} (latent std ~0.13)", flush=True)
    if err > 0.02:
        raise SystemExit("the encoder doesn't reproduce the stored latents")

    new = encode(enc, grids, np.arange(n_old, len(uids)), dev)
    base = AE.with_name(AE.stem + "_base.pt")
    if not base.exists():
        shutil.copyfile(AE, base)
    ae["latents"] = torch.cat([ae["latents"], new.half()])
    ae["uids"] = uids
    torch.save(ae, AE)
    print(f"-> {AE} ({len(uids)} latents; the pre-extension file is {base.name})")


if __name__ == "__main__":
    main()
