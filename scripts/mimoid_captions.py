# Stage 2 prep: a MiniLM embedding for every library form's caption, pooled
# exactly the way the app pools a thought (src/model.ts forward +
# src/bridge.ts poolEmbedding): every hidden state 0..6, specials dropped
# ([CLS]/[SEP]/[UNK]/…), mean over the remaining tokens, L2-normalized.
# The dreaming prior learns caption embedding -> form latent from these pairs.
#
# Captions: Cap3D (Luo et al. 2023), tiange/Cap3D on Hugging Face,
# Objaverse_files/cap3d_captions.json.gz — ODC-By 1.0, credit kept in the output.
# Run: .venv/Scripts/python scripts/mimoid_captions.py
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS = ROOT / "data" / "cap3d" / "Objaverse_files" / "cap3d_captions.json.gz"
UIDS = ROOT / "data" / "mimoid" / "library_all_uids.json"
OUT = ROOT / "data" / "mimoid_captions.npz"
REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 128  # the app's tokenizer.encode(text, 128)


def embed(texts, log=False):
    """[N, 7, 384] float32 — per-layer pooled, unit-norm, as the app computes a thought."""
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO).eval()
    special = torch.tensor(sorted(tok.all_special_ids))
    out = np.zeros((len(texts), 7, 384), np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            enc = tok(texts[i : i + 256], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            hs = model(**enc, output_hidden_states=True).hidden_states  # 7 × [B, T, 384]
            keep = (enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special)).float()
            n = keep.sum(1, keepdim=True).clamp(min=1)
            e = torch.stack([(h * keep[..., None]).sum(1) / n for h in hs], 1)  # [B, 7, 384]
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            out[i : i + len(e)] = e.numpy()
            if log and i % 2560 == 0:
                print(f"  {i + len(e)}/{len(texts)}", flush=True)
    return out


def main():
    caps = json.load(gzip.open(CAPTIONS))
    uids = json.loads(UIDS.read_text())["uids"]  # same order as library_all.npy / the AE latents
    missing = [u for u in uids if u not in caps]
    if missing:  # objects Cap3D's release lacks: the manifest's caption (cap3d.csv or the LVIS name)
        rows = (json.loads(l) for l in open(ROOT / "data" / "mimoid" / "library.jsonl", encoding="utf-8"))
        caps.update({r["uid"]: r["caption"] for r in rows if r["uid"] in set(missing)})
    texts = [caps[u].strip() for u in uids]
    print(f"{len(texts)} captions ({len(missing)} from the manifest)")
    out = embed(texts, log=True).astype(np.float16)

    np.savez(OUT, uids=np.array(uids), captions=np.array(texts), emb=out,
             credit=np.array("Captions: Cap3D (Luo et al. 2023, tiange/Cap3D), ODC-By 1.0"))
    # sanity: nearest captions by last-layer embedding for a few forms
    last = out[:, 6].astype(np.float32)
    for q in (0, 4, 100):
        sims = last @ last[q]
        nn_ = np.argsort(-sims)[1:4]
        print(f"'{texts[q]}' ~ " + " | ".join(f"'{texts[j]}' {sims[j]:.2f}" for j in nn_))
    print(f"-> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
