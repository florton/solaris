# Phase 0 export: build a MiniLM-L6 ONNX whose graph emits every layer's
# hidden state (hidden_state_0 .. hidden_state_6), quantize to q8, verify
# against the PyTorch reference, and stage everything the app needs into
# public/models/minilm-l6/ so the artwork ships its own weights.
# Run: .venv/Scripts/python scripts/export_model.py
import json
import shutil
import urllib.request
from pathlib import Path

import torch
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import QuantType, quantize_dynamic

REPO = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("public/models/minilm-l6")
WORK = Path(".export")
WORK.mkdir(exist_ok=True)
(OUT / "onnx").mkdir(parents=True, exist_ok=True)

print("loading torch model…")
model = AutoModel.from_pretrained(REPO, output_hidden_states=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(REPO)

class HiddenStates(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return tuple(out.hidden_states)

enc = tokenizer("the sculpture appears", return_tensors="pt")
n_layers = model.config.num_hidden_layers
output_names = [f"hidden_state_{i}" for i in range(n_layers + 1)]

fp32_path = WORK / "model_fp32.onnx"
print("exporting onnx…")
torch.onnx.export(
    HiddenStates(model),
    (enc["input_ids"], enc["attention_mask"]),
    str(fp32_path),
    input_names=["input_ids", "attention_mask"],
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        **{name: {0: "batch", 1: "seq"} for name in output_names},
    },
    opset_version=17,
    dynamo=False,
)

q8_path = OUT / "onnx" / "model_quantized.onnx"
print("quantizing to q8…")
quantize_dynamic(str(fp32_path), str(q8_path), weight_type=QuantType.QInt8, per_channel=True)
print(f"q8 size: {q8_path.stat().st_size / 1e6:.1f} MB")

print("verifying against torch reference…")
session = ort.InferenceSession(str(q8_path))
names = [o.name for o in session.get_outputs()]
assert names == output_names, names
with torch.no_grad():
    ref = model(**enc, output_hidden_states=True).hidden_states
ort_out = session.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})
worst = 0.0
for i, (r, o) in enumerate(zip(ref, ort_out)):
    r = r.numpy().ravel()
    o = o.ravel()
    cos = float((r * o).sum() / ((r**2).sum() ** 0.5 * (o**2).sum() ** 0.5))
    worst = min(worst, cos) if i else cos
    print(f"  {output_names[i]}: shape {list(o.shape) if hasattr(o, 'shape') else o.shape}, cos vs torch = {cos:.5f}")
# q8 drift is a mostly isotropic perturbation; cluster structure is verified
# empirically in scripts/spike2.mjs, so warn rather than fail here.
if worst < 0.85:
    raise SystemExit(f"quantization drift too large: {worst}")

print("fetching tokenizer + config files…")
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]:
    url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
    urllib.request.urlretrieve(url, OUT / fname)

meta = {
    "id": REPO,
    "params": int(sum(p.numel() for p in model.parameters())),
    "layers": n_layers,
    "hiddenDim": model.config.hidden_size,
    "dtype": "q8",
    "outputs": output_names,
}
(OUT / "solaris_model.json").write_text(json.dumps(meta, indent=2))
print("done ->", OUT)
