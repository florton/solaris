// Phase 1 exit check: do the embeddings cluster by meaning on real model output?
// For each probe phrase with hand-labeled groups, report the ratio
//   mean within-group distance / mean between-group distance
// per layer (lower = tighter semantic clusters).
// Run: node scripts/spike4.mjs
import { AutoTokenizer } from '@huggingface/transformers';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { layerGeometry } from '../.probe/geometry.mjs';

const PROBES = [
  { text: 'dog bark leash refrigerator', groups: { animal: ['dog', 'bark', 'leash'], other: ['refrigerator'] } },
  { text: 'gravity quantum entropy toaster bicycle', groups: { physics: ['gravity', 'quantum', 'entropy'], objects: ['toaster', 'bicycle'] } },
  {
    text: 'I miss the house I grew up in, the smell of rain on the wooden porch',
    groups: { memory: ['house', 'smell', 'rain', 'wooden', 'porch'], function: ['I', 'the', 'in', 'on'] },
  },
];

const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const q8 = await InferenceSession.create('public/models/minilm-l6/onnx/model_quantized.onnx');

for (const probe of PROBES) {
  const inputs = await tokenizer(probe.text);
  const ids = [...inputs.input_ids.data].map(Number);
  const toks = ids.map((i) => tokenizer.decode_single([i], {}));
  const keep = toks.map((t, i) => (/^\[.*\]$/.test(t) ? -1 : i)).filter((i) => i >= 0);
  const kept = keep.map((i) => toks[i]);
  const out = await q8.run({
    input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]),
  });

  // map group words to kept-token indices (wordpiece may split; match by prefix)
  const groupOf = new Array(kept.length).fill(null);
  for (const [gi, words] of Object.values(probe.groups).entries()) {
    for (const w of words) {
      const idx = kept.findIndex((t) => t.replace(/^##/, '') === w || w.startsWith(t.replace(/^##/, '')));
      if (idx >= 0) groupOf[idx] = gi;
    }
  }

  console.log(`\n### "${probe.text}"`);
  for (const name of Object.keys(out).sort()) {
    const tensor = out[name];
    const dims = tensor.dims[2];
    const n = keep.length;
    const rows = new Float32Array(n * dims);
    keep.forEach((tokIdx, r) => rows.set(tensor.data.slice(tokIdx * dims, (tokIdx + 1) * dims), r * dims));
    const g = layerGeometry(rows, n, dims);
    const pos = g.positions;
    const dist = (a, b) => Math.hypot(pos[a * 3] - pos[b * 3], pos[a * 3 + 1] - pos[b * 3 + 1], pos[a * 3 + 2] - pos[b * 3 + 2]);
    let within = 0, wCount = 0, between = 0, bCount = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (groupOf[i] === null || groupOf[j] === null) continue;
      if (groupOf[i] === groupOf[j]) { within += dist(i, j); wCount++; }
      else { between += dist(i, j); bCount++; }
    }
    const ratio = wCount && bCount ? (within / wCount) / (between / bCount) : NaN;
    console.log(`  ${name}: within/between ratio ${ratio.toFixed(2)}  (within ${(within / Math.max(wCount, 1)).toFixed(2)}, between ${(between / Math.max(bCount, 1)).toFixed(2)})`);
  }
}
