// Phase 0/1 spike 2: run the custom exported ONNX (fp32 and q8) on a probe
// phrase, build per-layer centered cosine-similarity matrices, and check that
// semantic cluster structure survives quantization.
// Run: node scripts/spike2.mjs
import { AutoTokenizer } from '@huggingface/transformers';
import { InferenceSession } from 'onnxruntime-node';
import { layerGeometry } from '../.probe/geometry.mjs';

const TEXT = process.argv[2] ?? 'dog bark leash refrigerator';
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const inputs = await tokenizer(TEXT);
const ids = [...inputs.input_ids.data].map(Number);
const toks = ids.map((i) => tokenizer.decode_single([i], {}));
// drop [CLS]/[SEP]
const keep = toks.map((t, i) => (/^\[.*\]$/.test(t) ? -1 : i)).filter((i) => i >= 0);
console.log('tokens:', keep.map((i) => toks[i]).join(' | '));

function simsFromSession(session, feed, label) {
  const t0 = performance.now();
  return session.run(feed).then((out) => {
    const ms = performance.now() - t0;
    console.log(`\n=== ${label} (forward ${ms.toFixed(1)}ms) ===`);
    const names = Object.keys(out).sort();
    for (const name of names) {
      const tensor = out[name];
      const [seq, dims] = [tensor.dims[1], tensor.dims[2]];
      const n = keep.length;
      const rows = new Float32Array(n * dims);
      keep.forEach((tokIdx, r) => rows.set(tensor.data.slice(tokIdx * dims, (tokIdx + 1) * dims), r * dims));
      const g = layerGeometry(rows, n, dims);
      const cluster = [0, 1, 2]; // dog bark leash
      const outlier = 3; // refrigerator
      let within = 0, cross = 0;
      for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) within += g.similarity[i * n + j];
      within /= 3;
      for (const i of cluster) cross += g.similarity[i * n + outlier];
      cross /= 3;
      console.log(`${name}: within-cluster ${within.toFixed(3)}  vs outlier ${cross.toFixed(3)}  separation ${(within - cross).toFixed(3)}`);
    }
  });
}

const feed = {
  input_ids: new BigInt64Array(ids.map(BigInt)),
  attention_mask: new BigInt64Array(ids.map(() => 1n)),
};
// onnxruntime-node expects tensors:
import { Tensor } from 'onnxruntime-node';
const feedT = {
  input_ids: new Tensor('int64', feed.input_ids, [1, ids.length]),
  attention_mask: new Tensor('int64', feed.attention_mask, [1, ids.length]),
};

const fp32 = await InferenceSession.create('.export/model_fp32.onnx');
const q8 = await InferenceSession.create('public/models/minilm-l6/onnx/model_quantized.onnx');
await simsFromSession(fp32, feedT, 'fp32');
await simsFromSession(q8, feedT, 'q8');
// q8 timing runs
for (let i = 0; i < 3; i++) {
  const t = performance.now();
  await q8.run(feedT);
  console.log(`q8 forward ${i}: ${(performance.now() - t).toFixed(1)}ms`);
}
