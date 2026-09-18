// Phase 1 diagnosis: on real model outputs, compare raw vs centered cosine
// structure, and check the spectral embedding's cluster geometry end to end.
// Run: node scripts/spike3.mjs ["phrase"] ...
import { AutoTokenizer } from '@huggingface/transformers';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { cosineSimilarity, centerColumns, layerGeometry } from '../.probe/geometry.mjs';

const PHRASES = process.argv.length > 2 ? process.argv.slice(2) : [
  'dog bark leash refrigerator',
  'I miss the house I grew up in, the smell of rain on the wooden porch',
  'gravity quantum entropy toaster bicycle',
];

const tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
const q8 = await InferenceSession.create('public/models/minilm-l6/onnx/model_quantized.onnx');

for (const text of PHRASES) {
  const inputs = await tokenizer(text);
  const ids = [...inputs.input_ids.data].map(Number);
  const toks = ids.map((i) => tokenizer.decode_single([i], {}));
  const keep = toks.map((t, i) => (/^\[.*\]$/.test(t) ? -1 : i)).filter((i) => i >= 0);
  const out = await q8.run({
    input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]),
  });

  console.log(`\n### "${text}" (${keep.length} tokens)`);
  for (const name of Object.keys(out).sort()) {
    const tensor = out[name];
    const dims = tensor.dims[2];
    const n = keep.length;
    const rows = new Float32Array(n * dims);
    keep.forEach((tokIdx, r) => rows.set(tensor.data.slice(tokIdx * dims, (tokIdx + 1) * dims), r * dims));

    const raw = cosineSimilarity(rows, n, dims);
    const centeredRows = Float32Array.from(rows);
    centerColumns(centeredRows, n, dims);
    const cen = cosineSimilarity(centeredRows, n, dims);

    const stats = (m) => {
      let mn = 1, mx = -1, sum = 0, cnt = 0, neg = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const v = m[i * n + j];
        mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; cnt++;
        if (v < 0) neg++;
      }
      return `min ${mn.toFixed(2)} mean ${(sum / cnt).toFixed(2)} max ${mx.toFixed(2)} neg ${(100 * neg / cnt).toFixed(0)}%`;
    };
    console.log(`${name}: raw[${stats(raw)}]  centered[${stats(cen)}]`);
  }

  // embedding separation for the 4-token probe at the last layer
  if (text === 'dog bark leash refrigerator') {
    const last = out['hidden_state_6'];
    const dims = last.dims[2];
    const n = keep.length;
    const rows = new Float32Array(n * dims);
    keep.forEach((tokIdx, r) => rows.set(last.data.slice(tokIdx * dims, (tokIdx + 1) * dims), r * dims));
    const g = layerGeometry(rows, n, dims);
    const pos = g.positions;
    console.log('  last-layer embedding positions (raw cosine):');
    keep.forEach((_, r) => console.log(`    ${toks[keep[r]].padEnd(14)} ${[0, 1, 2].map((d) => pos[r * 3 + d].toFixed(2).padStart(6)).join(' ')}`));
    const dpair = (a, b) => Math.hypot(pos[a * 3] - pos[b * 3], pos[a * 3 + 1] - pos[b * 3 + 1], pos[a * 3 + 2] - pos[b * 3 + 2]);
    console.log(`  dog-bark ${dpair(0, 1).toFixed(2)}  dog-leash ${dpair(0, 2).toFixed(2)}  bark-leash ${dpair(1, 2).toFixed(2)}  | dog-fridge ${dpair(0, 3).toFixed(2)}  bark-fridge ${dpair(1, 3).toFixed(2)}  leash-fridge ${dpair(2, 3).toFixed(2)}`);
    console.log('  edges:', g.edges.map(([i, j, w]) => `${toks[keep[i]]}-${toks[keep[j]]}:${w.toFixed(2)}`).join(' '));
  }
}
