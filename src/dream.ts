// The dreamer's plain math (the ONNX sessions live in worker.ts): a thought
// dreams one form per layer by retrieval + SDEdit. Mirrors
// scripts/mimoid_dream.py, the reference:
//   1. retrieve: cosine between the layer's pooled embedding and every library
//      caption embedding of that layer; pick one of the top k, weighted by
//      exp((sim - best) / temp)
//   2. noise the picked form's latent to t0
//   3. denoise it back with the prior, conditioned on the thought (DDIM + CFG)
// All randomness is one mulberry32 stream seeded by FNV-1a(text): 7 pick
// uniforms, then the noise field shared by every layer. Same thought, same forms.
import { mulberry32 } from './bridge.ts';

export interface DreamMeta {
  n: number; // library forms
  layers: number;
  hiddenDim: number;
  latShape: number[]; // [3, 8, 8, 8]
  latScale: number; // int8 -> normalized latent
  embScale: number[][]; // [layers][hiddenDim] int8 -> embedding
  grid: number; // decoder output is grid³, x fastest
  bound: number; // world half-extent of the grid (texel centers span [-1, 1])
  dream: { t0: number; steps: number; k: number; temp: number; guidance: number };
  params: number;
  dtype: string;
  credit: string;
}

/** 32-bit FNV-1a over the UTF-8 bytes. */
export function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(text)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h;
}

/** The thought's random stream: per-layer pick uniforms + one Box-Muller noise field. */
export function dreamStream(seed: number, nLayers: number, nNoise: number): { picks: number[]; noise: Float32Array } {
  const rand = mulberry32(seed);
  const picks = Array.from({ length: nLayers }, () => rand());
  const noise = new Float32Array(nNoise);
  for (let i = 0; i < nNoise; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - rand()));
    const a = 2 * Math.PI * rand();
    noise[i] = r * Math.cos(a);
    noise[i + 1] = r * Math.sin(a);
  }
  return { picks, noise };
}

/** Top-k caption match for one layer, then a weighted pick with uniform u. */
export function retrieve(
  meta: DreamMeta,
  emb: Int8Array, // [layers, n, hiddenDim]
  layer: number,
  query: Float32Array,
  u: number,
): { index: number; sim: number } {
  const { n, hiddenDim: D } = meta;
  const { k, temp } = meta.dream;
  const q = new Float32Array(D);
  for (let d = 0; d < D; d++) q[d] = query[d] * meta.embScale[layer][d];
  const topS = new Float64Array(k).fill(-Infinity);
  const topI = new Int32Array(k).fill(-1);
  const base = layer * n * D;
  for (let i = 0; i < n; i++) {
    const o = base + i * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += q[d] * emb[o + d];
    if (s <= topS[k - 1]) continue;
    let j = k - 1;
    while (j > 0 && topS[j - 1] < s) {
      topS[j] = topS[j - 1];
      topI[j] = topI[j - 1];
      j--;
    }
    topS[j] = s;
    topI[j] = i;
  }
  const w = Array.from(topS, (s) => Math.exp((s - topS[0]) / temp));
  const total = w.reduce((a, b) => a + b, 0);
  let cum = 0;
  for (let j = 0; j < k; j++) {
    cum += w[j] / total;
    if (cum >= u) return { index: topI[j], sim: topS[j] };
  }
  return { index: topI[k - 1], sim: topS[k - 1] };
}

/** Cosine schedule, clipped like training so t = 1 is (almost) pure noise. */
export function alphaSigma(t: number): [number, number] {
  const c = Math.min(Math.max(t, 0), 0.999);
  return [Math.cos((c * Math.PI) / 2), Math.sin((c * Math.PI) / 2)];
}

/** Library latent i -> normalized float latent. */
export function libraryLatent(meta: DreamMeta, lat: Int8Array, i: number): Float32Array {
  const m = meta.latShape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(m);
  for (let j = 0; j < m; j++) out[j] = lat[i * m + j] * meta.latScale;
  return out;
}
