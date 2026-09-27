// The dreamer's plain math (the ONNX decoder lives in dream-worker.ts). A
// thought dreams one form per layer, all from the same neighbourhood of the
// library, and the layers become one 4D body. Mirrors scripts/mimoid_dream.py:
//   1. retrieve: cosine between the thought's sentence embedding (the last
//      layer's pooled state: what MiniLM was trained to output) and every
//      library caption; the top `pool` forms are the thought's neighbourhood
//   2. walk the layers deepest first, each drawing a different form from the
//      neighbourhood, weighted by exp((sim - best) / temp). temp widens toward
//      the shallow layers, so the deep layers dream the closest match and the
//      shallow ones looser associations of it
//   3. decode each form's latent, extend its truncated SDF to a full distance
//      field (so neighbouring layers can melt into each other along w)
// The only randomness is one mulberry32 stream seeded by FNV-1a(text): same
// thought, same forms, forever.
import { mulberry32 } from './bridge.ts';

export interface DreamMeta {
  n: number; // library forms
  layers: number;
  hiddenDim: number;
  embLayer: number; // which pooled layer retrieval reads (6: the sentence embedding)
  latShape: number[]; // [3, 8, 8, 8]
  latScale: number; // int8 -> normalized latent
  embScale: number[]; // [hiddenDim] int8 -> embedding
  grid: number; // decoder output is grid³, x fastest
  bound: number; // world half-extent of the grid (texel centers span [-1, 1])
  trunc: number; // the decoder's SDF is clipped to ±trunc (world units)
  dream: { pool: number; tempDeep: number; tempShallow: number; meltMin: number; meltMax: number; meltKeep: number };
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

export interface Pick {
  index: number;
  sim: number;
}

/** One distinct library form per layer, deepest layer first. */
export function pickForms(meta: DreamMeta, emb: Int8Array, query: Float32Array, seed: number): Pick[] {
  const { n, hiddenDim: D, layers: L } = meta;
  const { pool: K, tempDeep, tempShallow } = meta.dream;
  const q = new Float32Array(D);
  for (let d = 0; d < D; d++) q[d] = query[d] * meta.embScale[d];
  const topS = new Float64Array(K).fill(-Infinity);
  const topI = new Int32Array(K).fill(-1);
  for (let i = 0; i < n; i++) {
    const o = i * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += q[d] * emb[o + d];
    if (s <= topS[K - 1]) continue;
    let j = K - 1;
    while (j > 0 && topS[j - 1] < s) {
      topS[j] = topS[j - 1];
      topI[j] = topI[j - 1];
      j--;
    }
    topS[j] = s;
    topI[j] = i;
  }
  const rand = mulberry32(seed);
  const left = Array.from({ length: K }, (_, j) => j);
  const picks: Pick[] = new Array(L);
  for (let l = L - 1; l >= 0; l--) {
    const temp = tempShallow + (tempDeep - tempShallow) * (L <= 1 ? 1 : l / (L - 1));
    const u = rand();
    const w = left.map((j) => Math.exp((topS[j] - topS[0]) / temp));
    const total = w.reduce((a, b) => a + b, 0);
    let cum = 0;
    let k = left.length - 1;
    for (let m = 0; m < left.length; m++) {
      cum += w[m] / total;
      if (cum >= u) {
        k = m;
        break;
      }
    }
    const j = left[k];
    if (left.length > 1) left.splice(k, 1);
    picks[l] = { index: topI[j], sim: topS[j] };
  }
  return picks;
}

/** 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher). */
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/** Truncated SDF (world units, x fastest) -> full signed distance field: the
 *  band inside ±trunc is kept, beyond it the distance to the band (in voxels,
 *  via a 3-D EDT) is added on. Slightly under-estimates, which is the safe
 *  side for sphere tracing. Truncated fields can't be interpolated: two forms
 *  that don't overlap average to "outside" everywhere and the morph vanishes. */
export function fullSdf(sdf: Float32Array, g: number, h: number, trunc: number): Float32Array {
  const N = g * g * g;
  const BIG = 1e20;
  const band = trunc * 0.98;
  const f = new Float64Array(N);
  for (let i = 0; i < N; i++) f[i] = Math.abs(sdf[i]) < band ? 0 : BIG;
  const line = new Float64Array(g);
  const out = new Float64Array(g);
  const v = new Int32Array(g);
  const z = new Float64Array(g + 1);
  const strides = [1, g, g * g];
  for (let axis = 0; axis < 3; axis++) {
    const st = strides[axis];
    const [sa, sb] = strides.filter((_, k) => k !== axis);
    for (let a = 0; a < g; a++)
      for (let b = 0; b < g; b++) {
        const base = a * sa + b * sb;
        for (let i = 0; i < g; i++) line[i] = f[base + i * st];
        edt1d(line, g, out, v, z);
        for (let i = 0; i < g; i++) f[base + i * st] = out[i];
      }
  }
  const res = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const s = sdf[i];
    // (an empty grid has no band at all: cap the distance at two grid widths)
    res[i] = f[i] === 0 ? s : Math.sign(s || 1) * (band + (Math.min(Math.sqrt(f[i]), 2 * g) - 1) * h);
  }
  return res;
}

/** How much to inflate the half-way point of the a -> b melt so it keeps at
 *  least meltKeep of the smaller form's volume: a plain SDF lerp between
 *  dissimilar forms thins out and tears in the middle; inflated, it passes
 *  through one fused mass instead (the ocean melting one mimoid into the next). */
export function meltAmount(a: Float32Array, b: Float32Array, meta: DreamMeta): number {
  const { meltMin, meltMax, meltKeep } = meta.dream;
  let va = 0;
  let vb = 0;
  const mid = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    if (a[i] < 0) va++;
    if (b[i] < 0) vb++;
    mid[i] = 0.5 * (a[i] + b[i]);
  }
  const target = Math.floor(meltKeep * Math.min(va, vb));
  if (target <= 0) return meltMin;
  mid.sort();
  // inside at the midpoint iff mid - c < 0: c = the target-th smallest mid value
  return Math.min(meltMax, Math.max(meltMin, mid[target]));
}

/** Library latent i -> normalized float latent. */
export function libraryLatent(meta: DreamMeta, lat: Int8Array, i: number): Float32Array {
  const m = meta.latShape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(m);
  for (let j = 0; j < m; j++) out[j] = lat[i * m + j] * meta.latScale;
  return out;
}
