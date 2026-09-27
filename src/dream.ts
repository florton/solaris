// The dreamer's plain math (the ONNX decoder lives in dream-worker.ts). A
// thought dreams one form per layer, all from the same neighbourhood of the
// library, and the layers become one 4D body. Mirrors scripts/mimoid_dream.py:
//   1. split the thought into phrases at connecting words ("tea | grandmas
//      kitchen | a rainstorm"); phrases whose best match is under minSim are
//      dropped, and with one phrase (or none) left the whole thought is it.
//      Unsplit, the dominant noun owns every layer (the tea thought's top 24
//      were all teapots)
//   2. retrieve per phrase: cosine between its sentence embedding (the last
//      layer's pooled state: what MiniLM was trained to output) and every
//      library caption; the top `pool` forms, same-caption repeats counted
//      once, are the phrase's neighbourhood
//   3. walk the layers deepest first: the first phrase (the subject) takes the
//      deep layers, each later phrase the next ones up (`allot`), so the walk
//      along w reads as a scene. Each layer draws a different form from its
//      phrase's neighbourhood, weighted by exp((sim - best) / temp). temp
//      widens toward the shallow layers, so the deep layers dream the closest
//      match and the shallow ones looser associations of it
//   4. decode each form's latent, extend its truncated SDF to a full distance
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
  dream: {
    pool: number;
    tempDeep: number;
    tempShallow: number;
    meltMin: number;
    meltMax: number;
    meltKeep: number;
    minSim: number; // a phrase whose best match is weaker tells no story and is dropped
    dup: number; // same-caption repeats (embedding cosine above this) count once in a neighbourhood
    maxPhrases: number;
    connect: string[]; // a phrase ends at these words
    stop: string[]; // a phrase of only these words is dropped
  };
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
  phrase: number; // which of the dreamt phrases it came from
}

/** Split a thought at connecting words; phrases of only stop words are dropped. */
export function phrases(text: string, walk: DreamMeta['dream']): string[] {
  const connect = new Set(walk.connect ?? []);
  const stop = new Set(walk.stop ?? []);
  const out: string[][] = [];
  let cur: string[] = [];
  for (const w of text.toLowerCase().match(/[a-z0-9']+/g) ?? []) {
    if (connect.has(w)) {
      if (cur.length) out.push(cur);
      cur = [];
    } else cur.push(w);
  }
  if (cur.length) out.push(cur);
  return out.filter((p) => p.some((w) => !stop.has(w))).map((p) => p.join(' '));
}

/** Layers per phrase, the first (deepest) phrase first: [7], [5, 2], [3, 2, 2], [2, 2, 2, 1]. */
export function allot(nPhrases: number, nLayers: number): number[] {
  if (nPhrases <= 1) return [nLayers];
  const first = Math.max(2, nLayers - 2 * (nPhrases - 1));
  const rest = nLayers - first;
  const m = nPhrases - 1;
  return [first, ...Array.from({ length: m }, (_, i) => Math.floor(rest / m) + (i < rest % m ? 1 : 0))];
}

/** Cosine between a query embedding and every library caption. */
export function similarities(meta: DreamMeta, emb: Int8Array, query: Float32Array): Float32Array {
  const { n, hiddenDim: D } = meta;
  const q = new Float32Array(D);
  for (let d = 0; d < D; d++) q[d] = query[d] * meta.embScale[d];
  const sims = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += q[d] * emb[o + d];
    sims[i] = s;
  }
  return sims;
}

/** Cosine between two library captions' (dequantized) embeddings. */
function captionCos(meta: DreamMeta, emb: Int8Array, i: number, j: number): number {
  const D = meta.hiddenDim;
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let d = 0; d < D; d++) {
    const s2 = meta.embScale[d] * meta.embScale[d];
    const a = emb[i * D + d];
    const b = emb[j * D + d];
    ab += a * b * s2;
    aa += a * a * s2;
    bb += b * b * s2;
  }
  return ab / (Math.sqrt(aa * bb) || 1);
}

/** The top `pool` forms not yet taken, skipping same-caption repeats. */
function neighbourhood(meta: DreamMeta, emb: Int8Array, sims: Float32Array, taken: Set<number>): number[] {
  const order = Array.from({ length: meta.n }, (_, i) => i).sort((a, b) => sims[b] - sims[a] || a - b);
  const top: number[] = [];
  for (const i of order) {
    if (top.length === meta.dream.pool) break;
    if (taken.has(i) || top.some((j) => captionCos(meta, emb, i, j) > meta.dream.dup)) continue;
    top.push(i);
  }
  return top;
}

/** One distinct library form per layer, deepest layer first: the first
 *  phrase's sims take the deep layers, each later phrase's the next ones up. */
export function pickForms(meta: DreamMeta, emb: Int8Array, simsPerPhrase: Float32Array[], seed: number): Pick[] {
  const { layers: L } = meta;
  const { tempDeep, tempShallow } = meta.dream;
  const rand = mulberry32(seed);
  const picks: Pick[] = new Array(L);
  const taken = new Set<number>();
  const counts = allot(simsPerPhrase.length, L);
  let l = L - 1;
  simsPerPhrase.forEach((sims, p) => {
    const top = neighbourhood(meta, emb, sims, taken);
    const left = top.map((_, j) => j);
    for (let c = 0; c < counts[p]; c++, l--) {
      const temp = tempShallow + (tempDeep - tempShallow) * (L <= 1 ? 1 : l / (L - 1));
      const u = rand();
      const w = left.map((j) => Math.exp((sims[top[j]] - sims[top[0]]) / temp));
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
      picks[l] = { index: top[j], sim: sims[top[j]], phrase: p };
      taken.add(top[j]);
    }
  });
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
