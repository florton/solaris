// Shared helpers between the encoder and the dreamer (dream.ts): the pooled
// per-layer embedding of a thought, and the portable PRNG. Both have exact
// Python mirrors so previews match the browser (scripts/mimoid_captions.py
// pools, scripts/mimoid_dream.py seeds).

/** mulberry32 — must stay bit-compatible with the mirrors in
 *  scripts/mimoid_dream.py and scripts/bridge_preview.py (all arithmetic is
 *  int32/uint32 exact). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Mean-pool a layer's token rows and L2-normalize — mirrors the pooling in
 *  scripts/mimoid_captions.py (specials are already stripped upstream). */
export function poolEmbedding(hidden: Float32Array, n: number, dims: number): Float32Array {
  const e = new Float32Array(dims);
  for (let i = 0; i < n; i++) for (let k = 0; k < dims; k++) e[k] += hidden[i * dims + k] / n;
  let norm = 0;
  for (let k = 0; k < dims; k++) norm += e[k] * e[k];
  norm = Math.sqrt(norm) || 1;
  for (let k = 0; k < dims; k++) e[k] /= norm;
  return e;
}
