// The bridge: pooled per-layer encoder embeddings -> latent codes for the
// imagination decoder. Fully deterministic: the projection matrices are
// generated from a fixed seed (mulberry32) at load, so the bridge is pure
// code — no weights file, same result in every browser. Smooth by
// construction: close embeddings land on close latents, so near-duplicate
// thoughts produce morphing forms rather than jumps. scripts/bridge_preview.py
// mirrors this math exactly (int32-compatible mulberry32) to calibrate and
// validate before anything ships.

export interface BridgeSpec {
  latentDim: number;
  hiddenDim: number;
  bridgeSeed: number;
  gain1: number; // pre-tanh gain on stage 1 (sqrt(3): unit-variance pre-activation)
  gain2: number; // pre-tanh gain on stage 2
  spread: number; // calibrated output scale (matches the latent cloud's reach)
  dimStd: number[]; // per-dim std of the trained latent cloud
  layerAlpha: number; // how far per-layer codes may depart from the base code
}

/** mulberry32 — must stay bit-compatible with the numpy mirror in
 *  scripts/bridge_preview.py (all arithmetic is int32/uint32 exact). */
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

function uniformMatrix(rand: () => number, rows: number, cols: number): Float32Array {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = rand() * 2 - 1;
  return m;
}

export class Bridge {
  private w1: Float32Array; // [hiddenDim x 64]
  private w2: Float32Array; // [64 x latentDim]
  private midDim = 64;

  constructor(public spec: BridgeSpec) {
    const rand = mulberry32(spec.bridgeSeed);
    this.w1 = uniformMatrix(rand, spec.hiddenDim, this.midDim);
    this.w2 = uniformMatrix(rand, this.midDim, spec.latentDim);
  }

  /** One embedding (unit-norm, hiddenDim) -> one latent code. */
  map(e: Float32Array, out: Float32Array, offset: number): void {
    const { hiddenDim, latentDim, gain1, gain2, spread, dimStd } = this.spec;
    const h = new Float32Array(this.midDim);
    for (let j = 0; j < this.midDim; j++) {
      let s = 0;
      for (let i = 0; i < hiddenDim; i++) s += e[i] * this.w1[i * this.midDim + j];
      h[j] = Math.tanh(s * gain1);
    }
    for (let k = 0; k < latentDim; k++) {
      let s = 0;
      for (let j = 0; j < this.midDim; j++) s += h[j] * this.w2[j * latentDim + k];
      out[offset + k] = Math.tanh(s * gain2) * dimStd[k] * spread;
    }
  }

  /** Per-layer latents: a shared base code from the mean embedding keeps the
   *  stack coherent; each layer departs from it by layerAlpha. */
  mapLayers(pooled: Float32Array[]): Float32Array {
    const L = pooled.length;
    const zDim = this.spec.latentDim;
    const mean = new Float32Array(pooled[0].length);
    for (const e of pooled) for (let i = 0; i < mean.length; i++) mean[i] += e[i] / L;
    const base = new Float32Array(zDim);
    this.map(mean, base, 0);
    const out = new Float32Array(L * zDim);
    const tmp = new Float32Array(zDim);
    for (let l = 0; l < L; l++) {
      this.map(pooled[l], tmp, 0);
      for (let k = 0; k < zDim; k++) {
        out[l * zDim + k] = base[k] + this.spec.layerAlpha * (tmp[k] - base[k]);
      }
    }
    return out;
  }
}

/** Mean-pool a layer's token rows and L2-normalize — mirrors the pooling in
 *  scripts/bridge_preview.py (specials are already stripped upstream). */
export function poolEmbedding(hidden: Float32Array, n: number, dims: number): Float32Array {
  const e = new Float32Array(dims);
  for (let i = 0; i < n; i++) for (let k = 0; k < dims; k++) e[k] += hidden[i * dims + k] / n;
  let norm = 0;
  for (let k = 0; k < dims; k++) norm += e[k] * e[k];
  norm = Math.sqrt(norm) || 1;
  for (let k = 0; k < dims; k++) e[k] /= norm;
  return e;
}
