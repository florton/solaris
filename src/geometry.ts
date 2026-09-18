// Geometry: hidden states -> per-layer similarity -> spectral embedding -> edges.
// Everything here is deterministic: no randomness, no sampling.

export interface LayerGeometry {
  /** n*3 spectral positions, centered, unit RMS radius */
  positions: Float32Array;
  /** n*n centered cosine similarity */
  similarity: Float32Array;
  /** per-token salience (norm of centered hidden state) */
  salience: Float32Array;
  /** [i, j, weight] filament list, i < j, deduplicated */
  edges: Array<[number, number, number]>;
}

/** Mean-center columns of an [n x dims] matrix in place. */
export function centerColumns(data: Float32Array, n: number, dims: number): void {
  for (let k = 0; k < dims; k++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += data[i * dims + k];
    mean /= n;
    for (let i = 0; i < n; i++) data[i * dims + k] -= mean;
  }
}

/** n*n cosine similarity of the rows of [n x dims]. */
export function cosineSimilarity(data: Float32Array, n: number, dims: number): Float32Array {
  const norms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < dims; k++) s += data[i * dims + k] * data[i * dims + k];
    norms[i] = Math.sqrt(s);
  }
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    out[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < dims; k++) dot += data[i * dims + k] * data[j * dims + k];
      const denom = norms[i] * norms[j];
      const v = denom > 1e-12 ? dot / denom : 0;
      out[i * n + j] = v;
      out[j * n + i] = v;
    }
  }
  return out;
}

/** Row norms of [n x dims]. */
export function rowNorms(data: Float32Array, n: number, dims: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < dims; k++) s += data[i * dims + k] * data[i * dims + k];
    out[i] = Math.sqrt(s);
  }
  return out;
}

/**
 * Jacobi eigenvalue algorithm for a real symmetric n*n matrix (row-major).
 * Returns eigenvalues ascending and eigenvectors as columns of `vectors` (row-major n*n).
 * Deterministic: fixed sweep count, no randomness.
 */
export function jacobiEigen(matrix: Float64Array, n: number, maxSweeps = 32): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i * n + j] * a[i * n + j];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-15) continue;
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p];
          const vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = a[i * n + i];

  // sort ascending by eigenvalue, carrying eigenvector columns along
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => values[x] - values[y]);
  const sv = new Float64Array(n);
  const sm = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    sv[r] = values[order[r]];
    for (let i = 0; i < n; i++) sm[i * n + r] = v[i * n + order[r]];
  }
  return { values: sv, vectors: sm };
}

/**
 * Subspace iteration for the k smallest eigenpairs of the combinatorial
 * Laplacian L = D - W. Iterates X <- (cI - L)^2 X where c = 2*max(deg) is a
 * Gershgorin bound on the largest eigenvalue (squaring widens the spectral
 * gaps), with Gram-Schmidt orthonormalization each step. The trivial constant
 * eigenvector is deflated analytically. Deterministic: fixed start basis,
 * fixed steps. Returns Laplacian eigenvalues ascending and eigenvectors as
 * columns of an n*k row-major array.
 */
function smallestLaplacianEigen(L: Float64Array, n: number, k: number): { values: Float64Array; vectors: Float64Array } {
  const kk = Math.min(k, n);
  // deterministic start basis
  let X: Float64Array = new Float64Array(n * kk);
  for (let j = 0; j < kk; j++) {
    for (let i = 0; i < n; i++) X[i * kk + j] = Math.sin((i + 1) * (j + 1) * 2.399963) + Math.cos((i + 1) * (j + 2) * 1.7);
  }

  const mul = (A: Float64Array, Yin: Float64Array): Float64Array => {
    const Y = new Float64Array(n * kk);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < kk; j++) {
        let s = 0;
        for (let p = 0; p < n; p++) s += A[i * n + p] * Yin[p * kk + j];
        Y[i * kk + j] = s;
      }
    }
    return Y;
  };

  // A = cI - L with c = 2*max(deg) (Gershgorin bound on the largest eigenvalue)
  let maxDeg = 0;
  for (let i = 0; i < n; i++) maxDeg = Math.max(maxDeg, L[i * n + i]);
  const c = 2 * maxDeg + 1e-9;
  const A = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) A[i] = -L[i];
  for (let i = 0; i < n; i++) A[i * n + i] += c;

  // Analytically deflate the trivial constant eigenvector (exact null vector
  // of the combinatorial Laplacian); without this the Fiedler vector is nearly
  // degenerate with it under cI - L and subspace iteration cannot separate them.
  const v0 = 1 / Math.sqrt(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] -= c * v0 * v0;

  const orthonormalize = (Y: Float64Array): void => {
    for (let j = 0; j < kk; j++) {
      for (let p = 0; p < j; p++) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += Y[i * kk + j] * Y[i * kk + p];
        for (let i = 0; i < n; i++) Y[i * kk + j] -= dot * Y[i * kk + p];
      }
      let norm = 0;
      for (let i = 0; i < n; i++) norm += Y[i * kk + j] * Y[i * kk + j];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) Y[i * kk + j] /= norm;
    }
  };

  const steps = n <= 16 ? 64 : 128;
  for (let s = 0; s < steps; s++) {
    X = mul(A, mul(A, X));
    orthonormalize(X);
  }

  // Rayleigh quotient per column against L, sort ascending
  const lambdas = new Float64Array(kk);
  const LX = mul(L, X);
  for (let j = 0; j < kk; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i * kk + j] * LX[i * kk + j];
    lambdas[j] = s;
  }
  const order = Array.from({ length: kk }, (_, i) => i).sort((a, b) => lambdas[a] - lambdas[b] || a - b);
  const out = new Float64Array(n * kk);
  const sortedLambdas = new Float64Array(kk);
  for (let r = 0; r < kk; r++) {
    sortedLambdas[r] = lambdas[order[r]];
    for (let i = 0; i < n; i++) out[i * kk + r] = X[i * kk + order[r]];
  }
  return { values: sortedLambdas, vectors: out };
}

/**
 * Spectral embedding of a similarity matrix into 3D.
 * Uses the normalized-Laplacian eigenvectors just above the trivial one.
 * Weights are sharpened (s^gamma) so near-saturated similarities still carry
 * structure. Sign of each axis is fixed deterministically (largest-|entry|
 * component positive) so the same thought always yields the same sculpture.
 */
export function spectralEmbed(similarity: Float32Array, n: number, outDims = 3, gamma = 4, floor = 0.02): Float32Array {
  const positions = new Float32Array(n * outDims);
  if (n === 0) return positions;
  if (n === 1) return positions; // single token at origin

  // W = clamp negative similarities to zero, sharpen
  const w = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) w[i] = Math.max(0, similarity[i]) ** gamma;
  for (let i = 0; i < n; i++) w[i * n + i] = 0;

  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += w[i * n + j];

  // Combinatorial Laplacian L = D - W. With the eigenvalue weighting below,
  // embedding distances approximate effective-resistance (commute) distance on
  // the association graph: bottleneck-separated tokens land far apart.
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    L[i * n + j] = i === j ? deg[i] : -w[i * n + j];
  }

  const kk = Math.min(outDims, n);
  const { values, vectors } = smallestLaplacianEigen(L, n, kk);
  // Columns 0..outDims-1: the trivial eigenvector was deflated analytically,
  // so the returned set starts at the Fiedler vector.
  // Eigenvalue weighting: small-eigenvalue axes carry the strong community
  // structure; scaling by 1/sqrt(lambda) (floored) keeps within-cluster noise
  // axes from dominating the sculpture at small token counts.
  for (let d = 0; d < outDims; d++) {
    const col = Math.min(d, kk - 1);
    const wScale = 1 / Math.sqrt(Math.max(values[col], floor));
    // deterministic sign
    let maxIdx = 0;
    for (let i = 1; i < n; i++) if (Math.abs(vectors[i * kk + col]) > Math.abs(vectors[maxIdx * kk + col])) maxIdx = i;
    const sign = vectors[maxIdx * kk + col] >= 0 ? 1 : -1;
    for (let i = 0; i < n; i++) positions[i * outDims + d] = sign * wScale * vectors[i * kk + col];
  }

  // center + normalize to unit RMS radius so layers are visually comparable
  for (let d = 0; d < outDims; d++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += positions[i * outDims + d];
    mean /= n;
    for (let i = 0; i < n; i++) positions[i * outDims + d] -= mean;
  }
  let rms = 0;
  for (let i = 0; i < n * outDims; i++) rms += positions[i] * positions[i];
  rms = Math.sqrt(rms / (n * outDims));
  if (rms > 1e-12) for (let i = 0; i < n * outDims; i++) positions[i] /= rms;

  return positions;
}

/**
 * Least-squares 3x3 transform T (row-major Float64Array(9)) carrying `from`
 * onto `to` (both n*3 row-major, assumed centered). Horn's quaternion method
 * via the shared Jacobi eigensolver, searched over all 8 axis-flip
 * conventions of `from` — per-layer spectral embeddings differ by arbitrary
 * rotations *and* reflections, and chaining the best transform per layer
 * makes the same token trace a continuous path through the layer stack.
 * Deterministic: fixed flip order, strict-less comparison, quaternion sign
 * fixed by largest-|component| positive.
 */
export function procrustesAlign(from: Float32Array, to: Float32Array, n: number): Float64Array {
  const horn = (f: Float32Array): Float64Array => {
    // cross-covariance M[i][j] = sum_k f[k][i] * to[k][j]  (from ⊗ to)
    const m = new Float64Array(9);
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < 3; i++) {
        const t = f[k * 3 + i];
        for (let j = 0; j < 3; j++) m[i * 3 + j] += t * to[k * 3 + j];
      }
    }
    const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
    const k4 = new Float64Array([
      m00 + m11 + m22, m12 - m21, m20 - m02, m01 - m10,
      m12 - m21, m00 - m11 - m22, m01 + m10, m02 + m20,
      m20 - m02, m01 + m10, -m00 + m11 - m22, m12 + m21,
      m01 - m10, m02 + m20, m12 + m21, -m00 - m11 + m22,
    ]);
    const { vectors } = jacobiEigen(k4, 4);
    // largest eigenvalue = last column (ascending order)
    let qw = vectors[3];
    let qx = vectors[7];
    let qy = vectors[11];
    let qz = vectors[15];
    const norm = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;
    const comps = [qw, qx, qy, qz];
    let maxI = 0;
    for (let i = 1; i < 4; i++) if (Math.abs(comps[i]) > Math.abs(comps[maxI])) maxI = i;
    if (comps[maxI] < 0) { qw = -qw; qx = -qx; qy = -qy; qz = -qz; }
    return new Float64Array([
      1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
      2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
      2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
    ]);
  };

  const apply = (f: Float32Array, t: Float64Array, out: Float64Array): void => {
    for (let k = 0; k < n; k++) {
      const x = f[k * 3];
      const y = f[k * 3 + 1];
      const z = f[k * 3 + 2];
      out[k * 3] = t[0] * x + t[1] * y + t[2] * z;
      out[k * 3 + 1] = t[3] * x + t[4] * y + t[5] * z;
      out[k * 3 + 2] = t[6] * x + t[7] * y + t[8] * z;
    }
  };

  const flipped = new Float32Array(n * 3);
  const aligned = new Float64Array(n * 3);
  let bestT: Float64Array | null = null;
  let bestResidual = Infinity;
  for (let flip = 0; flip < 8; flip++) {
    const sx = flip & 1 ? -1 : 1;
    const sy = flip & 2 ? -1 : 1;
    const sz = flip & 4 ? -1 : 1;
    for (let k = 0; k < n; k++) {
      flipped[k * 3] = from[k * 3] * sx;
      flipped[k * 3 + 1] = from[k * 3 + 1] * sy;
      flipped[k * 3 + 2] = from[k * 3 + 2] * sz;
    }
    const r = horn(flipped);
    // full transform: T = R * diag(sx, sy, sz)
    const t = new Float64Array(9);
    for (let i = 0; i < 3; i++) {
      t[i * 3] = r[i * 3] * sx;
      t[i * 3 + 1] = r[i * 3 + 1] * sy;
      t[i * 3 + 2] = r[i * 3 + 2] * sz;
    }
    apply(from, t, aligned);
    let residual = 0;
    for (let k = 0; k < n * 3; k++) residual += (aligned[k] - to[k]) ** 2;
    if (residual < bestResidual) {
      bestResidual = residual;
      bestT = t;
    }
  }
  return bestT!;
}

/** Top-k per token above a floor, deduplicated, sorted for determinism. */
export function selectEdges(similarity: Float32Array, n: number, topK = 3, floor = 0.2): Array<[number, number, number]> {
  const seen = new Set<number>();
  const edges: Array<[number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    const cands: Array<[number, number]> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const s = similarity[i * n + j];
      if (s > floor) cands.push([j, s]);
    }
    cands.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    for (const [j, s] of cands.slice(0, topK)) {
      const key = i < j ? i * n + j : j * n + i;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push(i < j ? [i, j, s] : [j, i, s]);
    }
  }
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return edges;
}

/** Full per-layer geometry from a hidden-state matrix [n x dims].
 *  Uses raw (uncentered) cosine similarity: for this model the raw sims are
 *  well-spread (~0.1..0.9), while centering forces most pairs negative and
 *  guts the association graph. */
export function layerGeometry(hidden: Float32Array, n: number, dims: number): LayerGeometry {
  const similarity = cosineSimilarity(hidden, n, dims);
  const salience = rowNorms(hidden, n, dims);
  const positions = spectralEmbed(similarity, n);
  const edges = selectEdges(similarity, n);
  return { positions, similarity, salience, edges };
}
