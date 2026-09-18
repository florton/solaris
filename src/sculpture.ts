// Sculpture: forward-pass result -> 4D render data.
// Vertex per (layer, token): xyz from the layer's spectral embedding,
// w = position along the layer axis. Each layer's embedding is only defined
// up to rotation/reflection, so every layer is Procrustes-aligned to the
// previous one — the same token then traces a continuous path through the
// stack and layer-to-layer reorganization reads as morphing, not teleporting.
// Color is per token and stable across layers: hue comes from the token's
// angular position in the final layer's embedding, so tokens the model holds
// together share color families and the inter-layer strands read as colored
// threads running through the depth axis.
// Edges: intra-layer association ribbons + inter-layer identity strands
// (same token, adjacent layers) so the stack reads as one object
// re-organizing through depth.
import { layerGeometry, procrustesAlign } from './geometry.ts';
import type { ForwardResult } from './model.ts';

export const EDGE_INTRA = 0;
export const EDGE_INTER = 1;

export interface Sculpture {
  tokens: string[];
  nTokens: number;
  nLayers: number;
  vertexCount: number;
  positions: Float32Array; // xyz per vertex (layer-major)
  wCoords: Float32Array; // 4th-axis coordinate per vertex
  brightness: Float32Array; // per-vertex base brightness (normalized salience)
  colors: Float32Array; // rgb per vertex, per-token hue
  tokenOf: Uint16Array; // vertex -> token index
  edgeIndices: Uint32Array; // 2 per edge
  edgeWeights: Float32Array; // 0..1 per edge
  edgeKind: Uint8Array; // EDGE_INTRA | EDGE_INTER per edge
  edgeCount: number;
  eigenMs: number;
}

const WORLD_RADIUS = 2.0;
const W_SPAN = 1.5; // half-extent of the layer axis
const INTER_LAYER_WEIGHT = 0.05;

function hsl2rgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (k: number) => {
    const t = (k + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(t - 3, 9 - t, 1));
  };
  return [f(0), f(8), f(4)];
}

export function buildSculpture(result: ForwardResult): Sculpture {
  const t0 = performance.now();
  const { tokens, hiddenStates, nLayers, hiddenDim } = result;
  const n = tokens.length;
  const vertexCount = nLayers * n;

  const positions = new Float32Array(vertexCount * 3);
  const wCoords = new Float32Array(vertexCount);
  const brightness = new Float32Array(vertexCount);
  const colors = new Float32Array(vertexCount * 3);
  const tokenOf = new Uint16Array(vertexCount);
  const edgeIdx: number[] = [];
  const edgeW: number[] = [];
  const edgeKind: number[] = [];

  let prevAligned: Float32Array | null = null;
  let lastPositions: Float32Array = new Float32Array(0);
  let lastSalience: Float32Array = new Float32Array(0);

  const wOf = (layer: number) => (nLayers <= 1 ? 0 : (layer / (nLayers - 1)) * 2 - 1) * W_SPAN;

  for (let l = 0; l < nLayers; l++) {
    const g = layerGeometry(hiddenStates[l], n, hiddenDim);
    lastSalience = g.salience;
    // chain-align each layer's embedding into the previous layer's frame
    let aligned = g.positions;
    if (prevAligned) {
      const t = procrustesAlign(g.positions, prevAligned, n);
      aligned = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const x = g.positions[i * 3];
        const y = g.positions[i * 3 + 1];
        const z = g.positions[i * 3 + 2];
        aligned[i * 3] = t[0] * x + t[1] * y + t[2] * z;
        aligned[i * 3 + 1] = t[3] * x + t[4] * y + t[5] * z;
        aligned[i * 3 + 2] = t[6] * x + t[7] * y + t[8] * z;
      }
    }
    prevAligned = aligned;
    lastPositions = aligned;
    // per-layer salience normalization
    let maxSal = 1e-9;
    for (let i = 0; i < n; i++) maxSal = Math.max(maxSal, g.salience[i]);
    for (let i = 0; i < n; i++) {
      const v = l * n + i;
      positions[v * 3] = aligned[i * 3] * WORLD_RADIUS;
      positions[v * 3 + 1] = aligned[i * 3 + 1] * WORLD_RADIUS;
      positions[v * 3 + 2] = aligned[i * 3 + 2] * WORLD_RADIUS;
      wCoords[v] = wOf(l);
      brightness[v] = 0.35 + 0.65 * (g.salience[i] / maxSal);
      tokenOf[v] = i;
    }
    for (const [i, j, w] of g.edges) {
      edgeIdx.push(l * n + i, l * n + j);
      edgeW.push(w);
      edgeKind.push(EDGE_INTRA);
    }
    // identity strands to the next layer
    if (l < nLayers - 1) {
      for (let i = 0; i < n; i++) {
        edgeIdx.push(l * n + i, (l + 1) * n + i);
        edgeW.push(INTER_LAYER_WEIGHT);
        edgeKind.push(EDGE_INTER);
      }
    }
  }

  // Per-token hue from the final layer's embedding: sort tokens by their
  // angular position and assign hues evenly by rank, so every sculpture spans
  // the full spectrum while tokens the model holds together (adjacent angles)
  // still get adjacent colors. The rainbow seam is placed in the largest
  // angular gap — the emptiest boundary between clusters.
  let maxSal = 1e-9;
  for (let i = 0; i < n; i++) maxSal = Math.max(maxSal, lastSalience[i]);
  const angles = new Float64Array(n);
  for (let i = 0; i < n; i++) angles[i] = Math.atan2(lastPositions[i * 3 + 1], lastPositions[i * 3]);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => angles[a] - angles[b] || a - b);
  let seam = 0;
  let maxGap = -1;
  for (let k = 0; k < n; k++) {
    const a0 = angles[order[k]];
    const a1 = angles[order[(k + 1) % n]] + (k === n - 1 ? 2 * Math.PI : 0);
    if (a1 - a0 > maxGap) {
      maxGap = a1 - a0;
      seam = k;
    }
  }
  const hues = new Float64Array(n);
  for (let t = 0; t < n; t++) hues[order[(seam + 1 + t) % n]] = n <= 1 ? 0.5 : t / (n - 1);
  for (let i = 0; i < n; i++) {
    const light = 0.56 + 0.12 * (lastSalience[i] / maxSal);
    const [r, g, b] = hsl2rgb(hues[i], 0.95, light);
    for (let l = 0; l < nLayers; l++) {
      const v = l * n + i;
      colors[v * 3] = r;
      colors[v * 3 + 1] = g;
      colors[v * 3 + 2] = b;
    }
  }

  return {
    tokens,
    nTokens: n,
    nLayers,
    vertexCount,
    positions,
    wCoords,
    brightness,
    colors,
    tokenOf,
    edgeIndices: Uint32Array.from(edgeIdx),
    edgeWeights: Float32Array.from(edgeW),
    edgeKind: Uint8Array.from(edgeKind),
    edgeCount: edgeW.length,
    eigenMs: performance.now() - t0,
  };
}
