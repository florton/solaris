// Inference worker: owns the ORT sessions off the main thread. If a backend
// (looking at you, webgpu on a flaky driver) blocks synchronously, the main
// thread survives to terminate us and retry with wasm.
// Two sessions: the encoder (thought -> hidden states) and the imagination
// decoder (latent -> SDF grid). The decoder is optional: if it fails to load,
// the piece simply stays a cloud.
import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;
let imagination: ort.InferenceSession | null = null;
let latentDim = 24;
const coordsCache = new Map<number, Float32Array>();
let activeMatId = 0;

function gridCoords(g: number, bound: number): Float32Array {
  let c = coordsCache.get(g);
  if (c) return c;
  c = new Float32Array(g * g * g * 3);
  let o = 0;
  for (let i = 0; i < g; i++) {
    const x = (i / (g - 1)) * 2 * bound - bound;
    for (let j = 0; j < g; j++) {
      const y = (j / (g - 1)) * 2 * bound - bound;
      for (let k = 0; k < g; k++) {
        c[o++] = x;
        c[o++] = y;
        c[o++] = (k / (g - 1)) * 2 * bound - bound;
      }
    }
  }
  coordsCache.set(g, c);
  return c;
}

async function loadImagination(url: string, backend: string): Promise<void> {
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    if (!buf.byteLength) throw new Error('empty decoder');
    imagination = await ort.InferenceSession.create(buf, {
      executionProviders: [backend],
      graphOptimizationLevel: 'all',
    });
    self.postMessage({ type: 'imagination-ready', backend });
  } catch (err) {
    self.postMessage({ type: 'imagination-unavailable', message: err instanceof Error ? err.message : String(err) });
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'load') {
    try {
      ort.env.wasm.wasmPaths = msg.ortBase;
      ort.env.wasm.numThreads = 1;
      const t0 = performance.now();
      const buf = await (await fetch(msg.modelUrl)).arrayBuffer();
      self.postMessage({ type: 'status', msg: `weights in (${(buf.byteLength / 1e6).toFixed(0)} MB), compiling ${msg.backend}…` });
      session = await ort.InferenceSession.create(buf, {
        executionProviders: [msg.backend],
        graphOptimizationLevel: 'all',
      });
      self.postMessage({ type: 'ready', backend: msg.backend, loadMs: performance.now() - t0, workerUrl: self.location.href });
      if (msg.imaginationUrl) {
        latentDim = msg.latentDim ?? latentDim;
        self.postMessage({ type: 'status', msg: 'growing imagination…' });
        void loadImagination(msg.imaginationUrl, msg.backend);
      }
    } catch (err) {
      self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (msg.type === 'materialize') {
    activeMatId = msg.reqId;
    if (!imagination) {
      self.postMessage({ type: 'materialize-unavailable', reqId: msg.reqId });
      return;
    }
    const { reqId, grid, bound } = msg as { reqId: number; grid: number; bound: number };
    const latents = new Float32Array(msg.latents);
    const nLayers = latents.length / latentDim;
    const coords = gridCoords(grid, bound);
    const m = grid * grid * grid;
    const t0 = performance.now();
    for (let l = 0; l < nLayers; l++) {
      if (reqId !== activeMatId) return; // superseded by a newer thought
      const z = latents.slice(l * latentDim, (l + 1) * latentDim);
      const out = await imagination.run({
        coords: new ort.Tensor('float32', coords, [m, 3]),
        z: new ort.Tensor('float32', z, [1, latentDim]),
      });
      const sdf = new Float32Array(out.sdf.data as Float32Array);
      self.postMessage({ type: 'form-layer', reqId, layer: l, grid, sdf }, { transfer: [sdf.buffer] });
    }
    self.postMessage({ type: 'materialize-done', reqId, ms: performance.now() - t0, grid });
    return;
  }
  if (msg.type === 'forward' && session) {
    const { ids, keep, outputs } = msg as {
      ids: number[];
      keep: number[];
      outputs: string[];
    };
    const seq = ids.length;
    const t0 = performance.now();
    const out = await session.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, seq]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(seq).fill(1n), [1, seq]),
    });
    const forwardMs = performance.now() - t0;
    const dims = out[outputs[0]].dims[2];
    const n = keep.length;
    const layers: Float32Array[] = [];
    const transfers: ArrayBuffer[] = [];
    for (const name of outputs) {
      const data = out[name].data as Float32Array;
      const rows = new Float32Array(n * dims);
      keep.forEach((tokIdx, r) => rows.set(data.slice(tokIdx * dims, (tokIdx + 1) * dims), r * dims));
      layers.push(rows);
      transfers.push(rows.buffer);
    }
    self.postMessage({ type: 'result', forwardMs, dims, layers }, { transfer: transfers });
  }
};
