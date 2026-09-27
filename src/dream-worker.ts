// Dream worker: owns the dreamer (dream.ts) — prior + decoder sessions and the
// form library — in its own ORT instance. ORT's wasm module runs one session
// at a time ("Session already started"), so sharing the encoder's worker made a
// thought typed mid-dream kill the dream; apart, they also run in parallel.
// Dreams run strictly one after another (FIFO); a cancelled one bails at its
// next denoising step. Grids go out as half floats, ready to upload.
import * as ort from 'onnxruntime-web';
import { alphaSigma, dreamStream, libraryLatent, retrieve, type DreamMeta } from './dream.ts';
import { f32ToF16 } from './half.ts';

interface Dreamer {
  meta: DreamMeta;
  prior: ort.InferenceSession;
  decoder: ort.InferenceSession;
  lat: Int8Array; // [n, 3·8·8·8]
  emb: Int8Array; // [layers, n, hiddenDim]
  captions: string[];
}
let dreamer: Dreamer | null = null;
let loaded: Promise<void> = Promise.resolve(); // dreams queue behind the load
let queue: Promise<void> = Promise.resolve();
const cancelled = new Set<number>();

/** Fetch every file, reporting combined download progress. */
async function fetchAll(base: string, files: string[]): Promise<ArrayBuffer[]> {
  const loaded = new Array(files.length).fill(0);
  const totals = new Array(files.length).fill(0);
  let lastPost = 0;
  const report = (force = false) => {
    const now = performance.now();
    if (!force && now - lastPost < 150) return;
    lastPost = now;
    const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
    self.postMessage({ type: 'progress', loaded: sum(loaded), total: sum(totals) });
  };
  return Promise.all(
    files.map(async (f, i) => {
      const r = await fetch(base + f);
      if (!r.ok || !r.body) throw new Error(`${f}: ${r.status}`);
      totals[i] = Number(r.headers.get('content-length')) || 0;
      const reader = r.body.getReader();
      const chunks: Uint8Array[] = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded[i] += value.length;
        report();
      }
      if (!totals[i]) totals[i] = loaded[i];
      report(true);
      const out = new Uint8Array(loaded[i]);
      let o = 0;
      for (const c of chunks) {
        out.set(c, o);
        o += c.length;
      }
      return out.buffer;
    }),
  );
}

async function load(base: string, meta: DreamMeta, ortBase: string): Promise<void> {
  try {
    ort.env.wasm.wasmPaths = ortBase;
    // threads need SharedArrayBuffer (COOP/COEP, see vite.config.ts). wasm even
    // when the encoder is on webgpu: ORT's WebGPU Conv3D is a naive kernel,
    // several times slower than wasm for these 3D convs.
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const [priorBuf, decBuf, latBuf, embBuf, capBuf] = await fetchAll(base, [
      'prior.onnx',
      'decoder.onnx',
      'library_lat.bin',
      'library_emb.bin',
      'captions.json',
    ]);
    const create = (buf: ArrayBuffer) =>
      ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const prior = await create(priorBuf);
    const decoder = await create(decBuf);
    const captions = JSON.parse(new TextDecoder().decode(capBuf)) as string[];
    dreamer = { meta, prior, decoder, lat: new Int8Array(latBuf), emb: new Int8Array(embBuf), captions };
    self.postMessage({ type: 'ready', threads: ort.env.wasm.numThreads });
  } catch (err) {
    self.postMessage({ type: 'unavailable', message: err instanceof Error ? err.message : String(err) });
  }
}

// wasm session.run computes synchronously and resolves via microtasks, so a
// dream loop never returns to the event loop on its own: without this, cancel
// messages were only seen after the whole dream had finished
const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0));

/** One thought -> one form per layer, each posted as it finishes. */
async function dream(reqId: number, pooled: Float32Array, seed: number): Promise<void> {
  const { meta, prior, decoder, lat, emb, captions } = dreamer!;
  const { t0, steps, guidance } = meta.dream;
  const D = meta.hiddenDim;
  const shape = meta.latShape;
  const m = shape.reduce((a, b) => a * b, 1);
  const { picks, noise } = dreamStream(seed, meta.layers, m);
  const g = new ort.Tensor('float32', new Float32Array([guidance]), [1]);
  const tStart = performance.now();
  for (let l = 0; l < meta.layers; l++) {
    const q = pooled.subarray(l * D, (l + 1) * D);
    const pick = retrieve(meta, emb, l, q, picks[l]);
    const x0 = libraryLatent(meta, lat, pick.index);
    const [a0, s0] = alphaSigma(t0);
    const x = new Float32Array(m);
    for (let j = 0; j < m; j++) x[j] = a0 * x0[j] + s0 * noise[j];
    const embT = new ort.Tensor('float32', q.slice(), [1, D]);
    const layerT = new ort.Tensor('int64', BigInt64Array.from([BigInt(l)]), [1]);
    for (let i = 0; i < steps; i++) {
      await yieldToEvents(); // let a cancel (or the next request) in
      if (cancelled.has(reqId)) return;
      const t = t0 - (t0 * i) / steps;
      const tn = t0 - (t0 * (i + 1)) / steps;
      const out = await prior.run({
        x: new ort.Tensor('float32', x.slice(), [1, ...shape]),
        t: new ort.Tensor('float32', new Float32Array([t]), [1]),
        emb: embT,
        layer: layerT,
        guidance: g,
      });
      const v = out.v.data as Float32Array;
      const [a, s] = alphaSigma(t);
      const [an, sn] = alphaSigma(tn);
      for (let j = 0; j < m; j++) {
        const x0p = a * x[j] - s * v[j];
        const eps = s * x[j] + a * v[j];
        x[j] = an * x0p + sn * eps;
      }
    }
    if (cancelled.has(reqId)) return;
    const out = await decoder.run({ z: new ort.Tensor('float32', x, [1, ...shape]) });
    const vol = f32ToF16(out.sdf.data as Float32Array);
    self.postMessage({ type: 'layer', reqId, layer: l, vol, source: captions[pick.index] }, { transfer: [vol.buffer] });
  }
  self.postMessage({ type: 'done', reqId, ms: performance.now() - tStart });
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'load') {
    loaded = load(msg.base, msg.meta, msg.ortBase);
  } else if (msg.type === 'cancel') {
    cancelled.add(msg.reqId);
  } else if (msg.type === 'dream') {
    const { reqId, seed } = msg as { reqId: number; seed: number };
    const pooled = new Float32Array(msg.pooled);
    queue = queue.then(async () => {
      try {
        await loaded;
        if (!dreamer) throw new Error('imagination unavailable');
        if (!cancelled.has(reqId)) await dream(reqId, pooled, seed);
      } catch (err) {
        self.postMessage({ type: 'failed', reqId, message: err instanceof Error ? err.message : String(err) });
      } finally {
        cancelled.delete(reqId);
      }
    });
  }
};
