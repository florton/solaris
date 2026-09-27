// Dream worker: owns the dreamer (dream.ts) — the decoder session and the form
// library — in its own ORT instance (ORT's wasm module runs one session at a
// time, so it stays apart from the encoder's worker). Loading is staged so the
// first thought never waits on the whole library:
//   1. decoder + caption index + captions  -> 'ready', dreams can run
//   2. the latents file streams in behind; until it lands, a dream fetches
//      just its 7 rows with HTTP Range requests (1.5 kB each)
// Dreams run strictly one after another (FIFO); a cancelled one bails between
// layers. Grids go out as half floats, ready to upload.
import * as ort from 'onnxruntime-web';
import { fullSdf, libraryLatent, meltAmount, pickForms, similarities, type DreamMeta } from './dream.ts';
import { f32ToF16 } from './half.ts';

interface Dreamer {
  meta: DreamMeta;
  base: string;
  decoder: ort.InferenceSession;
  emb: Int8Array; // [n, hiddenDim]
  lat: Int8Array | null; // [n, 3·8·8·8], once fully downloaded
  captions: string[];
}
let dreamer: Dreamer | null = null;
let loaded: Promise<void> = Promise.resolve(); // dreams queue behind the load
let queue: Promise<void> = Promise.resolve();
const cancelled = new Set<number>();

/** Fetch files, reporting combined download progress under `stage`. */
async function fetchAll(base: string, files: string[], stage: string): Promise<ArrayBuffer[]> {
  const got = new Array(files.length).fill(0);
  const totals = new Array(files.length).fill(0);
  let lastPost = 0;
  const report = (force = false) => {
    const now = performance.now();
    if (!force && now - lastPost < 150) return;
    lastPost = now;
    const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
    self.postMessage({ type: 'progress', stage, loaded: sum(got), total: sum(totals) });
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
        got[i] += value.length;
        report();
      }
      if (!totals[i]) totals[i] = got[i];
      report(true);
      const out = new Uint8Array(got[i]);
      let o = 0;
      for (const c of chunks) {
        out.set(c, o);
        o += c.length;
      }
      return out.buffer;
    }),
  );
}

async function load(base: string, meta: DreamMeta, ortBase: string, threads: number): Promise<void> {
  try {
    ort.env.wasm.wasmPaths = ortBase;
    // wasm even with WebGPU around: ORT's WebGPU Conv3D is a naive kernel.
    // threads (the main thread picks; see model.ts for the hang fallback)
    ort.env.wasm.numThreads = threads;
    const [decBuf, embBuf, capBuf] = await fetchAll(base, ['decoder.onnx', 'library_emb.bin', 'captions.json'], 'core');
    const decoder = await ort.InferenceSession.create(decBuf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const captions = JSON.parse(new TextDecoder().decode(capBuf)) as string[];
    dreamer ={ meta, base, decoder, emb: new Int8Array(embBuf), lat: null, captions };
    self.postMessage({ type: 'ready', threads: ort.env.wasm.numThreads });
  } catch (err) {
    self.postMessage({ type: 'unavailable', message: err instanceof Error ? err.message : String(err) });
    return;
  }
  // the rest of the library streams in behind (it also lands in the offline cache)
  fetchAll(base, ['library_lat.bin'], 'library')
    .then(([buf]) => {
      dreamer!.lat = new Int8Array(buf);
      self.postMessage({ type: 'library', bytes: buf.byteLength });
    })
    .catch(() => {}); // Range requests keep working without it
}

/** The latent rows a dream needs: from the full library if it's in, else by Range. */
async function latents(d: Dreamer, rows: number[]): Promise<Float32Array[]> {
  if (d.lat) return rows.map((i) => libraryLatent(d.meta, d.lat!, i));
  const m = d.meta.latShape.reduce((a, b) => a * b, 1);
  return Promise.all(
    rows.map(async (i) => {
      const r = await fetch(d.base + 'library_lat.bin', { headers: { Range: `bytes=${i * m}-${(i + 1) * m - 1}` } });
      if (!r.ok) throw new Error(`library_lat.bin: ${r.status}`);
      const buf = new Int8Array(await r.arrayBuffer());
      // a server that ignores Range sends the whole file: keep it
      if (r.status === 200 && buf.length >= d.meta.n * m) {
        d.lat = buf;
        return libraryLatent(d.meta, buf, i);
      }
      return libraryLatent(d.meta, buf, 0);
    }),
  );
}

// wasm session.run computes synchronously; yield so cancels get seen
const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0));

/** One thought -> one form per layer (posted as each finishes), then the melt
 *  amount for every pair of neighbouring layers. `phraseQueries` (two or more,
 *  or none) are the thought's phrases; the ones with a good enough match are
 *  dreamt in order, else the whole thought is. */
async function dream(reqId: number, query: Float32Array, phraseQueries: Float32Array[], seed: number): Promise<void> {
  const d = dreamer!;
  const { meta } = d;
  const tStart = performance.now();
  let sims = phraseQueries
    .map((q) => similarities(meta, d.emb, q))
    .filter((s) => s.reduce((a, b) => Math.max(a, b), -Infinity) >= meta.dream.minSim)
    .slice(0, meta.dream.maxPhrases);
  if (sims.length <= 1) sims = [similarities(meta, d.emb, query)];
  const picks = pickForms(meta, d.emb, sims, seed);
  const zs = await latents(
    d,
    picks.map((p) => p.index),
  );
  const g = meta.grid;
  const h = (2 * meta.bound) / g; // world units per voxel
  const fields: Float32Array[] = [];
  const melts: number[] = [];
  for (let l = 0; l < meta.layers; l++) {
    await yieldToEvents();
    if (cancelled.has(reqId)) return;
    const out = await d.decoder.run({ z: new ort.Tensor('float32', zs[l], [1, ...meta.latShape]) });
    const full = fullSdf(out.sdf.data as Float32Array, g, h, meta.trunc);
    fields.push(full);
    if (l > 0) melts.push(meltAmount(fields[l - 1], full, meta));
    const vol = f32ToF16(full);
    const p = picks[l];
    self.postMessage({ type: 'layer', reqId, layer: l, vol, index: p.index, source: d.captions[p.index], sim: p.sim }, { transfer: [vol.buffer] });
  }
  self.postMessage({ type: 'done', reqId, ms: performance.now() - tStart, melts });
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'load') {
    loaded = load(msg.base, msg.meta, msg.ortBase, msg.threads);
  } else if (msg.type === 'cancel') {
    cancelled.add(msg.reqId);
  } else if (msg.type === 'dream') {
    const { reqId, seed } = msg as { reqId: number; seed: number };
    const query = new Float32Array(msg.query);
    const phraseQueries = ((msg.phraseQueries ?? []) as ArrayBuffer[]).map((b) => new Float32Array(b));
    queue = queue.then(async () => {
      try {
        await loaded;
        if (!dreamer) throw new Error('imagination unavailable');
        if (!cancelled.has(reqId)) await dream(reqId, query, phraseQueries, seed);
      } catch (err) {
        self.postMessage({ type: 'failed', reqId, message: err instanceof Error ? err.message : String(err) });
      } finally {
        cancelled.delete(reqId);
      }
    });
  }
};
