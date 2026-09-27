// Model runtime (main-thread side): tokenizes, delegates inference to
// worker.ts. Backend ladder: webgpu -> wasm, with a watchdog that survives
// even a synchronously-blocked GPU backend (the worker gets terminated).
// Also starts the dreamer in its own worker (dream-worker.ts): pooled
// per-layer embeddings -> forms.
import { WordPieceTokenizer } from './tokenizer.ts';
import { poolEmbedding } from './bridge.ts';
import { fnv1a, phrases, type DreamMeta } from './dream.ts';

export interface ModelMeta {
  id: string;
  params: number;
  layers: number;
  hiddenDim: number;
  dtype: string;
  outputs: string[];
}

export interface ForwardResult {
  tokens: string[]; // display words (wordpiece fragments merged), specials removed
  hiddenStates: Float32Array[]; // per layer: [nTokens * hiddenDim], specials removed
  pooled: Float32Array[]; // per layer: unit-norm mean-pooled embedding [hiddenDim]
  nLayers: number;
  hiddenDim: number;
  forwardMs: number;
}

export type DreamStatus =
  | { state: 'off' } // no dream_model.json: the piece stays a cloud
  | { state: 'loading'; loaded: number; total: number }
  | { state: 'ready'; threads: number; library: boolean } // library: the full latents file is in
  | { state: 'unavailable'; message: string };

export interface DreamHandlers {
  onLayer: (layer: number, vol: Uint16Array, source: string, index: number) => void; // half-float full SDF, x fastest; index: library form
  onDone: (ms: number, melts: number[]) => void; // melts[l]: inflation of the layer l -> l+1 morph
  onFail: (message: string) => void;
}

const GPU_TIMEOUT_MS = 15000;

// Timer workers are exempt from background-tab throttling; a main-thread
// setTimeout in a hidden tab can be delayed for minutes.
function startWatchdog(ms: number, onTimeout: () => void): () => void {
  const url = URL.createObjectURL(new Blob([`setTimeout(() => self.postMessage(0), ${ms})`], { type: 'text/javascript' }));
  const timer = new Worker(url);
  timer.onmessage = () => onTimeout();
  return () => {
    timer.terminate();
    URL.revokeObjectURL(url);
  };
}

function spawnAndLoad(
  modelUrl: string,
  ortBase: string,
  backend: string,
  onStatus: (m: string) => void,
): Promise<{ worker: Worker; workerUrl: string }> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    let settled = false;
    const cancelWatchdog = startWatchdog(GPU_TIMEOUT_MS, () => {
      if (settled) return;
      settled = true;
      worker.terminate();
      reject(new Error(`${backend} timed out`));
    });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'status') onStatus(m.msg);
      if (m.type === 'ready' && !settled) {
        settled = true;
        cancelWatchdog();
        resolve({ worker, workerUrl: m.workerUrl ?? '' });
      }
      if (m.type === 'error' && !settled) {
        settled = true;
        cancelWatchdog();
        worker.terminate();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (e) => {
      if (settled) return;
      settled = true;
      cancelWatchdog();
      worker.terminate();
      reject(new Error(e.message ?? 'worker error'));
    };
    worker.postMessage({ type: 'load', modelUrl, ortBase, backend });
  });
}

/** The dream worker, restartable. ORT-web's wasm thread pool can hang for
 *  good while creating a session (seen in the in-app browser: 1 thread ready
 *  in 0.5 s, 2+ threads never). Threads make the decoder ~3× faster, so they
 *  are tried first; if the worker isn't ready 4 s after its download, it is
 *  replaced by a single-threaded one and everything sent meanwhile is replayed. */
class DreamHost {
  private worker!: Worker;
  private ready = false;
  private early: unknown[] = []; // messages before adopt()
  private sink: ((m: any) => void) | null = null;
  private held: { msg: unknown; transfer: Transferable[] }[] = []; // sent before 'ready'

  constructor(
    public meta: DreamMeta,
    private base: string,
    private ortBase: string,
  ) {
    this.spawn(self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
  }

  private deliver(m: unknown): void {
    if (this.sink) this.sink(m);
    else this.early.push(m);
  }

  private spawn(threads: number): void {
    const w = new Worker(new URL('./dream-worker.ts', import.meta.url), { type: 'module' });
    this.worker = w;
    let cancelWatchdog: (() => void) | null = null;
    w.onmessage = (e) => {
      if (w !== this.worker) return;
      const m = e.data;
      if (m.type === 'ready') {
        this.ready = true;
        cancelWatchdog?.();
        for (const h of this.held) w.postMessage(h.msg, h.transfer);
        this.held = [];
      }
      if (threads > 1 && !cancelWatchdog && m.type === 'progress' && m.stage === 'core' && m.total && m.loaded >= m.total) {
        cancelWatchdog = startWatchdog(4000, () => {
          if (this.ready || w !== this.worker) return;
          console.warn(`dreamer: ${threads}-thread wasm session never came up; retrying on 1 thread`);
          w.terminate();
          this.spawn(1);
        });
      }
      this.deliver(m);
    };
    w.onerror = (e) => this.deliver({ type: 'unavailable', message: e.message ?? 'dream worker error' });
    w.postMessage({ type: 'load', base: this.base, meta: this.meta, ortBase: this.ortBase, threads });
  }

  adopt(sink: (m: any) => void): void {
    this.sink = sink;
    for (const m of this.early) sink(m);
    this.early = [];
  }

  /** Dreams and cancels: held until the worker is ready, so a respawn loses nothing. */
  postMessage(msg: unknown, transfer: Transferable[] = []): void {
    if (this.ready) this.worker.postMessage(msg, transfer);
    else this.held.push({ msg, transfer });
  }
}

export class SolarisModel {
  private forwards = new Map<number, { resolve: (r: { forwardMs: number; dims: number; layers: Float32Array[] }) => void; reject: (e: Error) => void }>();
  private forwardCounter = 0;
  private dreams = new Map<number, DreamHandlers>();
  private dreamCounter = 0;
  private encoding = new Set<number>(); // dreams whose phrases are still being encoded
  private dreamWorker: DreamHost | null = null;
  public dream: DreamMeta | null = null;
  public dreamStatus: DreamStatus = { state: 'off' };
  public onDreamStatus: (s: DreamStatus) => void = () => {};

  private constructor(
    public meta: ModelMeta,
    private tokenizer: WordPieceTokenizer,
    private worker: Worker,
    public backend: string,
    public loadMs: number,
    public workerUrl: string,
  ) {
    this.worker.onmessage = (e) => {
      const m = e.data;
      const pending = this.forwards.get(m.reqId);
      if (!pending) return;
      this.forwards.delete(m.reqId);
      if (m.type === 'result') pending.resolve(m);
      else if (m.type === 'forward-failed') pending.reject(new Error(m.message));
    };
  }

  static async load(baseUrl: string, onStatus: (msg: string) => void): Promise<SolarisModel> {
    const root = new URL(baseUrl, document.baseURI).href;
    const modelBase = new URL('models/minilm-l6/', root).href;
    onStatus('reading model card…');
    const meta: ModelMeta = await (await fetch(`${modelBase}solaris_model.json`)).json();

    onStatus('loading vocabulary…');
    const tokenizer = await WordPieceTokenizer.load(`${modelBase}vocab.txt`);

    const modelUrl = `${modelBase}onnx/model_quantized.onnx`;
    const ortBase = new URL('ort/', root).href;
    // the imagination is optional (no card: no forms, no error) and downloads
    // alongside the encoder rather than after it
    const dreamer = SolarisModel.startDreamer(new URL('models/dream/', root).href, ortBase);
    const forced = new URLSearchParams(location.search).get('backend');
    // wasm unless asked: ORT-web's WebGPU kernels for this q8 graph return
    // hidden states unrelated to the real ones (cosine ~0 vs wasm/PyTorch on
    // every layer), so both the cloud and the dream retrieval were noise.
    // The encoder is small; wasm runs a thought in well under 100 ms.
    const wantGpu = forced === 'webgpu';

    let spawned: { worker: Worker; workerUrl: string } | null = null;
    let backend = 'webgpu';
    const t0 = performance.now();
    if (wantGpu) {
      try {
        onStatus('warming up webgpu…');
        spawned = await spawnAndLoad(modelUrl, ortBase, 'webgpu', onStatus);
      } catch {
        onStatus('webgpu unavailable — falling back to wasm…');
      }
    }
    if (!spawned) {
      backend = 'wasm';
      onStatus('warming up wasm…');
      spawned = await spawnAndLoad(modelUrl, ortBase, 'wasm', onStatus);
    }
    const m = new SolarisModel(meta, tokenizer, spawned.worker, backend, performance.now() - t0, spawned.workerUrl);
    m.adoptDreamer(await dreamer);
    return m;
  }

  private setDreamStatus(s: DreamStatus): void {
    this.dreamStatus = s;
    this.onDreamStatus(s);
  }

  private static async startDreamer(base: string, ortBase: string): Promise<DreamHost | null> {
    try {
      const r = await fetch(`${base}dream_model.json`);
      if (!r.ok) return null;
      return new DreamHost((await r.json()) as DreamMeta, base, ortBase);
    } catch {
      return null;
    }
  }

  private adoptDreamer(d: DreamHost | null): void {
    if (!d) return;
    this.dream = d.meta;
    this.dreamWorker = d;
    this.setDreamStatus({ state: 'loading', loaded: 0, total: 0 });
    d.adopt((m) => {
      if (m.type === 'progress') {
        if (m.stage === 'core') this.setDreamStatus({ state: 'loading', loaded: m.loaded, total: m.total });
      } else if (m.type === 'ready') this.setDreamStatus({ state: 'ready', threads: m.threads, library: false });
      else if (m.type === 'library') {
        if (this.dreamStatus.state === 'ready') this.setDreamStatus({ ...this.dreamStatus, library: true });
      } else if (m.type === 'unavailable') this.setDreamStatus({ state: 'unavailable', message: m.message });
      else {
        const h = this.dreams.get(m.reqId);
        if (!h) return; // cancelled
        if (m.type === 'layer') h.onLayer(m.layer, m.vol, m.source, m.index);
        else {
          this.dreams.delete(m.reqId);
          if (m.type === 'done') h.onDone(m.ms, m.melts);
          else if (m.type === 'failed') h.onFail(m.message);
        }
      }
    });
  }

  async forward(text: string): Promise<ForwardResult> {
    const { tokens, ids } = this.tokenizer.encode(text, 128);
    const keep: number[] = [];
    for (let i = 0; i < ids.length; i++) if (!/^\[.*\]$/.test(tokens[i])) keep.push(i);
    const reqId = ++this.forwardCounter;
    const result = await new Promise<{ forwardMs: number; dims: number; layers: Float32Array[] }>((resolve, reject) => {
      this.forwards.set(reqId, { resolve, reject });
      this.worker.postMessage({ type: 'forward', reqId, ids, keep, outputs: this.meta.outputs });
    });
    // merge wordpiece fragments into words for display: 'card','##amo','##mon' -> 'cardamom'
    const kept = keep.map((i) => tokens[i]);
    const words: string[] = [];
    let word = '';
    for (const t of kept) {
      if (t.startsWith('##')) word += t.slice(2);
      else word = t;
      words.push(word);
    }
    const n = keep.length;
    return {
      tokens: words,
      hiddenStates: result.layers,
      pooled: result.layers.map((h) => poolEmbedding(h, n, result.dims)),
      nLayers: result.layers.length,
      hiddenDim: result.dims,
      forwardMs: result.forwardMs,
    };
  }

  /** Queue a dream of the thought (the worker runs them in order and holds
   *  them until it has loaded). Returns an id for cancelDream, or null when
   *  the piece has no imagination. */
  requestDream(text: string, result: ForwardResult, handlers: DreamHandlers): number | null {
    if (!this.dreamWorker || this.dreamStatus.state === 'unavailable') return null;
    const reqId = ++this.dreamCounter;
    this.dreams.set(reqId, handlers);
    this.encoding.add(reqId);
    void this.sendDream(reqId, text, result);
    return reqId;
  }

  /** Encode the thought's phrases (when it splits into two or more), then hand
   *  the dream to the worker. A failed phrase encode dreams the whole thought. */
  private async sendDream(reqId: number, text: string, result: ForwardResult): Promise<void> {
    const layer = this.dream!.embLayer;
    const query = result.pooled[layer].slice();
    const split = phrases(text, this.dream!.dream);
    let phraseQueries: Float32Array[] = [];
    if (split.length > 1) {
      try {
        phraseQueries = (await Promise.all(split.map((p) => this.forward(p)))).map((r) => r.pooled[layer].slice());
      } catch {
        phraseQueries = [];
      }
    }
    if (!this.encoding.delete(reqId)) return; // cancelled while its phrases were encoding
    const buffers = [query.buffer, ...phraseQueries.map((q) => q.buffer)];
    this.dreamWorker!.postMessage(
      { type: 'dream', reqId, query: query.buffer, phraseQueries: phraseQueries.map((q) => q.buffer), seed: fnv1a(text) },
      buffers,
    );
  }

  cancelDream(reqId: number): void {
    if (!this.dreams.delete(reqId)) return;
    if (this.encoding.delete(reqId)) return; // never reached the worker
    this.dreamWorker?.postMessage({ type: 'cancel', reqId });
  }
}
