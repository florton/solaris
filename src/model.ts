// Model runtime (main-thread side): tokenizes, delegates inference to
// worker.ts. Backend ladder: webgpu -> wasm, with a watchdog that survives
// even a synchronously-blocked GPU backend (the worker gets terminated).
// Also owns the bridge: pooled per-layer embeddings -> imagination latents.
import { WordPieceTokenizer } from './tokenizer.ts';
import { Bridge, poolEmbedding, type BridgeSpec } from './bridge.ts';

export interface ModelMeta {
  id: string;
  params: number;
  layers: number;
  hiddenDim: number;
  dtype: string;
  outputs: string[];
}

export interface ImaginationMeta extends BridgeSpec {
  bound: number;
  grid: { webgpu: number; wasm: number };
}

export interface ForwardResult {
  tokens: string[]; // display words (wordpiece fragments merged), specials removed
  hiddenStates: Float32Array[]; // per layer: [nTokens * hiddenDim], specials removed
  pooled: Float32Array[]; // per layer: unit-norm mean-pooled embedding [hiddenDim]
  nLayers: number;
  hiddenDim: number;
  forwardMs: number;
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
  imaginationUrl: string | null,
  latentDim: number,
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
    worker.postMessage({ type: 'load', modelUrl, ortBase, backend, imaginationUrl, latentDim });
  });
}

export class SolarisModel {
  private callbacks = new Map<number, (r: { forwardMs: number; dims: number; layers: Float32Array[] }) => void>();
  private matCallbacks = new Map<number, {
    onLayer: (layer: number, sdf: Float32Array, grid: number) => void;
    onDone: (ms: number, grid: number) => void;
  }>();
  private matCounter = 0;
  public bridge: Bridge | null = null;
  public imagination: ImaginationMeta | null = null;

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
      if (m.type === 'result') {
        const cb = this.callbacks.get(0);
        if (cb) {
          this.callbacks.delete(0);
          cb(m);
        }
      } else if (m.type === 'form-layer' || m.type === 'materialize-done' || m.type === 'materialize-unavailable') {
        const cb = this.matCallbacks.get(m.reqId);
        if (!cb) return; // stale request
        if (m.type === 'form-layer') cb.onLayer(m.layer, m.sdf, m.grid);
        else {
          this.matCallbacks.delete(m.reqId);
          if (m.type === 'materialize-done') cb.onDone(m.ms, m.grid);
          else cb.onDone(NaN, 0);
        }
      }
    };
  }

  static async load(baseUrl: string, onStatus: (msg: string) => void): Promise<SolarisModel> {
    const root = new URL(baseUrl, document.baseURI).href;
    const modelBase = new URL('models/minilm-l6/', root).href;
    onStatus('reading model card…');
    const meta: ModelMeta = await (await fetch(`${modelBase}solaris_model.json`)).json();

    onStatus('loading vocabulary…');
    const tokenizer = await WordPieceTokenizer.load(`${modelBase}vocab.txt`);

    // the imagination is optional: no card, no forms, no error
    let imagination: ImaginationMeta | null = null;
    const imaginationBase = new URL('models/imagination/', root).href;
    try {
      const r = await fetch(`${imaginationBase}imagination_model.json`);
      if (r.ok) imagination = (await r.json()) as ImaginationMeta;
    } catch {
      imagination = null;
    }

    const modelUrl = `${modelBase}onnx/model_quantized.onnx`;
    const imaginationUrl = imagination ? `${imaginationBase}decoder.onnx` : null;
    const ortBase = new URL('ort/', root).href;
    const forced = new URLSearchParams(location.search).get('backend');
    // hidden tab: no one is watching and GPU shader compile can block for
    // minutes under timer throttling — go straight to wasm
    const wantGpu = forced ? forced === 'webgpu' : !document.hidden && typeof navigator !== 'undefined' && 'gpu' in navigator;

    let spawned: { worker: Worker; workerUrl: string } | null = null;
    let backend = 'webgpu';
    const t0 = performance.now();
    if (wantGpu) {
      try {
        onStatus('warming up webgpu…');
        spawned = await spawnAndLoad(modelUrl, ortBase, 'webgpu', onStatus, imaginationUrl, imagination?.latentDim ?? 24);
      } catch {
        onStatus('webgpu unavailable — falling back to wasm…');
      }
    }
    if (!spawned) {
      backend = 'wasm';
      onStatus('warming up wasm…');
      spawned = await spawnAndLoad(modelUrl, ortBase, 'wasm', onStatus, imaginationUrl, imagination?.latentDim ?? 24);
    }
    const m = new SolarisModel(meta, tokenizer, spawned.worker, backend, performance.now() - t0, spawned.workerUrl);
    if (imagination) {
      m.imagination = imagination;
      m.bridge = new Bridge(imagination);
    }
    return m;
  }

  async forward(text: string): Promise<ForwardResult> {
    const { tokens, ids } = this.tokenizer.encode(text, 128);
    const keep: number[] = [];
    for (let i = 0; i < ids.length; i++) if (!/^\[.*\]$/.test(tokens[i])) keep.push(i);
    const result = await new Promise<{ forwardMs: number; dims: number; layers: Float32Array[] }>((resolve) => {
      this.callbacks.set(0, resolve);
      this.worker.postMessage({ type: 'forward', ids, keep, outputs: this.meta.outputs });
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

  /** Dream the thought into per-layer SDF grids. Null when the piece has no
   *  imagination (decoder absent) — the caller just stays a cloud. */
  materialize(
    result: ForwardResult,
    onLayer: (layer: number, sdf: Float32Array, grid: number) => void,
  ): Promise<{ ms: number; grid: number } | null> | null {
    if (!this.bridge || !this.imagination) return null;
    const latents = this.bridge.mapLayers(result.pooled);
    const reqId = ++this.matCounter;
    const grid = this.backend === 'webgpu' ? this.imagination.grid.webgpu : this.imagination.grid.wasm;
    return new Promise((resolve) => {
      this.matCallbacks.set(reqId, {
        onLayer,
        onDone: (ms, g) => resolve(Number.isNaN(ms) ? null : { ms, grid: g }),
      });
      this.worker.postMessage(
        { type: 'materialize', reqId, latents: latents.buffer.slice(0), grid, bound: this.imagination!.bound },
      );
    });
  }
}
