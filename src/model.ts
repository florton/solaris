// Model runtime (main-thread side): tokenizes, delegates inference to
// worker.ts. Backend ladder: webgpu -> wasm, with a watchdog that survives
// even a synchronously-blocked GPU backend (the worker gets terminated).
import { WordPieceTokenizer } from './tokenizer.ts';

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

function spawnAndLoad(modelUrl: string, ortBase: string, backend: string, onStatus: (m: string) => void): Promise<{ worker: Worker; workerUrl: string }> {
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

export class SolarisModel {
  private callbacks = new Map<number, (r: { forwardMs: number; dims: number; layers: Float32Array[] }) => void>();

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

    const modelUrl = `${modelBase}onnx/model_quantized.onnx`;
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
    return new SolarisModel(meta, tokenizer, spawned.worker, backend, performance.now() - t0, spawned.workerUrl);
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
    return {
      tokens: words,
      hiddenStates: result.layers,
      nLayers: result.layers.length,
      hiddenDim: result.dims,
      forwardMs: result.forwardMs,
    };
  }
}
