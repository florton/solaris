// Inference worker: owns the encoder session off the main thread. If a backend
// (looking at you, webgpu on a flaky driver) blocks synchronously, the main
// thread survives to terminate us and retry with wasm. The dreamer lives in
// its own worker (dream-worker.ts).
import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;
// ORT runs one session.run at a time per module ("Session already started"):
// overlapping thoughts queue here instead of crashing each other
let running: Promise<void> = Promise.resolve();

async function forward(reqId: number, ids: number[], keep: number[], outputs: string[]): Promise<void> {
  const seq = ids.length;
  const t0 = performance.now();
  const out = await session!.run({
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
  self.postMessage({ type: 'result', reqId, forwardMs, dims, layers }, { transfer: transfers });
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
    } catch (err) {
      self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
    return;
  }
  if (msg.type === 'forward' && session) {
    const { reqId, ids, keep, outputs } = msg as { reqId: number; ids: number[]; keep: number[]; outputs: string[] };
    running = running
      .then(() => forward(reqId, ids, keep, outputs))
      .catch((err) => self.postMessage({ type: 'forward-failed', reqId, message: err instanceof Error ? err.message : String(err) }));
  }
};
